import os
import uuid
import re
import io
import PyPDF2
import pinecone
import google.generativeai as genai
import gradio as gr
from tqdm import tqdm
from dotenv import load_dotenv

# Load API keys from environment
load_dotenv()
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Configure Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API initialized successfully")
except Exception as e:
    print(f"❌ Gemini API init failed: {e}")
    exit(1)

# Configure Pinecone
try:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("chatbot")
    print("✅ Pinecone initialized successfully")
except Exception as e:
    print(f"❌ Pinecone init failed: {e}")
    exit(1)

# Set a session-specific namespace
SESSION_NAMESPACE = str(uuid.uuid4())
print(f"📝 Session namespace: {SESSION_NAMESPACE}")

# Store uploaded documents and chunks in memory for each session
uploaded_documents = {}  # {filename: {chunks: [], total_pages: int, upload_time: str}}
all_chunks = []  # Combined chunks from all documents

def clean_text(text):
    """Clean extracted text from PDF"""
    if not text:
        return ""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    lines = text.split('\n')
    return '\n'.join([line.strip() for line in lines if len(line.strip()) > 10])

def extract_text_from_pdf(file_bytes, filename):
    """Extract text from PDF bytes"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            print(f"📄 Extracting page {i+1} of {len(reader.pages)} from {filename}")
            text = page.extract_text()
            cleaned = clean_text(text)
            if cleaned:  # Only add non-empty pages
                pages.append({
                    "page_number": i + 1,
                    "text": cleaned,
                    "filename": filename
                })
        print(f"📄 Extracted {len(pages)} pages with content from {filename}")
        return pages
    except Exception as e:
        print(f"❌ PDF extraction failed for {filename}: {e}")
        return []

def chunk_pages(pages, chunk_size=1000, overlap=200):
    """Split pages into chunks"""
    chunks = []
    for page in pages:
        text = page.get('text', '').strip()
        page_number = page.get('page_number', -1)
        filename = page.get('filename', 'unknown')

        if not text:
            print(f"⚠️ Skipping empty page {page_number} from {filename}")
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            if len(chunk_text.strip()) > 50:  # Only add meaningful chunks
                chunks.append({
                    'page_number': page_number,
                    'text': chunk_text,
                    'filename': filename
                })

            start += chunk_size - overlap

    print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

def embed_and_store_chunks(chunks):
    """Embed chunks and store them in Pinecone"""
    if not chunks:
        print("⚠️ No chunks to embed")
        return False
        
    vectors = []

    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
        try:
            text = chunk['text']
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )["embedding"]

            chunk_id = f"chunk-{uuid.uuid4()}"

            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": text,
                    "page_number": chunk['page_number'],
                    "filename": chunk['filename']
                }
            })

        except Exception as e:
            print(f"❌ Embedding error on chunk {i}: {e}")
            continue

    if vectors:
        try:
            print(f"📤 Upserting {len(vectors)} vectors to namespace '{SESSION_NAMESPACE}'")
            res = index.upsert(vectors=vectors, namespace=SESSION_NAMESPACE)
            print("✅ Upsert successful")

            # Verify storage
            stats = index.describe_index_stats()
            ns_count = stats.get("namespaces", {}).get(SESSION_NAMESPACE, {}).get("vector_count", 0)
            print(f"📌 Stored vectors in namespace '{SESSION_NAMESPACE}': {ns_count}")
            return True

        except Exception as e:
            print(f"❌ Upsert failed: {e}")
            return False
    else:
        print("⚠️ No vectors to upsert — embedding failed or chunks were empty.")
        return False

def get_question_embedding(question):
    """Get embedding for a question"""
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=question,
            task_type="retrieval_query"
        )
        return result["embedding"]
    except Exception as e:
        print(f"❌ Query embedding failed: {e}")
        return []

def retrieve_relevant_chunks(question, top_k=5):
    """Retrieve relevant chunks from Pinecone"""
    try:
        vector = get_question_embedding(question)
        if not vector:
            return []
            
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=SESSION_NAMESPACE
        )
        
        if not results.get('matches'):
            return []
            
        return [
            {
                'text': match['metadata']['text'],
                'filename': match['metadata'].get('filename', 'unknown'),
                'page_number': match['metadata'].get('page_number', 'unknown'),
                'score': match['score']
            }
            for match in results['matches']
        ]
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return []

def generate_answer_from_context(question, context_chunks):
    """Generate answer using Gemini based on context"""
    if not context_chunks:
        return "⚠️ No relevant context found to answer your question."
        
    try:
        context_text = "\n\n".join([
            f"[From {chunk['filename']}, Page {chunk['page_number']}]:\n{chunk['text']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""You are a helpful assistant answering questions based on uploaded PDF documents.
Use only the context below to answer the question. If the context doesn't contain enough information, say so.
When referencing information, mention which document it came from.

Context:
{context_text}

Question: {question}

Answer:"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Answer generation failed: {e}"

def handle_pdf_upload(files):
    """Handle multiple PDF uploads and processing"""
    global uploaded_documents, all_chunks
    
    if not files:
        return "❌ Please upload at least one PDF file."
    
    # Ensure files is a list
    if not isinstance(files, list):
        files = [files]
    
    results = []
    new_chunks = []
    
    for file in files:
        try:
            # Get filename
            if hasattr(file, 'name'):
                filename = os.path.basename(file.name)
            else:
                filename = f"document_{len(uploaded_documents) + 1}.pdf"
                
            # Skip if already uploaded
            if filename in uploaded_documents:
                results.append(f"⚠️ {filename} already uploaded, skipping...")
                continue
            
            # Handle file input
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    file_bytes = f.read()
            elif hasattr(file, 'read'):
                file_bytes = file.read()
            elif hasattr(file, 'name'):
                with open(file.name, 'rb') as f:
                    file_bytes = f.read()
            else:
                results.append(f"❌ Unexpected file format for {filename}: {type(file)}")
                continue
            
            print(f"📁 Processing {filename}, size: {len(file_bytes)} bytes")
            
            # Extract text from PDF
            pages = extract_text_from_pdf(file_bytes, filename)
            if not pages:
                results.append(f"❌ Could not extract text from {filename}")
                continue
            
            # Create chunks
            doc_chunks = chunk_pages(pages)
            if not doc_chunks:
                results.append(f"❌ Could not create chunks from {filename}")
                continue
            
            # Store document info
            uploaded_documents[filename] = {
                'chunks': doc_chunks,
                'total_pages': len(pages),
                'total_chunks': len(doc_chunks)
            }
            
            new_chunks.extend(doc_chunks)
            results.append(f"✅ {filename}: {len(pages)} pages, {len(doc_chunks)} chunks")
            
        except Exception as e:
            results.append(f"❌ Error processing {filename}: {str(e)}")
    
    if new_chunks:
        # Add new chunks to all_chunks
        all_chunks.extend(new_chunks)
        
        # Embed and store new chunks
        success = embed_and_store_chunks(new_chunks)
        if success:
            results.append(f"\n🎉 Successfully processed {len(new_chunks)} new chunks!")
            results.append(f"📚 Total documents: {len(uploaded_documents)}")
            results.append(f"📄 Total chunks: {len(all_chunks)}")
        else:
            results.append("\n❌ Failed to store embeddings for some documents.")
    
    return "\n".join(results)

def answer_question(question):
    """Answer a question based on uploaded PDFs"""
    if not uploaded_documents:
        return "⚠️ Please upload at least one PDF first."
    
    if not question.strip():
        return "⚠️ Please enter a question."

    try:
        # Retrieve relevant context from all documents
        context_chunks = retrieve_relevant_chunks(question, top_k=5)
        
        if not context_chunks:
            return "⚠️ No relevant information found in any of the uploaded documents for your question."

        # Generate answer
        answer = generate_answer_from_context(question, context_chunks)
        
        # Add source information
        unique_sources = set(chunk['filename'] for chunk in context_chunks)
        source_info = f"\n\n📚 Sources: {', '.join(unique_sources)}"
        
        return answer + source_info
        
    except Exception as e:
        return f"❌ Error answering question: {str(e)}"

def get_document_list():
    """Get formatted list of uploaded documents"""
    if not uploaded_documents:
        return "No documents uploaded yet."
    
    doc_list = []
    for filename, info in uploaded_documents.items():
        doc_list.append(f"📄 {filename} ({info['total_pages']} pages, {info['total_chunks']} chunks)")
    
    return "\n".join(doc_list)

def clear_session():
    """Clear the current session"""
    global uploaded_documents, all_chunks
    uploaded_documents = {}
    all_chunks = []
    return "✅ Session cleared. You can upload new PDFs."

def remove_document(filename):
    """Remove a specific document"""
    global uploaded_documents, all_chunks
    
    if filename not in uploaded_documents:
        return f"❌ Document '{filename}' not found."
    
    # Remove chunks from all_chunks
    all_chunks = [chunk for chunk in all_chunks if chunk.get('filename') != filename]
    
    # Remove from uploaded_documents
    del uploaded_documents[filename]
    
    # Note: In a production system, you'd also want to remove vectors from Pinecone
    # This would require tracking vector IDs by filename
    
    return f"✅ Removed '{filename}'. Remaining documents: {len(uploaded_documents)}"

# Gradio UI
with gr.Blocks(title="Multi-PDF Chat Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Chat with Multiple PDFs")
    gr.Markdown("Upload multiple PDF documents and ask questions about their content using AI-powered search and generation.")
    
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="📄 Upload PDF(s)", 
                file_types=[".pdf"],
                file_count="multiple"  # Allow multiple file selection
            )
        with gr.Column(scale=1):
            upload_btn = gr.Button("📤 Process PDFs", variant="primary")
            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
    
    upload_status = gr.Textbox(
        label="📊 Upload Status", 
        interactive=False,
        max_lines=5
    )
    
    # Document management section
    with gr.Row():
        with gr.Column():
            doc_list_btn = gr.Button("📚 Show Uploaded Documents")
            doc_list = gr.Textbox(
                label="📚 Uploaded Documents",
                interactive=False,
                max_lines=5
            )

    gr.Markdown("### 💬 Ask Questions")
    
    with gr.Row():
        question_box = gr.Textbox(
            label="❓ Your question",
            placeholder="Ask anything about your uploaded PDFs...",
            lines=2
        )
    
    with gr.Row():
        ask_btn = gr.Button("🔍 Ask Question", variant="primary")
    
    answer_box = gr.Textbox(
        label="🤖 Answer", 
        interactive=False,
        lines=12
    )

    # Event handlers
    upload_btn.click(
        fn=handle_pdf_upload, 
        inputs=[file_input], 
        outputs=[upload_status]
    )
    
    clear_btn.click(
        fn=clear_session,
        outputs=[upload_status]
    )
    
    doc_list_btn.click(
        fn=get_document_list,
        outputs=[doc_list]
    )
    
    ask_btn.click(
        fn=answer_question, 
        inputs=[question_box], 
        outputs=[answer_box]
    )
    
    question_box.submit(
        fn=answer_question, 
        inputs=[question_box], 
        outputs=[answer_box]
    )

if __name__ == "__main__":
    demo.launch(
        share=True, 
        debug=True
    )
