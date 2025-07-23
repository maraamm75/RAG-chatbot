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

# Store uploaded chunks in memory for each session
uploaded_chunks = []

def clean_text(text):
    """Clean extracted text from PDF"""
    if not text:
        return ""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    lines = text.split('\n')
    return '\n'.join([line.strip() for line in lines if len(line.strip()) > 10])

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF bytes"""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            print(f"📄 Extracting page {i+1} of {len(reader.pages)}")
            text = page.extract_text()
            cleaned = clean_text(text)
            if cleaned:  # Only add non-empty pages
                pages.append({
                    "page_number": i + 1,
                    "text": cleaned
                })
        print(f"📄 Extracted {len(pages)} pages with content")
        return pages
    except Exception as e:
        print(f"❌ PDF extraction failed: {e}")
        return []

def chunk_pages(pages, chunk_size=1000, overlap=200):
    """Split pages into chunks"""
    chunks = []
    for page in pages:
        text = page.get('text', '').strip()
        page_number = page.get('page_number', -1)

        if not text:
            print(f"⚠️ Skipping empty page {page_number}")
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            if len(chunk_text.strip()) > 50:  # Only add meaningful chunks
                chunks.append({
                    'page_number': page_number,
                    'text': chunk_text
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
                    "page_number": chunk['page_number']
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

def retrieve_relevant_chunks(question, top_k=3):
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
            
        return [match['metadata']['text'] for match in results['matches']]
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return []

def generate_answer_from_context(question, context_chunks):
    """Generate answer using Gemini based on context"""
    if not context_chunks:
        return "⚠️ No relevant context found to answer your question."
        
    try:
        context = "\n\n".join(context_chunks)
        prompt = f"""You are a helpful assistant answering questions based on a document.
Use only the context below to answer the question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Answer generation failed: {e}"

def handle_pdf_upload(file):
    """Handle PDF upload and processing"""
    global uploaded_chunks
    
    if file is None:
        return "❌ Please upload a PDF file first."
    
    try:
        # Handle file input - file is the file path in newer Gradio versions
        if isinstance(file, str):
            # File path provided
            with open(file, 'rb') as f:
                file_bytes = f.read()
        elif hasattr(file, 'read'):
            # File-like object
            file_bytes = file.read()
        elif hasattr(file, 'name'):
            # File object with name attribute
            with open(file.name, 'rb') as f:
                file_bytes = f.read()
        else:
            return f"❌ Unexpected file format: {type(file)}"
        
        print(f"📁 Processing file, size: {len(file_bytes)} bytes")
        
        # Extract text from PDF
        pages = extract_text_from_pdf(file_bytes)
        if not pages:
            return "❌ Could not extract text from PDF. Make sure it's a text-based PDF."
        
        # Create chunks
        uploaded_chunks = chunk_pages(pages)
        if not uploaded_chunks:
            return "❌ Could not create chunks from PDF content."
        
        # Embed and store
        success = embed_and_store_chunks(uploaded_chunks)
        if success:
            return f"✅ PDF uploaded and processed successfully! Total chunks: {len(uploaded_chunks)}"
        else:
            return "❌ Failed to store embeddings. Please try again."
            
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return f"❌ Upload failed: {str(e)}"

def answer_question(question):
    """Answer a question based on uploaded PDF"""
    if not uploaded_chunks:
        return "⚠️ Please upload a PDF first."
    
    if not question.strip():
        return "⚠️ Please enter a question."

    try:
        # Retrieve relevant context
        context_chunks = retrieve_relevant_chunks(question, top_k=5)
        
        if not context_chunks:
            return "⚠️ No relevant information found in the document for your question."

        # Generate answer
        answer = generate_answer_from_context(question, context_chunks)
        return answer
        
    except Exception as e:
        return f"❌ Error answering question: {str(e)}"

def clear_session():
    """Clear the current session"""
    global uploaded_chunks
    uploaded_chunks = []
    return "✅ Session cleared. You can upload a new PDF."

# Gradio UI
with gr.Blocks(title="PDF Chat Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Chat with your PDF")
    gr.Markdown("Upload a PDF document and ask questions about its content using AI-powered search and generation.")
    
    with gr.Row():
        with gr.Column(scale=2):
            file_input = gr.File(
                label="📄 Upload your PDF", 
                file_types=[".pdf"]
            )
        with gr.Column(scale=1):
            upload_btn = gr.Button("📤 Process PDF", variant="primary")
            clear_btn = gr.Button("🗑️ Clear Session", variant="secondary")
    
    upload_status = gr.Textbox(
        label="📊 Status", 
        interactive=False,
        max_lines=3
    )

    gr.Markdown("### 💬 Ask Questions")
    
    with gr.Row():
        question_box = gr.Textbox(
            label="❓ Your question",
            placeholder="Ask anything about your uploaded PDF...",
            lines=2
        )
    
    with gr.Row():
        ask_btn = gr.Button("🔍 Ask Question", variant="primary")
    
    answer_box = gr.Textbox(
        label="🤖 Answer", 
        interactive=False,
        lines=10
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
