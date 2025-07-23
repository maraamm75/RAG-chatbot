import streamlit as st
from chatbot import (
    extract_text_from_pdf,
    chunk_pages,
    embed_and_store_chunks,
    retrieve_relevant_chunks,
    generate_answer_from_context
)

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot using Pinecone & Gemini")

# Session state to track file upload
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# File uploader
uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

if uploaded_file:
    st.session_state.pdf_uploaded = True

    # Step 1: Text Extraction
    with st.spinner("📄 Step 1: Extracting text from PDF..."):
        try:
            pages = extract_text_from_pdf(uploaded_file.read())
            st.success(f"✅ Step 1 Complete: Extracted {len(pages)} pages.")
        except Exception as e:
            st.error(f"❌ Error in text extraction: {e}")
            st.stop()

    # Step 2: Chunking
    with st.spinner("✂️ Step 2: Chunking text into smaller parts..."):
        try:
            chunks = chunk_pages(pages)
            st.success(f"✅ Step 2 Complete: Created {len(chunks)} chunks.")
        except Exception as e:
            st.error(f"❌ Error in chunking: {e}")
            st.stop()

    # Step 3: Embedding and storing
    with st.spinner("📦 Step 3: Generating embeddings and storing in Pinecone..."):
        try:
            embed_and_store_chunks(chunks)
            st.success("✅ Step 3 Complete: Embeddings stored in Pinecone.")
        except Exception as e:
            st.error(f"❌ Error in embedding or storage: {e}")
            st.stop()

# Step 4: QA interface
if st.session_state.pdf_uploaded:
    question = st.text_input("❓ Ask a question from the uploaded PDF:")
    if question:
        with st.spinner("🔍 Step 4: Retrieving relevant chunks..."):
            try:
                context = retrieve_relevant_chunks(question)
                st.success("✅ Retrieved relevant context for the question.")
            except Exception as e:
                st.error(f"❌ Error retrieving context: {e}")
                st.stop()

        with st.spinner("🤖 Step 5: Generating answer using Gemini..."):
            try:
                answer = generate_answer_from_context(question, context)
                st.markdown("### 💬 Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")