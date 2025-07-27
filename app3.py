import streamlit as st
import os
import pickle
import numpy as np
from typing import List, Dict
from chatbot import (
    extract_text_from_pdf,
    chunk_pages,
    LocalVectorStore,
    LocalEmbedder,
    LocalLLM,
    retrieve_relevant_chunks,
    generate_answer_from_context
)

# Configure Streamlit page
st.set_page_config(
    page_title="Local PDF RAG Chatbot", 
    layout="wide",
    page_icon="📄"
)

st.title("📄 Local PDF RAG Chatbot")
st.markdown("A completely local chatbot that runs without any API keys! Uses open-source models for embeddings and text generation.")

# Initialize components in session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = LocalVectorStore()

if "embedder" not in st.session_state:
    with st.spinner("🔄 Loading embedding model..."):
        st.session_state.embedder = LocalEmbedder()

if "llm" not in st.session_state:
    with st.spinner("🔄 Loading language model..."):
        st.session_state.llm = LocalLLM()

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

# Sidebar for model info
with st.sidebar:
    st.header("🤖 Model Information")
    st.info("**Embedding Model**: all-MiniLM-L6-v2")
    st.info("**Language Model**: GPT-2 (Local)")
    st.info("**Vector Store**: In-memory FAISS")
    
    st.header("📊 Statistics")
    if st.session_state.document_chunks:
        st.metric("Documents Processed", len(set(chunk.get('filename', 'unknown') for chunk in st.session_state.document_chunks)))
        st.metric("Total Chunks", len(st.session_state.document_chunks))
        st.metric("Vectors Stored", st.session_state.vector_store.get_count())
    else:
        st.metric("Documents Processed", 0)
        st.metric("Total Chunks", 0)
        st.metric("Vectors Stored", 0)

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once"
    )
    
    if uploaded_files and st.button("🔄 Process PDFs", type="primary"):
        all_new_chunks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_text.text(f"Processing {filename}...")
            
            # Step 1: Text Extraction
            try:
                file_bytes = uploaded_file.read()
                pages = extract_text_from_pdf(file_bytes, filename)
                st.success(f"✅ {filename}: Extracted {len(pages)} pages")
            except Exception as e:
                st.error(f"❌ Error extracting text from {filename}: {e}")
                continue
            
            # Step 2: Chunking
            try:
                chunks = chunk_pages(pages)
                all_new_chunks.extend(chunks)
                st.success(f"✅ {filename}: Created {len(chunks)} chunks")
            except Exception as e:
                st.error(f"❌ Error chunking {filename}: {e}")
                continue
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_new_chunks:
            # Step 3: Embedding and storing
            status_text.text("Generating embeddings and storing vectors...")
            try:
                # Generate embeddings
                texts = [chunk['text'] for chunk in all_new_chunks]
                embeddings = st.session_state.embedder.embed_documents(texts)
                
                # Store in vector database
                st.session_state.vector_store.add_documents(texts, embeddings, all_new_chunks)
                
                # Update session state
                st.session_state.document_chunks.extend(all_new_chunks)
                st.session_state.pdf_uploaded = True
                
                st.success(f"✅ Successfully processed {len(all_new_chunks)} chunks from {len(uploaded_files)} files!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error in embedding or storage: {e}")
        
        status_text.empty()
        progress_bar.empty()

with col2:
    st.header("💬 Chat Interface")
    
    if st.session_state.pdf_uploaded:
        # Display uploaded documents
        unique_docs = list(set(chunk.get('filename', 'unknown') for chunk in st.session_state.document_chunks))
        st.info(f"📚 Loaded documents: {', '.join(unique_docs)}")
        
        # Question input
        question = st.text_area(
            "❓ Ask a question about your documents:",
            placeholder="What is the main topic discussed in the document?",
            height=100
        )
        
        col_ask, col_clear = st.columns([3, 1])
        
        with col_ask:
            ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.vector_store = LocalVectorStore()
                st.session_state.document_chunks = []
                st.session_state.pdf_uploaded = False
                st.rerun()
        
        if ask_button and question.strip():
            with st.spinner("🔍 Searching for relevant information..."):
                try:
                    # Retrieve relevant chunks
                    context_chunks = retrieve_relevant_chunks(
                        question, 
                        st.session_state.embedder, 
                        st.session_state.vector_store,
                        top_k=5
                    )
                    
                    if context_chunks:
                        st.success(f"✅ Found {len(context_chunks)} relevant chunks")
                    else:
                        st.warning("⚠️ No relevant information found")
                        
                except Exception as e:
                    st.error(f"❌ Error retrieving context: {e}")
                    context_chunks = []
            
            if context_chunks:
                with st.spinner("🤖 Generating answer..."):
                    try:
                        answer = generate_answer_from_context(
                            question, 
                            context_chunks, 
                            st.session_state.llm
                        )
                        
                        st.markdown("### 💬 Answer:")
                        st.write(answer)
                        
                        # Show sources
                        with st.expander("📚 Sources"):
                            for i, chunk in enumerate(context_chunks):
                                st.markdown(f"**Source {i+1}** (Score: {chunk.get('score', 'N/A'):.3f})")
                                st.markdown(f"*From: {chunk.get('filename', 'Unknown')} - Page {chunk.get('page_number', 'Unknown')}*")
                                st.text(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
                                st.markdown("---")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {e}")
    else:
        st.info("👆 Please upload and process PDF files first to start chatting!")

# Footer
st.markdown("---")
st.markdown(
    "🚀 **Local PDF RAG Chatbot** - Powered by open-source models | "
    "No API keys required | Runs completely offline"
)
