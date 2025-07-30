import streamlit as st
import os
import numpy as np
import requests
import time
from typing import List, Dict
from chatbot import (
    extract_text_from_pdf,
    chunk_pages,
    LocalVectorStore,
    LocalEmbedder,
    IntelligentOllamaLLM,
    retrieve_relevant_chunks,
    generate_intelligent_answer
)

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 Intelligent PDF Chatbot", 
    layout="wide",
    page_icon="🧠"
)

# Enhanced CSS for better interface
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #01243d;
    border-left: 4px solid #2196f3;
    color: #1565c0;
}
.assistant-message {
    background-color: #f1f8e9;
    border-left: 4px solid #4caf50;
    color: #2e7d32;
}
.general-message {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
    color: #e65100;
}
.message-header {
    font-weight: bold;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
}
.message-content {
    line-height: 1.6;
    font-size: 1rem;
}
.status-info {
    background-color: #4f3101;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff9800;
    margin-bottom: 1rem;
}
.ollama-status {
    background-color: #140217;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #9c27b0;
    margin-bottom: 1rem;
}
.mode-selector {
    background-color: #02030a;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Intelligent PDF Chatbot with Ollama")
st.markdown("*Advanced AI assistant that combines document knowledge with general intelligence*")

# Helper functions
def check_ollama_connection():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, response.json().get('models', [])
        else:
            return False, []
    except:
        return False, []

@st.cache_resource
def load_embedder():
    return LocalEmbedder()

def load_ollama_llm(model_name):
    return IntelligentOllamaLLM(model_name)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = LocalVectorStore()

if "embedder" not in st.session_state:
    with st.spinner("🔄 Loading embedding model..."):
        try:
            st.session_state.embedder = load_embedder()
            st.success("✅ Embedding model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading embedder: {e}")
            st.stop()

# Enhanced available models
AVAILABLE_MODELS = {
    "🦙 LLaMA3-8B Instruct (koesn)": "koesn/llama3-8b-instruct:latest",
    "🦙 LLaMA3 Latest": "llama3:latest",
    "🦙 LLaMA 3.2 Latest": "llama3.2:latest",
    "🔥 Mistral 7B": "mistral:7b",
    "🔥 Mistral Latest": "mistral:latest",
    "⚡ Qwen3 1.7B (Fast)": "qwen3:1.7b",
    "⚡ Qwen3 Latest": "qwen3:latest",
    "🧠 DeepSeek R1 (Reasoning)": "deepseek-r1:latest"
}

# Response modes
RESPONSE_MODES = {
    "🧠 Enhanced (Recommended)": "enhanced",
    "📚 Documents Only": "context_only", 
    "🌍 General Knowledge": "general"
}

if "current_model" not in st.session_state:
    st.session_state.current_model = "llama3.2:latest"

if "current_model_display" not in st.session_state:
    st.session_state.current_model_display = "🦙 LLaMA 3.2 Latest"

if "response_mode" not in st.session_state:
    st.session_state.response_mode = "enhanced"

if "llm" not in st.session_state:
    with st.spinner(f"🔄 Connecting to {st.session_state.current_model_display}..."):
        try:
            st.session_state.llm = load_ollama_llm(st.session_state.current_model)
            st.success(f"✅ {st.session_state.current_model_display} connected successfully!")
        except Exception as e:
            st.error(f"❌ Error connecting to model: {e}")
            st.info("💡 Make sure Ollama is running with: **ollama serve**")
            st.stop()

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for settings and document management
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Ollama Status Check
    st.subheader("🦙 Ollama Status")
    
    if st.button("🔄 Check Available Models", use_container_width=True):
        is_connected, available_ollama_models = check_ollama_connection()
        if is_connected:
            model_names = [model['name'] for model in available_ollama_models]
            st.markdown(f"""
            <div class="ollama-status">
                <strong>✅ Ollama Connected!</strong><br>
                <strong>Available Models:</strong><br>
                {', '.join(model_names[:8])}
                {f"<br>... and {len(model_names)-8} more" if len(model_names) > 8 else ""}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Cannot connect to Ollama")
            st.info("💡 Start Ollama with: **ollama serve**")
    
    # Model Selection
    st.subheader("🤖 Language Model")
    
    selected_model_display = st.selectbox(
        "Choose your Ollama model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.values()).index(st.session_state.current_model) if st.session_state.current_model in AVAILABLE_MODELS.values() else 2,
        help="Different models have different strengths and speeds"
    )
    
    selected_model = AVAILABLE_MODELS[selected_model_display]
    
    # Response Mode Selection
    st.subheader("🧠 Intelligence Mode")
    
    selected_mode_display = st.selectbox(
        "Choose response mode:",
        options=list(RESPONSE_MODES.keys()),
        index=0,  # Default to Enhanced
        help="How the AI should approach your questions"
    )
    
    selected_mode = RESPONSE_MODES[selected_mode_display]
    st.session_state.response_mode = selected_mode
    
    # Mode explanation
    st.markdown(f"""
    <div class="mode-selector">
        <strong>Current Mode: {selected_mode_display}</strong><br>
        {
            "🧠 <em>Uses both documents AND general knowledge for comprehensive answers</em>" if selected_mode == "enhanced" else
            "📚 <em>Answers only from your uploaded documents</em>" if selected_mode == "context_only" else
            "🌍 <em>Uses general AI knowledge, ignores documents</em>"
        }
    </div>
    """, unsafe_allow_html=True)
    
    # Update LLM if changed
    if selected_model != st.session_state.current_model:
        st.session_state.current_model = selected_model
        st.session_state.current_model_display = selected_model_display
        
        with st.spinner(f"🔄 Switching to {selected_model_display}..."):
            try:
                st.session_state.llm = load_ollama_llm(selected_model)
                st.success(f"✅ Switched to {selected_model_display}")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error switching model: {e}")
                st.info("💡 Make sure the model is installed in Ollama")
    
    # Current configuration info
    st.info(f"**Model**: {st.session_state.current_model_display}")
    st.info(f"**Mode**: {selected_mode_display}")
    st.info(f"**Embedding**: all-MiniLM-L6-v2")
    st.info(f"**Storage**: FAISS (local)")
    
    st.markdown("---")
    
    # Document upload section
    st.header("📤 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload multiple PDFs for intelligent analysis"
    )
    
    if uploaded_files and st.button("🔄 Process PDFs", type="primary", use_container_width=True):
        all_new_chunks = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            filename = uploaded_file.name
            status_text.text(f"Processing {filename}...")
            
            # Step 1: Enhanced Text Extraction
            try:
                file_bytes = uploaded_file.read()
                pages = extract_text_from_pdf(file_bytes, filename)
                if pages:
                    st.success(f"✅ {filename}: {len(pages)} pages extracted")
                else:
                    st.warning(f"⚠️ {filename}: No text extracted")
                    continue
            except Exception as e:
                st.error(f"❌ Extraction error {filename}: {e}")
                continue
            
            # Step 2: Enhanced Chunking
            try:
                chunks = chunk_pages(pages, chunk_size=1200, overlap=200)
                all_new_chunks.extend(chunks)
                st.success(f"✅ {filename}: {len(chunks)} intelligent chunks created")
            except Exception as e:
                st.error(f"❌ Chunking error {filename}: {e}")
                continue
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        if all_new_chunks:
            # Step 3: Embedding and storing
            status_text.text("Generating embeddings and storing...")
            try:
                # Generate embeddings
                texts = [chunk['text'] for chunk in all_new_chunks]
                embeddings = st.session_state.embedder.embed_documents(texts)
                
                # Store in vector database
                st.session_state.vector_store.add_documents(texts, embeddings, all_new_chunks)
                
                # Update session state
                st.session_state.document_chunks.extend(all_new_chunks)
                st.session_state.pdf_uploaded = True
                
                st.success(f"✅ {len(all_new_chunks)} chunks processed successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Embedding error: {e}")
        
        status_text.empty()
        progress_bar.empty()
    
    # Statistics
    st.markdown("---")
    st.header("📊 Statistics")
    if st.session_state.document_chunks:
        unique_docs = list(set(chunk.get('filename', 'unknown') for chunk in st.session_state.document_chunks))
        st.metric("Documents processed", len(unique_docs))
        st.metric("Total chunks", len(st.session_state.document_chunks))
        st.metric("Stored vectors", st.session_state.vector_store.get_count())
        st.metric("Chat messages", len(st.session_state.chat_history))
        
        # Show loaded documents
        with st.expander("📚 Loaded documents"):
            for doc in unique_docs:
                st.text(f"• {doc}")
    else:
        st.metric("Documents processed", 0)
        st.metric("Total chunks", 0)
        st.metric("Stored vectors", 0)
        st.metric("Chat messages", len(st.session_state.chat_history))
    
    # Clear buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🗑️ Delete All", use_container_width=True):
            st.session_state.vector_store = LocalVectorStore()
            st.session_state.document_chunks = []
            st.session_state.pdf_uploaded = False
            st.session_state.chat_history = []
            st.rerun()

# Main chat interface
st.header("💬 Intelligent Conversation")

# Display current status
if st.session_state.pdf_uploaded:
    unique_docs = list(set(chunk.get('filename', 'unknown') for chunk in st.session_state.document_chunks))
    st.markdown(f"""
    <div class="status-info">
        <strong>🧠 Ready to chat intelligently with {st.session_state.current_model_display}!</strong><br>
        Mode: {selected_mode_display}<br>
        Documents: {', '.join(unique_docs[:3])}
        {f" and {len(unique_docs)-3} more..." if len(unique_docs) > 3 else ""}
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.response_mode == "general":
    st.markdown(f"""
    <div class="status-info">
        <strong>🌍 General Knowledge Mode Active</strong><br>
        Ready to answer any question using {st.session_state.current_model_display}!<br>
        <em>Upload documents to enable enhanced mode</em>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ Please upload PDF documents or switch to General Knowledge mode to start chatting.")

# Chat history display
if st.session_state.chat_history:
    st.subheader("💬 Conversation History")
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">👤 You</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            model_display = message.get("model", st.session_state.current_model_display)
            mode = message.get("mode", "enhanced")
            
            # Different styling based on mode
            message_class = "assistant-message"
            if mode == "general":
                message_class = "general-message"
            
            mode_emoji = "🧠" if mode == "enhanced" else "🌍" if mode == "general" else "📚"
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="message-header">{mode_emoji} {model_display} ({mode.title()} Mode)</div>
                <div class="message-content">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# Question input section
st.markdown("---")
st.subheader("❓ Ask Your Question")

# Create a form for better UX
with st.form(key="question_form", clear_on_submit=True):
    question = st.text_area(
        "Type your question here:",
        placeholder="Ask anything! I can help with document analysis, general knowledge, explanations, translations, and much more...",
        height=100,
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ask_button = st.form_submit_button("🔍 Ask Question", type="primary", use_container_width=True)
    
    with col2:
        example_button = st.form_submit_button("💡 Examples", use_container_width=True)
    
    with col3:
        smart_button = st.form_submit_button("🧠 Smart Ask", use_container_width=True)

# Handle example questions
if example_button:
    if st.session_state.response_mode == "general":
        example_questions = [
            "Explain quantum physics in simple terms",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Translate 'Hello world' to French and Spanish",
            "What's the difference between AI and ML?",
            "Explain the theory of relativity",
            "How do solar panels work?",
            "What are the causes of climate change?",
            "Explain blockchain technology"
        ]
    elif st.session_state.response_mode == "context_only":
        example_questions = [
            "What are the main topics in the documents?",
            "Summarize the key findings",
            "What conclusions are mentioned?",
            "Are there any recommendations?",
            "What methodology was used?",
            "What are the main results?",
            "Explain the context of this study",
            "What problems are identified?",
            "What solutions are proposed?"
        ]
    else:  # enhanced mode
        example_questions = [
            "Analyze the documents and provide broader context",
            "Compare findings with current industry standards",
            "What are the implications of these results?",
            "How do these findings relate to global trends?",
            "Provide a comprehensive analysis with external insights",
            "What are the real-world applications?",
            "How does this compare to similar studies?",
            "What are the future implications?",
            "Provide expert commentary on these findings"
        ]
    
    import random
    question = random.choice(example_questions)
    st.info(f"Example question: {question}")

# Handle smart questions (context-aware suggestions)
if smart_button:
    if st.session_state.document_chunks:
        # Analyze document content to suggest smart questions
        doc_sample = " ".join([chunk['text'][:100] for chunk in st.session_state.document_chunks[:3]])
        smart_questions = [
            f"What insights can you provide about {st.session_state.document_chunks[0].get('filename', 'this document')}?",
            "Provide an expert analysis with broader context",
            "What are the key takeaways and their real-world implications?",
            "Compare these findings with industry best practices",
            "What questions should I be asking about this content?"
        ]
    else:
        smart_questions = [
            "What's the latest breakthrough in artificial intelligence?",
            "Explain complex topics in simple terms",
            "What should I know about current global trends?",
            "Help me understand a difficult concept",
            "What are the most important developments in science?"
        ]
    
    import random
    question = random.choice(smart_questions)
    st.info(f"Smart suggestion: {question}")

# Process the question
if ask_button and question.strip():
    # Add user question to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })
    
    # Determine if we need document retrieval
    context_chunks = []
    if st.session_state.response_mode in ["enhanced", "context_only"] and st.session_state.pdf_uploaded:
        with st.spinner("🔍 Searching for relevant information..."):
            try:
                context_chunks = retrieve_relevant_chunks(
                    question, 
                    st.session_state.embedder, 
                    st.session_state.vector_store,
                    top_k=5  # More chunks for better context
                )
                
                if context_chunks:
                    st.success(f"✅ Found {len(context_chunks)} relevant chunks")
                elif st.session_state.response_mode == "context_only":
                    st.warning("⚠️ No relevant information found in documents")
                    
            except Exception as e:
                st.error(f"❌ Search error: {e}")
                context_chunks = []
    
    # Generate intelligent answer
    mode_emoji = "🧠" if st.session_state.response_mode == "enhanced" else "🌍" if st.session_state.response_mode == "general" else "📚"
    
    with st.spinner(f"{mode_emoji} {st.session_state.current_model_display} generating intelligent response..."):
        try:
            answer = generate_intelligent_answer(
                question, 
                context_chunks, 
                st.session_state.llm,
                st.session_state.response_mode
            )
            
            if answer and answer.strip() and not answer.startswith("❌"):
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "model": st.session_state.current_model_display,
                    "mode": st.session_state.response_mode
                })
                
                # Refresh to show new message
                st.rerun()
            else:
                fallback_msg = "I couldn't generate an appropriate response. Please rephrase your question or check if the documents contain the requested information."
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": fallback_msg,
                    "model": st.session_state.current_model_display,
                    "mode": st.session_state.response_mode
                })
                st.rerun()
                
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}\n\n💡 **Suggestions:**\n- Check that Ollama is running: `ollama serve`\n- Try a smaller model (Qwen3 1.7B)\n- Restart the application if needed"
            st.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "model": st.session_state.current_model_display,
                "mode": st.session_state.response_mode
            })
            st.rerun()

elif ask_button and not question.strip():
    st.warning("Please enter a question!")

elif ask_button and st.session_state.response_mode == "context_only" and not st.session_state.pdf_uploaded:
    st.error("Please upload PDF documents first or switch to General Knowledge mode!")

# Advanced help section
st.markdown("---")
with st.expander("ℹ️ How to use this Intelligent Chatbot"):
    st.markdown(f"""
    ### 🧠 Intelligence Modes:
    
    **🧠 Enhanced Mode (Recommended)**
    - Combines document content with general AI knowledge
    - Provides comprehensive, contextual answers
    - Best for analysis, insights, and detailed explanations
    
    **📚 Documents Only Mode**
    - Answers strictly from your uploaded PDFs
    - Perfect for document-specific queries
    - Ensures responses are grounded in your content
    
    **🌍 General Knowledge Mode**
    - Pure AI knowledge without document context
    - Great for learning, explanations, and general questions
    - Works without any uploaded documents
    
    ### 🚀 Advanced Features:
    
    - **🌍 Multilingual Support**: Ask in any language, get answers in the same language
    - **📄 Advanced PDF Processing**: Handles complex layouts, tables, and multiple formats
    - **🧠 Intelligent Chunking**: Smart text segmentation for better context
    - **🔍 Semantic Search**: Finds relevant content even with different wording
    - **💭 Context Awareness**: Maintains conversation context and memory
    - **⚡ Model Flexibility**: Choose the right model for your needs
    
    ### 💡 Pro Tips:
    
    - **Specific Questions**: "Explain the methodology used in section 3"
    - **Analytical Queries**: "What are the implications of these findings?"
    - **Comparative Analysis**: "How does this compare to industry standards?"
    - **Multilingual**: "Résume ce document en français" or "¿Qué dice sobre...?"
    - **Complex Reasoning**: Use DeepSeek R1 for deep analytical questions
    
    ### 🦙 Model Recommendations:
    
    - **Qwen3 1.7B**: Lightning fast, great for simple questions
    - **LLaMA 3.2**: Best all-around performance
    - **Mistral**: Excellent for technical content
    - **DeepSeek R1**: Superior reasoning and analysis
    - **LLaMA3-8B**: Most comprehensive responses
    
    ### 🔧 Technical Features:
    
    - ✅ **100% Private**: All processing happens locally
    - ✅ **No Limits**: Unlimited questions and document size
    - ✅ **Multi-format**: Supports various PDF types and layouts
    - ✅ **Real-time**: Instant responses with efficient caching
    - ✅ **Customizable**: Adjust parameters for your needs
    
    ### 🌟 Use Cases:
    
    - **Research Analysis**: Deep dive into academic papers
    - **Business Intelligence**: Analyze reports and documents
    - **Legal Review**: Examine contracts and legal documents
    - **Technical Documentation**: Understand complex manuals
    - **Educational Support**: Learn from textbooks and materials
    - **Content Creation**: Generate insights from source materials
    
    ### 🛠️ Troubleshooting:
    
    ```bash
    # Start Ollama
    ollama serve
    
    # Check available models
    ollama list
    
    # Pull a new model
    ollama pull qwen3:1.7b
    
    # Test a model
    ollama run llama3.2:latest "Hello"
    ```
    """)

# Footer with status
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    if response.status_code == 200:
        ollama_status = "🟢 Ollama Connected"
        models_count = len(response.json().get('models', []))
        status_detail = f"({models_count} models available)"
    else:
        ollama_status = "🟡 Ollama Issue"
        status_detail = f"(Status: {response.status_code})"
except:
    ollama_status = "🔴 Ollama Disconnected"
    status_detail = "(Start with: ollama serve)"

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    🧠 Intelligent PDF Chatbot | {ollama_status} {status_detail} | 
    Mode: {selected_mode_display} | Model: {st.session_state.current_model_display}<br>
    🔒 Your data stays private and secure on your machine
</div>
""", unsafe_allow_html=True)
