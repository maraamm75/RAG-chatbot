import os
import sys
import pickle
import fitz  # PyMuPDF
import numpy as np
import faiss
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import requests
import time
import json
import warnings
import re
warnings.filterwarnings("ignore")

class LocalEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding model with offline fallback"""
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # MiniLM dimension
        
        # Try loading the model with multiple fallback strategies
        self._load_model()
    
    def _load_model(self):
        """Load model with multiple fallback strategies"""
        print(f"🔄 Loading embedding model: {self.model_name}")
        
        local_path = f"./models/embedding_model"
        if os.path.exists(local_path):
            try:
                print("📁 Loading from local directory...")
                self.model = SentenceTransformer(local_path)
                print("✅ Embedding model loaded successfully from local directory!")
                return
            except Exception as e:
                print(f"⚠️ Failed to load from local directory: {e}")
        
        # Try offline mode first
        try:
            print("🌐 Attempting to load with offline mode...")
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
            print("✅ Embedding model loaded successfully (offline mode)!")
            return
        except Exception as e:
            print(f"⚠️ Offline mode failed: {e}")
        
        # Try online download as last resort
        try:
            print("🌐 Attempting to download model (online)...")
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            os.environ.pop('HF_DATASETS_OFFLINE', None)
            self.model = SentenceTransformer(self.model_name)
            print("✅ Embedding model loaded successfully (online)!")
            
            # Save model for offline use
            os.makedirs("./models", exist_ok=True)
            self.model.save("./models/embedding_model")
            print("💾 Model saved for offline use!")
            return
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model. Error: {e}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for documents"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            print(f"✅ Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode([query], convert_to_numpy=True)[0]
            return embedding
        except Exception as e:
            print(f"❌ Error generating query embedding: {e}")
            raise

class IntelligentOllamaLLM:
    def __init__(self, model_name: str = "llama3.2:latest"):
        """Initialize Enhanced Ollama LLM with intelligent features"""
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        
        # Test connection and model availability
        self._verify_ollama_connection()
        self._verify_model_availability()
    
    def _verify_ollama_connection(self):
        """Verify that Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server is running")
            else:
                raise Exception(f"Ollama server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("❌ Ollama server is not running. Please start it with: ollama serve")
        except requests.exceptions.Timeout:
            raise RuntimeError("❌ Timeout connecting to Ollama server")
        except Exception as e:
            raise RuntimeError(f"❌ Error connecting to Ollama: {e}")
    
    def _verify_model_availability(self):
        """Verify that the specified model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            available_models = response.json()
            
            # Extract model names
            model_names = [model['name'] for model in available_models.get('models', [])]
            
            if self.model_name in model_names:
                print(f"✅ Model {self.model_name} is available")
            else:
                print(f"⚠️ Available models: {model_names}")
                raise RuntimeError(f"❌ Model {self.model_name} is not available. Available models: {model_names}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"❌ Error checking available models: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            available_models = response.json()
            return [model['name'] for model in available_models.get('models', [])]
        except Exception as e:
            print(f"❌ Error getting available models: {e}")
            return []

    def generate_intelligent_response(self, question: str, context_chunks: List[Dict], mode: str = "enhanced") -> str:
        """
        Generate intelligent responses based on different modes:
        - 'enhanced': Use context + general knowledge
        - 'general': Pure general knowledge
        - 'context_only': Only use provided context
        """
        
        if mode == "general" or not context_chunks:
            return self._generate_general_response(question)
        elif mode == "context_only":
            return self._generate_context_response(question, context_chunks)
        else:  # enhanced mode
            return self._generate_enhanced_response(question, context_chunks)
    
    def _generate_general_response(self, question: str) -> str:
        """Generate response using general knowledge only"""
        # Detect language
        language = self._detect_language(question)
        
        # Create a smart general knowledge prompt
        if language == "french":
            prompt = f"""Tu es un assistant intelligent multilingue. Réponds de manière complète, précise et détaillée à la question suivante en français. Utilise tes connaissances générales pour fournir une réponse informative et utile.

Question: {question}

Réponse détaillée:"""
        else:
            prompt = f"""You are an intelligent multilingual assistant. Provide a complete, accurate and detailed answer to the following question. Use your general knowledge to give an informative and helpful response.

Question: {question}

Detailed answer:"""
        
        return self._call_ollama(prompt, max_length=400, temperature=0.7)
    
    def _generate_context_response(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate response using only provided context"""
        if not context_chunks:
            return "Je n'ai pas trouvé d'informations pertinentes dans les documents pour répondre à votre question."
        
        # Build context
        context_text = self._build_context(context_chunks, max_length=2500)
        language = self._detect_language(question)
        
        if language == "french":
            prompt = f"""Basé strictement sur le contexte fourni ci-dessous, réponds de manière complète et détaillée à la question en français. Si l'information n'est pas dans le contexte, dis-le clairement.

Contexte:
{context_text}

Question: {question}

Réponse basée sur le contexte:"""
        else:
            prompt = f"""Based strictly on the context provided below, answer the question completely and in detail. If the information is not in the context, state it clearly.

Context:
{context_text}

Question: {question}

Answer based on context:"""
        
        return self._call_ollama(prompt, max_length=350, temperature=0.4)
    
    def _generate_enhanced_response(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate enhanced response combining context + general knowledge"""
        # Build context if available
        context_text = ""
        if context_chunks:
            context_text = self._build_context(context_chunks, max_length=2000)
        
        language = self._detect_language(question)
        
        # Create enhanced prompt that allows using both context and general knowledge
        if language == "french":
            if context_text:
                prompt = f"""Tu es un assistant intelligent et érudit. Tu as accès à des documents spécifiques ET à tes connaissances générales. Utilise les deux sources pour fournir la meilleure réponse possible.

Documents disponibles:
{context_text}

Consignes:
- Utilise d'abord les informations des documents si elles sont pertinentes
- Complète avec tes connaissances générales si nécessaire
- Sois précis, détaillé et informatif
- Réponds en français
- Si tu utilises des informations externes aux documents, indique-le subtilement

Question: {question}

Réponse complète et intelligente:"""
            else:
                prompt = f"""Tu es un assistant intelligent multilingue. Réponds de manière complète, précise et détaillée à la question suivante en français. Utilise tes connaissances générales pour fournir une réponse informative et utile.

Question: {question}

Réponse détaillée:"""
        else:
            if context_text:
                prompt = f"""You are an intelligent and knowledgeable assistant. You have access to specific documents AND your general knowledge. Use both sources to provide the best possible answer.

Available documents:
{context_text}

Instructions:
- First use information from documents if relevant
- Supplement with your general knowledge if necessary
- Be precise, detailed and informative
- If you use information external to the documents, indicate it subtly

Question: {question}

Complete and intelligent answer:"""
            else:
                prompt = f"""You are an intelligent multilingual assistant. Provide a complete, accurate and detailed answer to the following question. Use your general knowledge to give an informative and helpful response.

Question: {question}

Detailed answer:"""
        
        return self._call_ollama(prompt, max_length=450, temperature=0.6)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        french_indicators = [
            'est', 'sont', 'dans', 'avec', 'pour', 'sur', 'par', 'de', 'du', 'des', 'le', 'la', 'les', 'un', 'une',
            'que', 'qui', 'quoi', 'comment', 'pourquoi', 'où', 'quand', 'quel', 'quelle', 'quels', 'quelles',
            'pouvez', 'vous', 'peux', 'peut', 'puis', 'explique', 'expliquer'
        ]
        
        text_lower = text.lower()
        french_count = sum(1 for word in french_indicators if word in text_lower)
        
        return "french" if french_count >= 2 else "english"
    
    def _build_context(self, context_chunks: List[Dict], max_length: int = 2000) -> str:
        """Build optimized context from chunks"""
        context_texts = []
        total_length = 0
        
        # Sort chunks by relevance score if available
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)
        
        for chunk in sorted_chunks:
            chunk_text = chunk['text'].strip()
            if total_length + len(chunk_text) <= max_length:
                context_texts.append(chunk_text)
                total_length += len(chunk_text)
            else:
                # Add partial chunk if it fits meaningfully
                remaining_space = max_length - total_length
                if remaining_space > 200:  # Only add if meaningful content can fit
                    context_texts.append(chunk_text[:remaining_space] + "...")
                break
        
        return "\n\n".join(context_texts)
    
    def _call_ollama(self, prompt: str, max_length: int = 300, temperature: float = 0.7) -> str:
        """Call Ollama API with enhanced parameters"""
        try:
            # Prepare the request payload with optimized parameters
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_length,
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "repeat_last_n": 64,
                    "num_ctx": 4096,  # Larger context window
                    "num_thread": 8,   # Use multiple threads
                }
            }
            
            # Make the request with longer timeout for complex responses
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180  # 3 minutes timeout for complex questions
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # Enhanced response cleaning
                cleaned_text = self._clean_response(generated_text)
                return cleaned_text
            else:
                error_msg = f"Ollama API returned status {response.status_code}"
                print(f"❌ {error_msg}: {response.text}")
                return f"Erreur: {error_msg}"
                
        except requests.exceptions.Timeout:
            return "❌ Timeout lors de la génération de la réponse. La question était peut-être trop complexe."
        except requests.exceptions.ConnectionError:
            return "❌ Impossible de se connecter au serveur Ollama. Vérifiez qu'il est démarré avec 'ollama serve'."
        except Exception as e:
            error_msg = f"Erreur lors de la génération: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _clean_response(self, text: str) -> str:
        """Enhanced response cleaning"""
        if not text:
            return "Aucune réponse générée."
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common artifacts and prefixes
        artifacts = [
            "Human:", "Assistant:", "User:", "AI:", "Claude:", "GPT:",
            "Question:", "Answer:", "Response:", "Réponse:", "Question :",
            "Based on the context", "According to the document", "D'après le contexte",
            "Basé sur le contexte", "According to", "D'après"
        ]
        
        # Remove artifacts from the beginning
        for artifact in artifacts:
            if text.startswith(artifact):
                text = text[len(artifact):].strip()
                break
        
        # Remove conversation markers throughout the text
        conversation_markers = [
            '\n\nHuman:', '\n\nUser:', '\n\nQuestion:', '\n\nQ:', 
            '\n\nAssistant:', '\n\nAI:', '\n\nA:',
            '\n\n---', '\n\n###', 'Context:', 'Source:', 'Contexte:'
        ]
        
        for marker in conversation_markers:
            if marker in text:
                text = text.split(marker)[0]
        
        # Clean up any remaining formatting issues
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize multiple newlines
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?', ':', '…')):
            # Find the last complete sentence
            sentences = re.split(r'[.!?]+', text)
            if len(sentences) > 1 and sentences[-1].strip():
                # If the last part looks incomplete, remove it
                if len(sentences[-1].strip()) < 15 or not sentences[-1].strip()[0].isupper():
                    text = '.'.join(sentences[:-1])
                    if not text.endswith(('.', '!', '?')):
                        text += '.'
                else:
                    text = text.rstrip() + '.'
        
        # Final cleanup
        text = text.strip()
        
        # Return meaningful response or fallback
        if len(text) < 10:
            return "Je n'ai pas pu générer une réponse appropriée. Veuillez reformuler votre question."
        
        return text

# Keep the original classes for compatibility
OllamaLLM = IntelligentOllamaLLM

class LocalVectorStore:
    def __init__(self):
        """Initialize FAISS vector store"""
        self.index = None
        self.documents = []
        self.dimension = 384  # MiniLM dimension
        print("✅ Local vector store initialized")
    
    def add_documents(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        """Add documents to the vector store"""
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents with metadata
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            doc_data = {
                'text': text,
                'index': len(self.documents),
                **meta
            }
            self.documents.append(doc_data)
        
        print(f"✅ Added {len(texts)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Perform search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Return results with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
    def get_count(self) -> int:
        """Get number of documents in the store"""
        return len(self.documents)

# Enhanced PDF processing functions
def extract_text_from_pdf(pdf_bytes: bytes, filename: str) -> List[Dict]:
    """Enhanced PDF text extraction with better handling"""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Try different extraction methods
            text = page.get_text()
            
            # If text extraction failed, try OCR-like approach
            if not text.strip():
                # Get text with layout preservation
                text = page.get_text("text")
                
            # If still no text, try blocks
            if not text.strip():
                blocks = page.get_text("blocks")
                text = "\n".join([block[4] for block in blocks if block[4].strip()])
            
            if text.strip():  # Only add non-empty pages
                # Clean up the text
                text = clean_extracted_text(text)
                pages.append({
                    'text': text,
                    'page_number': page_num + 1,
                    'filename': filename
                })
        
        doc.close()
        return pages
        
    except Exception as e:
        print(f"❌ Error processing PDF {filename}: {str(e)}")
        return []

def clean_extracted_text(text: str) -> str:
    """Clean extracted text from PDFs"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Page numbers
    text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text)  # Page headers
    text = text.replace('–', '-') # Em dash to hyphentext =  text = text.replace("’", "'").replace("‘", "'")
    text = text.replace('"', '"').replace('"', '"')  # Curly quotes
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks
    return text.strip()

def chunk_pages(pages: List[Dict], chunk_size: int = 1200, overlap: int = 200) -> List[Dict]:
    """Enhanced chunking with better boundary detection"""
    chunks = []
    
    for page in pages:
        text = page['text']
        
        # Simple chunking by characters with smart boundaries
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            # Try to break at natural boundaries
            if end < len(text):
                # Look for sentence boundaries
                sentence_ends = [m.start() for m in re.finditer(r'[.!?]+\s+', chunk_text)]
                if sentence_ends:
                    # Use the last sentence boundary in the latter half
                    valid_ends = [pos for pos in sentence_ends if pos > len(chunk_text) * 0.6]
                    if valid_ends:
                        boundary = valid_ends[-1] + 1
                        end = start + boundary
                        chunk_text = text[start:end]
                
                # If no good sentence boundary, try paragraph breaks
                if end == start + chunk_size:
                    para_breaks = [m.start() for m in re.finditer(r'\n\s*\n', chunk_text)]
                    if para_breaks:
                        valid_breaks = [pos for pos in para_breaks if pos > len(chunk_text) * 0.6]
                        if valid_breaks:
                            boundary = valid_breaks[-1]
                            end = start + boundary
                            chunk_text = text[start:end]
                
                # Finally, try word boundaries
                if end == start + chunk_size:
                    last_space = chunk_text.rfind(' ')
                    if last_space > len(chunk_text) * 0.7:
                        end = start + last_space
                        chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'page_number': page['page_number'],
                    'filename': page['filename'],
                    'chunk_id': len(chunks)
                })
            
            start = end - overlap
            if start >= len(text):
                break
    
    return chunks

def retrieve_relevant_chunks(question: str, embedder: LocalEmbedder, vector_store: LocalVectorStore, top_k: int = 5) -> List[Dict]:
    """Enhanced retrieval with better relevance scoring"""
    try:
        # Generate query embedding
        query_embedding = embedder.embed_query(question)
        
        # Search for similar chunks
        results = vector_store.search(query_embedding, top_k)
        
        # Filter results by minimum relevance threshold
        min_score = 0.1  # Adjust based on your needs
        filtered_results = [result for result in results if result.get('score', 0) > min_score]
        
        return filtered_results
        
    except Exception as e:
        print(f"❌ Error in retrieval: {e}")
        return []

def generate_intelligent_answer(question: str, context_chunks: List[Dict], llm: IntelligentOllamaLLM, mode: str = "enhanced") -> str:
    """
    Generate intelligent answers with different modes:
    - 'enhanced': Context + general knowledge (default)
    - 'general': Pure general knowledge
    - 'context_only': Only use provided context
    """
    try:
        return llm.generate_intelligent_response(question, context_chunks, mode)
    except Exception as e:
        print(f"❌ Error generating intelligent answer: {e}")
        return f"Erreur lors de la génération de la réponse: {str(e)}"

# Keep backward compatibility
def generate_direct_answer(question: str, context_chunks: List[Dict], llm) -> str:
    """Backward compatibility function"""
    if isinstance(llm, IntelligentOllamaLLM):
        return generate_intelligent_answer(question, context_chunks, llm, "enhanced")
    else:
        # Fallback for old LLM class
        return f"Error: LLM type not supported. Expected IntelligentOllamaLLM, got {type(llm)}"

# Example usage and testing functions
def main():
    """Example usage of the chatbot components"""
    try:
        print("🚀 Starting Local Chatbot Components Test")
        
        # Initialize components
        print("\n1. Initializing Embedder...")
        embedder = LocalEmbedder()
        
        print("\n2. Initializing Vector Store...")
        vector_store = LocalVectorStore()
        
        print("\n3. Initializing LLM...")
        llm = IntelligentOllamaLLM()
        
        print("\n✅ All components initialized successfully!")
        print("📝 You can now use these components in your main application.")
        
        return embedder, vector_store, llm
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return None, None, None

if __name__ == "__main__":
    main()
