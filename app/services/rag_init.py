import os
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from src.helper import download_hugging_face_embeddings
from ..config import settings
from pinecone import Pinecone

class RAGManager:
    _instance = None
    _chain = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def _ensure_initialized(self):
        if not self.initialized:
            self._initialize()
            self.initialized = True

    def _initialize(self):
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # We use the existing helper for embeddings
            embeddings = download_hugging_face_embeddings()
            
            # Load the existing index
            self.vectorstore = PineconeVectorStore(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=settings.PINECONE_API_KEY
            )
            
            # Initialize Gemini
            self.llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.3
            )
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            raise e

    def get_retriever(self):
        self._ensure_initialized()
        return self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def get_llm(self):
        self._ensure_initialized()
        return self.llm

rag_manager = RAGManager()
