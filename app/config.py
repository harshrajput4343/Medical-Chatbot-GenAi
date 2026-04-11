from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Medical Chatbot v2"
    DEBUG: bool = False
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200  # 30 days
    
    # Supabase Settings
    SUPABASE_URL: str
    SUPABASE_KEY: str
    DATABASE_URL: str  # Direct Postgres URL from Supabase for SQLAlchemy
    
    # AI Search & LLM Settings
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "medical-chatbot"
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-3-flash"
    GROQ_API_KEY: Optional[str] = None
    
    # Model config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
