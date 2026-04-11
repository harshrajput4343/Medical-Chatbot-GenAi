from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from ..database import get_db
from ..middleware.auth_middleware import require_auth
from ..models.user import User
from ..config import settings

router = APIRouter(prefix="/api/v1", tags=["general"])

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model": "gemini-1.5-pro",
        "pinecone_index": settings.PINECONE_INDEX_NAME
    }

@router.get("/user/me")
async def get_me(user: User = Depends(require_auth)):
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "created_at": user.created_at
    }
