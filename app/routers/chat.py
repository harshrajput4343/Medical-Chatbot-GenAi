from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.chat import ChatHistory
from ..models.user import User
from ..middleware.auth_middleware import require_auth
from ..services.chat_service import chat_service
from typing import List
import uuid

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

@router.post("")
async def chat_endpoint(
    question: str = Body(..., embed=True),
    is_multimodal: bool = Body(False, embed=True),
    db: Session = Depends(get_db),
    user: User = Depends(require_auth)
):
    try:
        response = await chat_service.get_response(question, is_multimodal)
        
        # Save to history
        chat_item = ChatHistory(
            user_id=user.id,
            question=question,
            answer=response["answer"],
            sources=response["sources"]
        )
        db.add(chat_item)
        db.commit()
        
        return {
            "id": str(chat_item.id),
            "answer": response["answer"],
            "sources": response["sources"],
            "response_time_ms": response["response_time_ms"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(
    db: Session = Depends(get_db),
    user: User = Depends(require_auth),
    limit: int = 50
):
    history = db.query(ChatHistory).filter(ChatHistory.user_id == user.id).order_by(ChatHistory.timestamp.desc()).limit(limit).all()
    return history

@router.delete("/history/{chat_id}")
async def delete_history(
    chat_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_auth)
):
    chat_item = db.query(ChatHistory).filter(ChatHistory.id == chat_id, ChatHistory.user_id == user.id).first()
    if not chat_item:
        raise HTTPException(status_code=404, detail="Chat entry not found")
    
    db.delete(chat_item)
    db.commit()
    return {"status": "deleted"}
