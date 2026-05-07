from fastapi import APIRouter, Request, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import func
from ..database import get_db
from ..middleware.auth_middleware import get_current_user
from ..models.user import User
from ..models.chat import ChatHistory
from ..models.intake import PatientIntake
import os

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def home(request: Request, user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("landing.html", {"request": request})

@router.get("/auth/login", response_class=HTMLResponse)
async def login_page(request: Request, user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("auth/login.html", {"request": request})

@router.get("/auth/register", response_class=HTMLResponse)
async def register_page(request: Request, user: User = Depends(get_current_user)):
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("auth/register.html", {"request": request})

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_index(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/auth/login")
    
    # Stats
    total_questions = db.query(ChatHistory).filter(ChatHistory.user_id == user.id).count()
    total_users = db.query(User).count()
    avg_len_query = db.query(func.avg(func.length(ChatHistory.answer))).filter(ChatHistory.user_id == user.id).scalar() or 0
    total_intakes = db.query(PatientIntake).filter(PatientIntake.user_id == user.id).count()
    
    recent_chats = db.query(ChatHistory).filter(ChatHistory.user_id == user.id).order_by(ChatHistory.timestamp.desc()).limit(10).all()
    recent_intakes = db.query(PatientIntake).filter(PatientIntake.user_id == user.id).order_by(PatientIntake.created_at.desc()).limit(5).all()

    return templates.TemplateResponse("dashboard/index.html", {
        "request": request,
        "user": user,
        "stats": {
            "total_questions": total_questions,
            "total_users": total_users,
            "avg_length": int(avg_len_query),
            "total_intakes": total_intakes,
        },
        "recent_chats": recent_chats,
        "recent_intakes": recent_intakes,
    })

@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, user: User = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/auth/login")
    return templates.TemplateResponse("dashboard/chat.html", {"request": request, "user": user})
