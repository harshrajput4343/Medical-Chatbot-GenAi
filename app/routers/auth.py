from fastapi import APIRouter, Depends, HTTPException, status, Response, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from ..database import get_db
from ..models.user import User
from ..services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register")
async def register(
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    name: str = Form(...),
    db: Session = Depends(get_db)
):
    db_user = db.query(User).filter(User.email == email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = AuthService.get_password_hash(password)
    new_user = User(email=email, hashed_password=hashed_password, name=name)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    token = AuthService.create_access_token(data={"sub": new_user.email})
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
    return response

@router.post("/login")
async def login(
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == email).first()
    if not user or not AuthService.verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = AuthService.create_access_token(data={"sub": user.email})
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
    return response

@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/auth/login")
    response.delete_cookie("access_token")
    return response
