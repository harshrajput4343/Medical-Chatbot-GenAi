from fastapi import Request, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..database import get_db
from ..services.auth_service import AuthService
from ..models.user import User

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        return None
    
    # Remove 'Bearer ' if present
    if token.startswith("Bearer "):
        token = token[7:]
    
    payload = AuthService.decode_token(token)
    if not payload:
        return None
    
    email: str = payload.get("sub")
    if email is None:
        return None
    
    user = db.query(User).filter(User.email == email).first()
    return user

async def require_auth(request: Request, user: User = Depends(get_current_user)):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
