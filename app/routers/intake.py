from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import uuid

from ..database import get_db
from ..middleware.auth_middleware import get_current_user, require_auth
from ..models.user import User
from ..models.intake import PatientIntake
from ..services.intake_pdf import generate_intake_pdf

router = APIRouter(tags=["intake"])
templates = Jinja2Templates(directory="app/templates")


# --- Pydantic schema for intake save ---
class IntakeSaveRequest(BaseModel):
    name: str
    age: int
    sex: str
    chief_complaint: str
    duration: str
    past_symptoms: Optional[str] = None
    previous_medications: Optional[str] = None
    previous_tests: Optional[str] = None
    previous_visit: bool = False


# --- Page route ---
@router.get("/intake", response_class=HTMLResponse)
async def intake_page(request: Request, user: User = Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/auth/login")
    return templates.TemplateResponse("dashboard/intake.html", {"request": request, "user": user})


# --- API: Save intake ---
@router.post("/api/v1/intake/save")
async def save_intake(
    data: IntakeSaveRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_auth),
):
    intake = PatientIntake(
        user_id=user.id,
        name=data.name,
        age=data.age,
        sex=data.sex,
        chief_complaint=data.chief_complaint,
        duration=data.duration,
        past_symptoms=data.past_symptoms,
        previous_medications=data.previous_medications,
        previous_tests=data.previous_tests,
        previous_visit=data.previous_visit,
    )
    db.add(intake)
    db.commit()
    db.refresh(intake)
    return {"intake_id": str(intake.id)}


# --- API: Generate & stream PDF ---
@router.get("/api/v1/intake/{intake_id}/pdf")
async def get_intake_pdf(
    intake_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_auth),
):
    intake = (
        db.query(PatientIntake)
        .filter(PatientIntake.id == intake_id, PatientIntake.user_id == user.id)
        .first()
    )
    if not intake:
        raise HTTPException(status_code=404, detail="Intake record not found")

    pdf_buffer = generate_intake_pdf(intake)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="patient_report.pdf"'
        },
    )
