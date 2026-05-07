from sqlalchemy import Column, String, Integer, Text, Boolean, DateTime, ForeignKey, UUID
import uuid
from datetime import datetime
from ..database import Base

class PatientIntake(Base):
    __tablename__ = "patient_intakes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    chief_complaint = Column(Text, nullable=False)
    duration = Column(String, nullable=False)
    past_symptoms = Column(Text, nullable=True)
    previous_medications = Column(Text, nullable=True)
    previous_tests = Column(Text, nullable=True)
    previous_visit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
