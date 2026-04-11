from sqlalchemy import Column, DateTime, ForeignKey, Text, JSON, UUID
import uuid
from datetime import datetime
from ..database import Base

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
