from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    email      = Column(String, unique=True, index=True, nullable=False)
    name       = Column(String, nullable=False)
    password   = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    papers = relationship("Paper", back_populates="owner")


class Paper(Base):
    __tablename__ = "papers"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename         = Column(String, nullable=False)
    original_name    = Column(String, nullable=False)
    file_path        = Column(String, nullable=False)
    vectorstore_path = Column(String, nullable=False)
    chunk_count      = Column(Integer, default=0)
    uploaded_at      = Column(DateTime, default=datetime.utcnow)

    # Cached insights (stored as JSON string)
    title         = Column(String, nullable=True)
    authors       = Column(String, nullable=True)
    year          = Column(String, nullable=True)
    problem       = Column(Text, nullable=True)
    methodology   = Column(Text, nullable=True)
    dataset       = Column(String, nullable=True)
    contributions = Column(Text, nullable=True)
    results       = Column(Text, nullable=True)
    limitations   = Column(Text, nullable=True)

    owner    = relationship("User", back_populates="papers")
    messages = relationship("ChatMessage", back_populates="paper")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(Integer, primary_key=True, index=True)
    paper_id   = Column(Integer, ForeignKey("papers.id"), nullable=False)
    role       = Column(String, nullable=False)  # "user" or "assistant"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    paper = relationship("Paper", back_populates="messages")