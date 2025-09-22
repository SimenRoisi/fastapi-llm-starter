from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (String, Integer, DateTime, ForeignKey, func, Text, Index, UniqueConstraint, MetaData)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
# ^ SQLAlchemy 2.x typing-friendly API


naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=naming_convention)

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True) # 320 is RFC cap
    api_key: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        UniqueConstraint("api_key", name="uq_users_api_key")
    )

class ApiUsage(Base):
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    api_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    endpoint: Mapped[str] = mapped_column(String(128), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    __table_args__ = (
        Index("ix_api_usage_api_key_endpoint_time", "api_key", "endpoint", "timestamp"),
    )

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    owner: Mapped["User"] = relationship(backref="documents")