from sqlalchemy import Column, Integer, String
from .database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    user_login_id = Column(String(255), unique=True, index=True, nullable=False)
    user_login_password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)

