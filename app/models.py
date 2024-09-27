from sqlalchemy import Column, Integer, String, ForeignKey, Float
from .database import Base
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    user_login_id = Column(String(255), unique=True, index=True, nullable=False)
    user_login_password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    my_page = relationship("MyPage", back_populates="user", uselist=False)

    @property
    def mypage_id(self):
        return self.my_page.my_page_id if self.my_page else None

class MyPage(Base):
    __tablename__ = "my_page"
    
    my_page_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    quiz_correct_number = Column(Integer, default=0)
    shadowing_category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=True)
    shadowing_accuracy_sum = Column(Float, default=0.0)
    shadowing_solved_number = Column(Integer, default=0)
    shadowing_accuracy_avg = Column(Float, default=0.0)
    voice_category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=True)
    voice_accuracy_sum = Column(Float, default=0.0)
    voice_solved_number = Column(Integer, default=0)
    voice_accuracy_avg = Column(Float, default=0.0)
    
    user = relationship("User", back_populates="my_page")
    shadowing_category = relationship("Category", foreign_keys=[shadowing_category_id])
    voice_category = relationship("Category", foreign_keys=[voice_category_id])

class Category(Base):
    __tablename__ = "categories"
    
    category_id = Column(Integer, primary_key=True, index=True)
    category_name = Column(String(255), nullable=False)
    description = Column(String(255), nullable=False)
    category_image_url = Column(String(255), nullable=True)
    
    sign_quiz = relationship("SignQuiz", back_populates="category")
    voice_quiz = relationship("VoiceQuiz", back_populates="category")

class SignQuiz(Base):
    __tablename__ = "sign_quiz"
    
    sign_quiz_id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=False)
    answer_sign = Column(String(255), nullable=False)
    sign_url = Column(String(255), nullable=False)
    
    category = relationship("Category", back_populates="sign_quiz")

class VoiceQuiz(Base):
    __tablename__ = "voice_quiz"
    
    voice_quiz_id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=False)
    answer_voice = Column(String(255), nullable=False)
    answer_text = Column(String(255), nullable=False)
    
    category = relationship("Category", back_populates="voice_quiz")

class Word(Base):
    __tablename__ = "word"
    
    word_id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=False)
    word_text = Column(String(255), nullable=False)
    sign_url = Column(String(255), nullable=True)  # 추가
    answer_voice = Column(String(255), nullable=True)  # 추가
    
    category = relationship("Category")