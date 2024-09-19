from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from .database import engine, get_db
from .models import User
from .schemas import UserCreate, UserLogin, UserResponse
from .database import Base
from fastapi.middleware.cors import CORSMiddleware

# 비밀번호 해싱을 위한 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# DB 초기화
Base.metadata.create_all(bind=engine)

tags_metadata = [
    {
        "name": "유저 API",
        "description": "유저 관련 API",  # 태그에 대한 설명 추가
    }
]

# FastAPI 앱 생성
app = FastAPI(
    title="MecaLions",
    description="수어 번역 및 음석 번역 서비스 API 문서",
    version="1.0.0",
    openapi_tags=tags_metadata
)

# CORS 설정 (필요시)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5137",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 패스워드 해싱 함수
def hash_password(password: str):
    return pwd_context.hash(password)

# 패스워드 검증 함수
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# 회원가입 엔드포인트
@app.post("/register", response_model=UserResponse, summary="회원가입",tags=["유저 API"], description="새로운 사용자를 등록합니다.")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_login_id == user.user_login_id).first()
    if db_user:
        raise HTTPException(status_code=400, detail="아이디가 이미 존재합니다.")

    db_email = db.query(User).filter(User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="이메일이 이미 존재합니다.")

    hashed_password = hash_password(user.user_login_password)
    
    new_user = User(
        user_login_id=user.user_login_id,
        user_login_password=hashed_password,
        email=user.email,
        phone=user.phone
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# 로그인 엔드포인트
@app.post("/login", response_model=UserResponse, summary="로그인",tags=["유저 API"], description="로그인을 통해 사용자를 인증합니다.")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_login_id == user.user_login_id).first()

    if not db_user:
        raise HTTPException(status_code=400, detail="아이디가 존재하지 않습니다.")

    if not verify_password(user.user_login_password, db_user.user_login_password):
        raise HTTPException(status_code=400, detail="비밀번호가 일치하지 않습니다.")

    return db_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", port=8000, reload=True)