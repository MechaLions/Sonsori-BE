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

# 아이디 중복 확인 API (CONFLICT, 409 반환)
@app.get("/checkIdDuplicate/{user_login_id}", 
         summary="아이디 중복 확인", 
         tags=["유저 API"],
         responses={
             200: {
                 "description": "OK",
                 "content": {
                     "application/json": {
                         "example": {"message": "사용 가능한 아이디입니다."}
                     }
                 }
             },
             409: {
                 "description": "CONFLICT",
                 "content": {
                     "application/json": {
                         "example": {"message": "아이디가 이미 존재합니다."}
                     }
                 }
             }
         })
def check_id_duplicate(user_login_id: str, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_login_id == user_login_id).first()
    if db_user:
        raise HTTPException(status_code=409, detail={"message": "아이디가 이미 존재합니다."})
    return {"message": "사용 가능한 아이디입니다."}

# 회원가입 엔드포인트 (CONFLICT, 409 반환)
@app.post("/register", 
          response_model=UserResponse, 
          summary="회원가입", 
          tags=["유저 API"], 
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {"message": "회원가입 성공"}
                      }
                  }
              },
              409: {
                  "description": "CONFLICT",
                  "content": {
                      "application/json": {
                          "example": {"message": "아이디가 이미 존재합니다."}
                      }
                  }
              }
          })
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_login_id == user.user_login_id).first()
    if db_user:
        raise HTTPException(status_code=409, detail={"message": "아이디가 이미 존재합니다."})

    hashed_password = hash_password(user.user_login_password)
    
    new_user = User(
        user_login_id=user.user_login_id,
        user_login_password=hashed_password,
        name=user.name  # name 필드 추가
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# 로그인 엔드포인트 (NOT FOUND, 404 반환)
@app.post("/login", 
          response_model=UserResponse, 
          summary="로그인", 
          tags=["유저 API"], 
          description="로그인을 통해 사용자를 인증합니다.",
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {"message": "로그인 성공"}
                      }
                  }
              },
              404: {
                  "description": "NOT FOUND",
                  "content": {
                      "application/json": {
                          "example": {"message": "아이디 또는 비밀번호가 잘못되었습니다."}
                      }
                  }
              }
          })
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_login_id == user.user_login_id).first()

    if not db_user:
        raise HTTPException(status_code=404, detail={"message": "아이디가 존재하지 않습니다."})

    if not verify_password(user.user_login_password, db_user.user_login_password):
        raise HTTPException(status_code=404, detail={"message": "비밀번호가 일치하지 않습니다."})

    return db_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", port=8000, reload=True)

