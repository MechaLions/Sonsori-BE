from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from .database import engine, get_db
from .models import User
from .schemas import UserCreate, UserLogin, UserResponse
from .database import Base
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
import mediapipe as mp


# 비밀번호 해싱을 위한 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# DB 초기화
Base.metadata.create_all(bind=engine)

tags_metadata = [
    {
        "name": "유저 API",
        "description": "유저 관련 API",
    },
    {
        "name": "수어 API",
        "description": "수어 관련 API",
    }
]

# FastAPI 앱 생성
app = FastAPI(
    title="MecaLions",
    description="수어 번역 및 음성 번역 서비스 API 문서",
    version="1.0.0",
    openapi_tags=tags_metadata
)

# CORS 설정 (필요시)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 커스텀 422 에러 핸들러
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"message": "필수 필드가 누락되었습니다.", "details": exc.errors()},
    )

# 패스워드 해싱 함수
def hash_password(password: str):
    return pwd_context.hash(password)

# 패스워드 검증 함수
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


"""
수어 인식 API 파트
"""
# SocketIO 설정
sio = SocketManager(app=app, cors_allowed_origins=[])

# Mediapipe와 관련된 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# LSTM 모델 로드 및 초기화
actions = np.array(['None', '계산', '고맙다', '괜찮다', '기다리다', '나', '네', '다음',
                    '달다', '더', '도착', '돈', '또', '맵다', '먼저', '무엇', '물', '물음',
                    '부탁', '사람', '수저', '시간', '아니요', '어디', '얼마', '예약', '오다',
                    '우리', '음식', '이거', '인기', '있다', '자리', '접시', '제일', '조금',
                    '주문', '주세요', '짜다', '책', '추천', '화장실', '확인'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#model.load_weights("/home/ubuntu/model/actionxhand_data0524_0513.h5")  # 모델 파일 경로
model.load_weights("‪C:/Users/user/python/Sign-Language-Translator/actionxhand_data0524_0513.h5")  # 모델 파일 경로

# LabelEncoder 및 기타 모델 로드
#rlf = joblib.load("/home/ubuntu/model/sentence_model.pkl")
#data = pd.read_excel("/home/ubuntu/model/sentence_data.xlsx", engine='openpyxl')
rlf = joblib.load("C:/Users/user/python/Sign-Language-Translator/sentence_model.pkl")
data = pd.read_excel("C:/Users/user/python/Sign-Language-Translator/sentence_data.xlsx", engine='openpyxl')
data_x = data.drop(['sentence'], axis=1)
data_y = data['sentence']
le = LabelEncoder()
le.fit(data['sentence'])

# 수어 번역 WebSocket 핸들러
@sio.on('image')
async def image(sid, data_image):
    global sequence, sentence, predictions, count
    sequence, sentence, predictions = [], [], []
    count = 0
    threshold = 0.5

    if data_image == "delete":
        if len(sentence) != 0:
            sequence.clear()
            count = 0
            delete_word = sentence[-1] + "가 삭제되었습니다."
            sentence.pop(-1)
            await sio.emit('delete_back', delete_word)
        else:
            await sio.emit('delete_back', "번역된 단어가 없습니다.")
        return

    # 수어 번역 로직
    frame = readb64(data_image)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        count += 1
        image, results = mediapipe_detection(frame, holistic)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        if len(sequence) % 30 == 0:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0 and actions[np.argmax(res)] != 'None' and actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
            if len(sentence) == 5:
                input_data = make_num_df(make_word_df(*sentence[:5]))
                y_pred = rlf.predict(input_data)
                predict_word = le.inverse_transform(y_pred)
                sentence.clear()
                await sio.emit('result', predict_word)
            else:
                await sio.emit('response_back', sentence[-1])




"""
유저 API 파트
"""
# 아이디 중복 확인 API (CONFLICT, 409 반환, 422 커스터마이징 추가)
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
             },
             422: {
                 "description": "VALIDATION ERROR",
                 "content": {
                     "application/json": {
                         "example": {"message": "필수 필드가 누락되었습니다."}
                     }
                 }
             }
         })
def check_id_duplicate(user_login_id: str, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_login_id == user_login_id).first()
    if db_user:
        raise HTTPException(status_code=409, detail={"message": "아이디가 이미 존재합니다."})
    return {"message": "사용 가능한 아이디입니다."}

# 회원가입 엔드포인트 (user 객체 반환 및 example value 통일)
@app.post("/register", 
          response_model=UserResponse, 
          summary="회원가입", 
          tags=["유저 API"], 
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {
                              "user_id": 1,
                              "user_login_id": "example_id",
                              "name": "example_name"
                          }
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
              },
              422: {
                  "description": "VALIDATION ERROR",
                  "content": {
                      "application/json": {
                          "example": {"message": "필수 필드가 누락되었습니다."}
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
        name=user.name
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# 로그인 엔드포인트 (user 객체 반환 및 example value 통일)
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
                          "example": {
                              "user_id": 1,
                              "user_login_id": "example_id",
                              "name": "example_name"
                          }
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
              },
              422: {
                  "description": "VALIDATION ERROR",
                  "content": {
                      "application/json": {
                          "example": {"message": "필수 필드가 누락되었습니다."}
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

