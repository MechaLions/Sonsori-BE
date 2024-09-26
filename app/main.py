from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi_socketio import SocketManager
from passlib.context import CryptContext
import mediapipe as mp
import random
import numpy as np
from sqlalchemy.orm import Session
from .database import engine, get_db, Base
from .models import User, MyPage, Word, Category  # 데이터베이스 모델들
from .schemas import UserCreate, UserLogin, UserResponse, WordCreate, WordResponse, CategoryCreate, CategoryResponse  # 스키마
from tensorflow.keras.models import Sequential  # LSTM 모델
from tensorflow.keras.layers import LSTM, Dense  # LSTM 레이어
from sklearn.preprocessing import LabelEncoder  # 정답 텍스트 인코딩
import joblib  # 모델 로딩을 위한 joblib
import pandas as pd  # 엑셀 파일 로드용

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
    },
    {
        "name": "음성 API",
        "description": "발음 교정 관련 API",
    },
    {
        "name": "dev API",
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
    "http://127.0.0.1",
    "http://localhost:8000",
    "http://0.0.0.0:8000",
    "http://127.0.0.1:8000",
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
#model.load_weights("‪C:/Users/user/python/Sign-Language-Translator/actionxhand_data0524_0513.h5")  # 모델 파일 경로
model.load_weights("/app/models/actionxhand_data0524_0513.h5")


# LabelEncoder 및 기타 모델 로드
#rlf = joblib.load("/home/ubuntu/model/sentence_model.pkl")
#data = pd.read_excel("/home/ubuntu/model/sentence_data.xlsx", engine='openpyxl') #모델 파일 경료
#rlf = joblib.load("C:/Users/user/python/Sign-Language-Translator/sentence_model.pkl")
#data = pd.read_excel("C:/Users/user/python/Sign-Language-Translator/sentence_data.xlsx", engine='openpyxl') #모델 파일 경료
rlf = joblib.load("/app/models/sentence_model.pkl")
data = pd.read_excel("/app/models/sentence_data.xlsx", engine='openpyxl')

data_x = data.drop(['sentence'], axis=1)
data_y = data['sentence']
le = LabelEncoder()
le.fit(data['sentence'])

# 소켓 연결 확인 이벤트
@sio.on('connect')
async def connect(sid, environ):
    print(f"Client {sid} connected")
    await sio.emit('connection_response', {'message': 'Socket connected!'})

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

#수어 번역 텍스트와 정답 텍스트 비교 및 정확도 계산 API
@app.post("/shadowing/calculateAccuracy",
          summary="수어 번역 텍스트와 정답 텍스트 비교 및 정확도 계산",
          tags=["수어 API"],
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {
                              "word_id": 1,
                              "translated_text": "안녕하세요",
                              "accuracy": 100
                          }
                      }
                  }
              },
              404: {
                  "description": "NOT FOUND",
                  "content": {
                      "application/json": {
                          "example": {"message": "Word 정보를 찾을 수 없습니다."}
                      }
                  }
              }
          })
async def calculate_accuracy(user_id: int, word_id: int, translated_text: str, db: Session = Depends(get_db)):
    word = db.query(Word).filter(Word.word_id == word_id).first()
    if not word:
        raise HTTPException(status_code=404, detail={"message": "Word 정보를 찾을 수 없습니다."})

    correct_text = word.word_text  # DB에서 정답 텍스트 가져오기
    correct_words = correct_text.split(" ")
    translated_words = translated_text.split(" ")
    
    correct_count = sum(1 for word in translated_words if word in correct_words)
    accuracy = (correct_count / len(correct_words)) * 100 if correct_words else 0  # %로 변환

    # 사용자 MyPage 정보 업데이트
    my_page = db.query(MyPage).filter(MyPage.user_id == user_id).first()
    if my_page:
        my_page.shadowing_accuracy_sum += accuracy
        my_page.shadowing_solved_number += 1
        db.commit()
    else:
        raise HTTPException(status_code=404, detail={"message": "MyPage 정보를 찾을 수 없습니다."})

    return {
        "word_id": word_id,
        "correct_text": correct_text,
        "translated_text": translated_text,
        "accuracy": int(accuracy)  # 소수점 제거하여 정수로 반환
    }


#카테고리 ID를 통한 평균 정확도 저장 API
@app.post("/shadowing/saveShadowingAccuracy",
          summary="카테고리 ID를 통한 평균 정확도 저장",
          tags=["수어 API"],
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {"message": "평균 정확도가 성공적으로 업데이트되었습니다."}
                      }
                  }
              },
              404: {
                  "description": "NOT FOUND",
                  "content": {
                      "application/json": {
                          "example": {"message": "MyPage 정보를 찾을 수 없습니다."}
                      }
                  }
              },
              400: {
                  "description": "BAD REQUEST",
                  "content": {
                      "application/json": {
                          "example": {"message": "풀이한 문제가 없어 평균을 계산할 수 없습니다."}
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
async def saveShadowingAccuracy(user_id: int, category_id: int, db: Session = Depends(get_db)):
    my_page = db.query(MyPage).filter(MyPage.user_id == user_id).first()
    if my_page:
        if my_page.shadowing_solved_number > 0:
            my_page.shadowing_accuracy_avg = my_page.shadowing_accuracy_sum / my_page.shadowing_solved_number
            my_page.shadowing_category_id = category_id
            my_page.shadowing_solved_number = 0
            my_page.shadowing_accuracy_sum = 0.0
            db.commit()
        else:
            raise HTTPException(status_code=400, detail={"message": "풀이한 문제가 없어 평균을 계산할 수 없습니다."})
    else:
        raise HTTPException(status_code=404, detail={"message": "MyPage 정보를 찾을 수 없습니다."})

    return {"message": "평균 정확도가 성공적으로 업데이트되었습니다."}
    

#특정 카테고리 ID로 10개의 랜덤 단어 반환 API
@app.get("/shadowing/{category_id}/words", 
         summary="특정 카테고리에서 10개 랜덤 단어 반환", 
         tags=["수어 API"],
         responses={
             200: {
                 "description": "OK",
                 "content": {
                     "application/json": {
                         "words": [
                                 {"word_id": 1, "word_text": "사과", "sign_url": "http://example.com/sign1"},
                                 {"word_id": 2, "word_text": "바나나", "sign_url": "http://example.com/sign2"}
                        ]
                     }
                 }
             },
             404: {
                 "description": "NOT FOUND",
                 "content": {
                     "application/json": {
                         "example": {"message": "해당 카테고리에서 단어를 찾을 수 없습니다."}
                     }
                 }
             }
         })
async def get_random_words_from_category(category_id: int, db: Session = Depends(get_db)):
    words = db.query(Word).filter(Word.category_id == category_id).all()
    if not words:
        raise HTTPException(status_code=404, detail={"message": "해당 카테고리에서 단어를 찾을 수 없습니다."})
    
    random_words = random.sample(words, min(len(words), 10))
    
    return [{"word_id": word.word_id, "word_text": word.word_text, "sign_url": word.sign_url} for word in random_words]

#카테고리와 상관없이 5개의 랜덤 단어 반환 API
@app.get("/quiz/words/random", 
         summary="카테고리와 상관없이 5개 랜덤 단어 반환", 
         tags=["수어 API"],
         responses={
             200: {
                 "description": "OK",
                 "content": {
                     "application/json": {
                         "example": [
                             {"word_id": 1, "word_text": "사과", "sign_url": "http://example.com/sign1"},
                             {"word_id": 2, "word_text": "바나나", "sign_url": "http://example.com/sign2"}
                         ]
                     }
                 }
             }
         })
async def get_random_words(db: Session = Depends(get_db)):
    words = db.query(Word).all()
    random_words = random.sample(words, min(len(words), 5))
    return {
        "words": [
            {"word_id": word.word_id, "word_text": word.word_text, "sign_url": word.sign_url} 
            for word in random_words
        ]
    }





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

    # MyPage 생성
    new_mypage = MyPage(user_id=new_user.user_id)
    db.add(new_mypage)
    db.commit()
    db.refresh(new_mypage)

    return new_user  # UserResponse 모델이 mypage_id를 포함하도록 설정

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

#특정 유저의 MyPage 조회 API
@app.get("/mypage/{user_id}", 
         summary="특정 유저의 MyPage 조회", 
         tags=["유저 API"],
         responses={
             200: {
                 "description": "OK",
                 "content": {
                     "application/json": {
                         "example": {
                             "name": "홍길동",
                             "quiz_correct_number": 5,
                             "shadowing_accuracy_avg": 80.5,
                             "shadowing_category_name": "과일",
                             "voice_accuracy_avg": 90.2,
                             "voice_category_name": "음식"
                         }
                     }
                 }
             },
             404: {
                 "description": "NOT FOUND",
                 "content": {
                     "application/json": {
                         "example": {"message": "MyPage 정보를 찾을 수 없습니다."}
                     }
                 }
             }
         })
async def get_mypage(user_id: int, db: Session = Depends(get_db)):
    # 유저 조회
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail={"message": "User 정보를 찾을 수 없습니다."})

    # MyPage 조회
    my_page = db.query(MyPage).filter(MyPage.user_id == user_id).first()
    if not my_page:
        raise HTTPException(status_code=404, detail={"message": "MyPage 정보를 찾을 수 없습니다."})

    # Shadowing, Voice 카테고리 이름 조회
    shadowing_category_name = None
    voice_category_name = None
    if my_page.shadowing_category_id:
        shadowing_category = db.query(Category).filter(Category.category_id == my_page.shadowing_category_id).first()
        shadowing_category_name = shadowing_category.category_name if shadowing_category else None

    if my_page.voice_category_id:
        voice_category = db.query(Category).filter(Category.category_id == my_page.voice_category_id).first()
        voice_category_name = voice_category.category_name if voice_category else None

    return {
        "name": user.name,
        "quiz_correct_number": my_page.quiz_correct_number,
        "shadowing_accuracy_avg": my_page.shadowing_accuracy_avg,
        "shadowing_category_name": shadowing_category_name,
        "voice_accuracy_avg": my_page.voice_accuracy_avg,
        "voice_category_name": voice_category_name
    }


"""
dev API 파트
"""
@app.post("/words/", 
          response_model=WordResponse, 
          summary="단어 추가", 
          tags=["dev API"], 
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {
                              "word_id": 1,
                              "category_id": 1,
                              "word_text": "사과",
                              "sign_url": "http://example.com/sign1",
                              "answer_voice": "사과"
                          }
                      }
                  }
              },
              422: {
                  "description": "VALIDATION ERROR",
                  "content": {
                      "example": {"message": "필수 필드가 누락되었습니다."}
                  }
              }
          })
def create_word(word: WordCreate, db: Session = Depends(get_db)):
    new_word = Word(**word.dict())
    db.add(new_word)
    db.commit()
    db.refresh(new_word)
    return new_word

@app.post("/categories/", 
          response_model=CategoryResponse, 
          summary="카테고리 추가", 
          tags=["dev API"], 
          responses={
              201: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {
                              "category_id": 1,
                              "category_name": "과일",
                              "description": "과일 관련 단어 모음",
                              "category_image_url": "http://example.com/image1"
                          }
                      }
                  }
              },
              422: {
                  "description": "VALIDATION ERROR",
                  "content": {
                      "example": {"message": "필수 필드가 누락되었습니다."}
                  }
              }
          })
def create_category(category: CategoryCreate, db: Session = Depends(get_db)):
    new_category = Category(**category.dict())
    db.add(new_category)
    db.commit()
    db.refresh(new_category)
    return new_category

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", port=8000, reload=False)

