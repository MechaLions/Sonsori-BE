from fastapi import FastAPI, Request, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from dotenv import load_dotenv
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
import torch
import soundfile as sf
import requests
import torchaudio.transforms as T
import random
import subprocess
import os
import io
import torchaudio
import numpy as np
from sqlalchemy.orm import Session
from .database import engine, get_db, Base
from .models import User, MyPage, Word, Category  # 데이터베이스 모델들
from .schemas import UserCreate, UserLogin, UserResponse, WordCreate, WordResponse, CategoryCreate, CategoryResponse, CheckIDRequest, WordUpdate, WordListResponse, CategoryUpdate, QuizScoreUpdate, TranslatedTextRequest  # 스키마
from typing import List


#비밀번호 해싱을 위한 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#DB 초기화
Base.metadata.create_all(bind=engine)

#.env 파일서 환경 변수 로드
load_dotenv()

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
        "name": "퀴즈 API",
        "description": "퀴즈 관련 API",
    },
    {
        "name": "카테고리 API",
        "description": "카테고리 전체 목록 조회 API",
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
    "http://localhost:5500",
    "https://sonsori.web.app/",
    "https://sonsori.web.app",
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
#수어 번역 텍스트와 정답 텍스트 비교 및 정확도 계산 API
@app.post("/shadowing/calculateAccuracy/{user_id}/{word_id}",
          summary="수어 번역 텍스트와 정답 텍스트 비교 및 정확도 계산",
          tags=["수어 API"],
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {
                              "word_id": 1,
                              "correct_text": "안녕하세요",
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
async def calculate_accuracy(user_id: int, word_id: int, request: TranslatedTextRequest, db: Session = Depends(get_db)):
    word = db.query(Word).filter(Word.word_id == word_id).first()
    if not word:
        raise HTTPException(status_code=404, detail={"message": "Word 정보를 찾을 수 없습니다."})

    correct_text = word.word_text  # DB에서 정답 텍스트 가져오기
    correct_words = correct_text.split(" ")
    translated_words = request.translated_text.split(" ")
    
    correct_count = sum(1 for word in translated_words if word in correct_words)
    accuracy = (correct_count / len(correct_words)) * 100 if correct_words else 0  # %로 변환

    # 사용자 MyPage 정보 업데이트
    my_page = db.query(MyPage).filter(MyPage.user_id == user_id).first()
    if my_page:
        my_page.shadowing_accuracy_sum += int(accuracy)
        my_page.shadowing_solved_number += 1
        db.commit()
    else:
        raise HTTPException(status_code=404, detail={"message": "MyPage 정보를 찾을 수 없습니다."})

    return {
        "word_id": word_id,
        "correct_text": correct_text,
        "translated_text": request.translated_text,
        "accuracy": int(accuracy)  # 소수점 제거하여 정수로 반환
    }


#카테고리 ID를 통한 평균 정확도 저장 API
@app.post("/shadowing/saveShadowingAccuracy/{user_id}/{category_id}",
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
            my_page.shadowing_accuracy_avg = int(my_page.shadowing_accuracy_sum / my_page.shadowing_solved_number)
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
                         "example": {
                            "words": [
                                 {"word_id": 1, "word_text": "사과", "sign_url": "http://example.com/sign1"},
                                 {"word_id": 2, "word_text": "바나나", "sign_url": "http://example.com/sign2"}
                            ]
                         }
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
async def get_random_words_from_category(category_id: int, db: Session = Depends(get_db)):
    words = db.query(Word).filter(Word.category_id == category_id).all()
    if not words:
        raise HTTPException(status_code=404, detail={"message": "해당 카테고리에서 단어를 찾을 수 없습니다."})
    
    random_words = random.sample(words, min(len(words), 10))
    
    return [{"word_id": word.word_id, "word_text": word.word_text, "sign_url": word.sign_url} for word in random_words]


"""
음성 API 파트
"""
# 1. 특정 카테고리에서 10개 랜덤 음성 단어 반환 API
@app.get("/voice/{category_id}/words", 
         summary="특정 카테고리에서 10개 음성 문제 반환", 
         tags=["음성 API"],
         responses={
             200: {
                 "description": "OK",
                 "content": {
                     "application/json": {
                         "example": {
                             "words": [
                                 {"word_id": 1, "word_text": "사과", "answer_voice": "사과"},
                                 {"word_id": 2, "word_text": "바나나", "answer_voice": "바나나"}
                             ]
                         }
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
async def get_random_voice_words_from_category(category_id: int, db: Session = Depends(get_db)):
    words = db.query(Word).filter(Word.category_id == category_id).all()
    if not words:
        raise HTTPException(status_code=404, detail={"message": "해당 카테고리에서 단어를 찾을 수 없습니다."})
    
    random_words = random.sample(words, min(len(words), 10))
    return {"words": [{"word_id": word.word_id, "word_text": word.word_text, "answer_voice": word.answer_voice} for word in random_words]}


# 2. 음성 파일을 받아 음절별 정확도 계산 및 응답 반환 API
processor = Wav2Vec2Processor.from_pretrained("/app/voice_model/")
model = Wav2Vec2ForCTC.from_pretrained("/app/voice_model/")

def load_audio(file_data: bytes, file_extension: str):
    audio_buffer = io.BytesIO(file_data)

    if file_extension == "mp3":
        # MP3 파일을 WAV로 변환
        with open("temp.mp3", "wb") as f:
            f.write(audio_buffer.getbuffer())
        subprocess.run(["ffmpeg", "-i", "temp.mp3", "-ar", "16000", "-ac", "1", "output.wav"])
        audio_tensor, sample_rate = torchaudio.load("output.wav")
        os.remove("temp.mp3")
        os.remove("output.wav")
    elif file_extension == "wav":
        # WAV 파일은 바로 처리
        audio_tensor, sample_rate = torchaudio.load(audio_buffer)
    else:
        raise ValueError("지원되지 않는 오디오 형식입니다: MP3와 WAV만 지원됩니다")
    
    # 16kHz로 리샘플링
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
    
    return audio_tensor.squeeze().numpy(), 16000


# 모델 추론을 위한 함수
def query_local_model(file_data: bytes, file_extension: str):
    # 오디오 로드 및 변환
    audio_data, sample_rate = load_audio(file_data, file_extension)
    #print(f"Before query squeeze: {audio_data.shape}")
    
    inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    input_values = inputs['input_values']
    #print(f"Shape after adjustment for model input: {input_values.shape}")

    # 모델에 입력하여 추론 수행
    with torch.no_grad():
        logits = model(input_values).logits

    # 예측 결과를 텍스트로 디코딩
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, clean_up_tokenization_spaces=True)[0]
    
    return transcription
    
# 2. 음성 파일을 받아 음절별 정확도 계산 및 응답 반환 API
@app.post("/voice/calculateAccuracy/{user_id}/{word_id}", 
          summary="음성 파일을 받아 음절별 정확도 계산",
          tags=["음성 API"],
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {
                              "word_id": 1,
                              "correct_text": "밥 먹었어",
                              "correct_pronunciation": "밥 머거써",
                              "voice_recognition_result": "밥머거써",
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
async def calculate_voice_accuracy(
    user_id: int, 
    word_id: int, 
    audio_file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """
    음성 파일을 받아서 해당 word_id의 단어와 비교해 음절별 정확도를 계산하고 반환하는 API
    """
    # Word DB에서 word_id에 해당하는 단어 가져오기
    word = db.query(Word).filter(Word.word_id == word_id).first()
    if not word:
        raise HTTPException(status_code=404, detail={"message": "Word 정보를 찾을 수 없습니다."})

    # 정답 텍스트와 올바른 발음 정보
    correct_text = word.answer_voice
    correct_pronunciation = word.answer_voice.replace(" ", "")  # 공백 제거

    # 프론트에서 업로드된 음성 파일을 메모리에서 바로 읽음
    file_data = await audio_file.read()

    # 파일 확장자를 확인
    file_extension = audio_file.filename.split(".")[-1].lower()

    # Use the local model to recognize the audio
    recognized_text = query_local_model(file_data, file_extension).replace(" ", "")  # 공백 제거

    # 음절별로 비교
    correct_characters = list(correct_text)  # 정답 텍스트를 음절 단위로 분리
    recognized_characters = list(recognized_text)  # 음성 인식 결과를 음절 단위로 분리
    
    correct_count = sum(1 for i, char in enumerate(recognized_characters) if i < len(correct_characters) and char == correct_characters[i])
    accuracy = (correct_count / len(correct_characters)) * 100 if correct_characters else 0  # %로 변환

    # MyPage 테이블에서 사용자의 정보 가져와서 정확도 업데이트
    my_page = db.query(MyPage).filter(MyPage.user_id == user_id).first()
    if my_page:
        my_page.voice_accuracy_sum += int(accuracy)
        my_page.voice_solved_number += 1
        db.commit()
    else:
        raise HTTPException(status_code=404, detail={"message": "MyPage 정보를 찾을 수 없습니다."})

    # 최종 응답
    return {
        "word_id": word_id,
        "correct_text": correct_text,
        "correct_pronunciation": correct_pronunciation,  # 옳은 발음 정보
        "voice_recognition_result": recognized_text,  # 음성 인식 결과 (공백 제거된 결과)
        "accuracy": int(accuracy)  # 정확도 반환
    }


# 3. 음성 문제 평균 정확도 저장 API
@app.post("/voice/saveVoiceAccuracy/{user_id}/{category_id}", 
          summary="카테고리 ID를 통한 음성 번역 평균 정확도 저장", 
          tags=["음성 API"],
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
async def save_voice_accuracy(user_id: int, category_id: int, db: Session = Depends(get_db)):
    """
    카테고리 ID를 받아 해당 유저의 음성 번역 문제 평균 정확도를 계산하고 저장하는 API
    """
    # MyPage 테이블에서 사용자 정보 가져오기
    my_page = db.query(MyPage).filter(MyPage.user_id == user_id).first()
    if not my_page:
        raise HTTPException(status_code=404, detail={"message": "MyPage 정보를 찾을 수 없습니다."})

    if my_page.voice_solved_number > 0:
        # 평균 정확도 계산 및 저장
        my_page.voice_accuracy_avg = int(my_page.voice_accuracy_sum / my_page.voice_solved_number)
        my_page.voice_category_id = category_id
        my_page.voice_solved_number = 0
        my_page.voice_accuracy_sum = 0.0
        db.commit()
    else:
        raise HTTPException(status_code=400, detail={"message": "풀이한 문제가 없어 평균을 계산할 수 없습니다."})

    return {"message": "평균 정확도가 성공적으로 업데이트되었습니다."}



"""
카테고리 API 파트
"""
@app.get("/categories", 
         summary="카테고리 전체 목록 조회", 
         tags=["카테고리 API"],
         responses={
             200: {
                 "description": "OK",
                 "content": {
                     "application/json": {
                         "example": {
                             "categories": [
                                 {"category_id": 1, "category_name": "과일", "description": "과일 관련 단어 모음", "category_image_url": "http://example.com/image1"},
                                 {"category_id": 2, "category_name": "동물", "description": "동물 관련 단어 모음", "category_image_url": "http://example.com/image2"}
                             ]
                         }
                     }
                 }
             },
             404: {
                 "description": "NOT FOUND",
                 "content": {
                     "application/json": {
                         "example": {"message": "카테고리가 없습니다."}
                     }
                 }
             }
         })
async def get_all_categories(db: Session = Depends(get_db)):
    categories = db.query(Category).all()
    if not categories:
        return JSONResponse(status_code=404, content={"message": "카테고리가 없습니다."})
    
    # 카테고리 목록을 JSON으로 반환
    return {"categories": [
        {"category_id": category.category_id, "category_name": category.category_name, 
         "description": category.description, "category_image_url": category.category_image_url}
        for category in categories
    ]}
        
"""
퀴즈 API 파트
"""
@app.get("/quiz", 
         summary="퀴즈 10개 묶음 받는 API", 
         tags=["퀴즈 API"],
         responses={
             200: {
                 "description": "퀴즈를 성공적으로 가져왔습니다.",
                 "content": {
                     "application/json": {
                         "example": {
                             "quiz": [
                                 {
                                     "type": "sign_language",
                                     "word_id": 1,
                                     "correct_text": "사과",
                                     "sign_url": "http://example.com/sign1"
                                 },
                                 {
                                     "type": "sign_language",
                                     "word_id": 2,
                                     "correct_text": "바나나",
                                     "sign_url": "http://example.com/sign2"
                                 },
                                 {
                                     "type": "sign_language",
                                     "word_id": 3,
                                     "correct_text": "수박",
                                     "sign_url": "http://example.com/sign3"
                                 },
                                 {
                                     "type": "sign_language",
                                     "word_id": 4,
                                     "correct_text": "포도",
                                     "sign_url": "http://example.com/sign4"
                                 },
                                 {
                                     "type": "sign_language",
                                     "word_id": 5,
                                     "correct_text": "참외",
                                     "sign_url": "http://example.com/sign5"
                                 },
                                 {
                                     "type": "multiple_choice",
                                     "word_id": 6,
                                     "correct_text": "컴퓨터",
                                     "options": [
                                         "모니터",
                                         "키보드",
                                         "마우스",
                                         "컴퓨터"
                                     ]
                                 },
                                 {
                                     "type": "multiple_choice",
                                     "word_id": 7,
                                     "correct_text": "책상",
                                     "options": [
                                         "의자",
                                         "책상",
                                         "소파",
                                         "침대"
                                     ]
                                 },
                                 {
                                     "type": "multiple_choice",
                                     "word_id": 8,
                                     "correct_text": "핸드폰",
                                     "options": [
                                         "핸드폰",
                                         "노트북",
                                         "태블릿",
                                         "카메라"
                                     ]
                                 },
                                 {
                                     "type": "multiple_choice",
                                     "word_id": 9,
                                     "correct_text": "자동차",
                                     "options": [
                                         "자동차",
                                         "자전거",
                                         "버스",
                                         "오토바이"
                                     ]
                                 },
                                 {
                                     "type": "multiple_choice",
                                     "word_id": 10,
                                     "correct_text": "바다",
                                     "options": [
                                         "강",
                                         "호수",
                                         "바다",
                                         "산"
                                     ]
                                 }
                             ]
                         }
                     }
                 }
             },
             404: {
                 "description": "퀴즈를 위한 충분한 단어가 없습니다.",
                 "content": {
                     "application/json": {
                         "example": {"message": "퀴즈 문제를 위한 단어가 충분하지 않습니다."}
                     }
                 }
             }
         })
async def get_quiz(db: Session = Depends(get_db)):
    # 수어 문제 5개 선택 (sign_url이 존재하는 단어)
    sign_language_words = db.query(Word).filter(Word.sign_url.isnot(None)).all()

    if len(sign_language_words) < 5:
        raise HTTPException(status_code=404, detail="수어 문제를 위한 단어가 충분하지 않습니다.")

    sign_language_questions = random.sample(sign_language_words, 5)

    # 객관식 문제 5개 선택
    all_words = db.query(Word).all()
    if len(all_words) < 8:  # 객관식 문제 하나당 4개의 보기가 필요하므로 최소 8개 이상의 단어가 필요
        raise HTTPException(status_code=404, detail="객관식 문제를 위한 단어가 충분하지 않습니다.")

    multiple_choice_questions = []
    for _ in range(5):
        correct_word = random.choice(all_words)
        all_words_except_correct = [word for word in all_words if word.word_id != correct_word.word_id]
        options = random.sample(all_words_except_correct, 3)  # 오답 3개 선택
        options.append(correct_word)  # 정답을 보기 중 하나로 추가
        random.shuffle(options)  # 보기 순서 섞기
        multiple_choice_questions.append({
            "type": "multiple_choice",
            "word_id": correct_word.word_id,
            "correct_text": correct_word.word_text,
            "sign_url": correct_word.sign_url,
            "options": [option.word_text for option in options]
        })

    # 최종 퀴즈 반환 (5개 수어 문제 + 5개 객관식 문제)
    quiz = multiple_choice_questions + [
        {
            "type": "sign_language",
            "word_id": word.word_id,
            "correct_text": word.word_text,
            "sign_url": word.sign_url
        } for word in sign_language_questions
    ]

    return {"quiz": quiz}


@app.post("/quiz/{user_id}/record", 
          summary="퀴즈 점수 기록", 
          tags=["퀴즈 API"],
          responses={
              200: {
                  "description": "OK",
                  "content": {
                      "application/json": {
                          "example": {"message": "퀴즈 점수가 기록되었습니다."}
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
              422: {
                  "description": "VALIDATION ERROR",
                  "content": {
                      "application/json": {
                          "example": {"message": "필수 필드가 누락되었습니다."}
                      }
                  }
              }
          })
async def record_quiz_score(user_id: int, quiz_data: QuizScoreUpdate, db: Session = Depends(get_db)):
    # MyPage 조회
    my_page = db.query(MyPage).filter(MyPage.user_id == user_id).first()
    if not my_page:
        raise HTTPException(status_code=404, detail={"message": "MyPage 정보를 찾을 수 없습니다."})

    # 퀴즈 맞은 문제 수 업데이트
    my_page.quiz_correct_number = quiz_data.quiz_correct_number
    db.commit()

    return {"message": "퀴즈 점수가 기록되었습니다."}



"""
유저 API 파트
"""
# 아이디 중복 확인 API (CONFLICT, 409 반환, 422 커스터마이징 추가)
@app.post("/checkIdDuplicate", 
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
def check_id_duplicate(request: CheckIDRequest, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.user_login_id == request.user_login_id).first()
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
dev API
"""
@app.post("/dev/words/add", 
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

@app.post("/dev/categories/add", 
          response_model=CategoryResponse, 
          summary="카테고리 추가", 
          tags=["dev API"], 
          responses={
              200: {
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

@app.patch("/dev/words/{word_id}",
           response_model=WordResponse,
           summary="단어 수정",
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
               404: {
                   "description": "NOT FOUND",
                   "content": {
                       "application/json": {
                           "example": {"message": "Word not found"}
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
async def update_word(word_id: int, word_data: WordUpdate, db: Session = Depends(get_db)):
    word = db.query(Word).filter(Word.word_id == word_id).first()
    if not word:
        raise HTTPException(status_code=404, detail={"message": "Word not found"})
    
    word.word_text = word_data.word_text
    word.answer_voice = word_data.answer_voice
    word.sign_url = word_data.sign_url

    db.commit()
    db.refresh(word)
    return word


@app.get("/dev/words",
         response_model=List[WordListResponse],
         summary="단어 전체 목록 조회",
         tags=["dev API"],
         responses={
             200: {
                 "description": "OK",
                 "content": {
                     "application/json": {
                         "example": [
                             {
                                 "word_id": 1,
                                 "category_id": 1,
                                 "word_text": "사과",
                                 "sign_url": "http://example.com/sign1",
                                 "answer_voice": "사과"
                             },
                             {
                                 "word_id": 2,
                                 "category_id": 2,
                                 "word_text": "바나나",
                                 "sign_url": "http://example.com/sign2",
                                 "answer_voice": "바나나"
                             }
                         ]
                     }
                 }
             }
         })
async def get_all_words(db: Session = Depends(get_db)):
    words = db.query(Word).all()
    return words

@app.patch("/dev/categories/{category_id}",
           response_model=CategoryResponse,
           summary="카테고리 수정",
           tags=["dev API"],
           responses={
               200: {
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
               404: {
                   "description": "NOT FOUND",
                   "content": {
                       "application/json": {
                           "example": {"message": "Category not found"}
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
async def update_category(category_id: int, category_data: CategoryUpdate, db: Session = Depends(get_db)):
    category = db.query(Category).filter(Category.category_id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail={"message": "Category not found"})
    
    category.category_name = category_data.category_name
    category.description = category_data.description
    category.category_image_url = category_data.category_image_url

    db.commit()
    db.refresh(category)
    return category

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", port=8000, reload=False)

