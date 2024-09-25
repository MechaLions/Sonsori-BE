from pydantic import BaseModel

class UserCreate(BaseModel):
    user_login_id: str
    user_login_password: str
    name: str

class UserLogin(BaseModel):
    user_login_id: str
    user_login_password: str

class UserResponse(BaseModel):
    user_id: int
    user_login_id: str
    name: str

    class Config:
        orm_mode = True

class MyPageResponse(BaseModel):
    my_page_id: int
    quiz_correct_number: int
    shadowing_accuracy_avg: float
    voice_accuracy_avg: float

    class Config:
        orm_mode = True

class CategoryCreate(BaseModel):
    category_name: str
    description: str
    category_image_url: str

class CategoryResponse(BaseModel):
    category_id: int
    category_name: str
    description: str
    category_image_url: str

    class Config:
        orm_mode = True

class SignQuizCreate(BaseModel):
    category_id: int
    answer_sign: str
    sign_url: str

class SignQuizResponse(BaseModel):
    sign_quiz_id: int
    category_id: int
    answer_sign: str
    sign_url: str

    class Config:
        orm_mode = True

class VoiceQuizCreate(BaseModel):
    category_id: int
    answer_voice: str
    answer_text: str

class VoiceQuizResponse(BaseModel):
    voice_quiz_id: int
    category_id: int
    answer_voice: str
    answer_text: str

    class Config:
        orm_mode = True

class WordCreate(BaseModel):
    category_id: int
    word_text: str
    sign_url: str  # 추가
    answer_voice: str  # 추가

class WordResponse(BaseModel):
    word_id: int
    category_id: int
    word_text: str
    sign_url: str  # 추가
    answer_voice: str  # 추가

    class Config:
        orm_mode = True