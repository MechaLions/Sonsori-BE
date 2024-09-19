from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    user_login_id: str
    user_login_password: str
    email: EmailStr
    phone: str

class UserLogin(BaseModel):
    user_login_id: str
    user_login_password: str

class UserResponse(BaseModel):
    user_id: int
    user_login_id: str
    email: EmailStr
    phone: str

    class Config:
        orm_mode = True
