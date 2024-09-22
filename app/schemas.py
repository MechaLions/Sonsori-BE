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

