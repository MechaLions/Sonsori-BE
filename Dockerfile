# Dockerfile

# 베이스 이미지로 Python 사용
FROM python:3.10-slim

# 작업 디렉토리 생성
WORKDIR /app

# 필요한 파일들을 복사
COPY ./requirements.txt /app/requirements.txt
COPY ./app /app/app
COPY ./mecalions /app/mecalions
COPY ./models /app/models
COPY ./webSocket /app/webSocket

# .env 파일 복사
COPY .env /app/.env

# 의존성 설치
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Uvicorn을 이용해 FastAPI 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
