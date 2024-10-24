# Dockerfile
# 베이스 이미지로 Python 사용
FROM python:3.10-slim

# 작업 디렉토리 생성
WORKDIR /app

# 시스템 패키지 업데이트 및 기본 ffmpeg, sox 설치
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg sox

# 시스템 패키지 업데이트
#RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

# 의존성 설치
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt && pip install --no-cache-dir torch==2.0.0+cpu torchaudio==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# 필요한 파일들을 복사
COPY ./app /app/app
COPY ./voice_model /app/voice_model

# .env 파일 복사
COPY .env /app/.env

# Uvicorn을 이용해 FastAPI 서버 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

