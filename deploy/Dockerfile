# Only needed if you deploy somewhere that wants a container (Fly.io, Cloud Run,
# Railway). Render and Streamlit Cloud do not need this.
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY models/ ./models/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
