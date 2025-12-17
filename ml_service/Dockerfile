FROM python:3.9-slim
WORKDIR /app
COPY app.py .
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pydantic psycopg2-binary prometheus-client
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
