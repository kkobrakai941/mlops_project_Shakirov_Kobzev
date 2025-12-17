FROM python:3.9-slim
WORKDIR /app
COPY app.py /app/app.py
RUN pip install --no-cache-dir streamlit requests pandas psycopg2-binary
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
