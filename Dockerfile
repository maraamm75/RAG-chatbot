# Use a slim Python base image
FROM python:3.10-slim

ENV HOME=/app
WORKDIR /app

COPY requirements.txt .
COPY src/ ./src/

RUN mkdir -p /app/.streamlit && chmod -R 777 /app/.streamlit

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

