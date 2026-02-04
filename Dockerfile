# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Run
ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "ai_review.main:app", "--host", "0.0.0.0", "--port", "8000"]
