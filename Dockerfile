FROM python:3.10-slim

WORKDIR /app

# Install dependencies first for Docker layer caching
COPY requirements.txt .

# We filter out UI libraries like streamlit to keep the backend image lightweight
RUN grep -v "streamlit" requirements.txt > backend_reqs.txt
RUN pip install --no-cache-dir -r backend_reqs.txt

# Copy the rest of the application
COPY . .

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
