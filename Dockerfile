# 1. Pull an optimized, official Python runtime image from DockerHub
FROM python:3.10-slim

# 2. Prevent Python from writing .pyc files to disk and ensure logs print immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Establish the internal working directory within the container
WORKDIR /app

# 4. Install bare-metal OS dependencies required for high-performance calculations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy over the requirements sheet first to optimize container caching layers
COPY requirements.txt .

# 6. Install the precise data science engine frameworks
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your project files into the container directory
COPY . .

# 8. Open network access ports (8000 for FastAPI, 8501 for Streamlit Dashboard)
EXPOSE 8000
EXPOSE 8501

# 9. Fire up both servers concurrently using an inline shell command execution
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0