FROM python:3.11-slim

WORKDIR /app

# 🔧 System dependencies (needed for numpy / faiss / etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 📦 Copy requirements first (better caching)
COPY requirements.txt .

# 🚀 Install Python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 📁 Copy app
COPY . .

# 🌐 Document port (Render will override via $PORT)
EXPOSE 10000

# ▶️ Start FastAPI (dynamic port support)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
