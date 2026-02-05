# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Environment Settings
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

# -----------------------------
# Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install System Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy Requirements First (Docker Cache Optimization)
# -----------------------------
COPY requirements.txt .

# -----------------------------
# Install Python Dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy Project Files
# -----------------------------
COPY . .

# -----------------------------
# Expose Ports
# -----------------------------
EXPOSE 8501
EXPOSE 8000

# -----------------------------
# Health Check (Optional but Professional)
# -----------------------------
HEALTHCHECK CMD curl --fail http://localhost:8000 || exit 1

# -----------------------------
# Run FastAPI + Streamlit Together
# -----------------------------
CMD ["supervisord", "-c", "supervisord.conf"]
 
