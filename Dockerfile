FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY mplsoccer_viz_server.py .
COPY main.py .

# Create output directory
RUN mkdir -p /app/output/mplsoccer

# Create non-root user
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app

USER mcpuser

# Railway will provide the PORT environment variable
CMD ["python", "main.py"]
