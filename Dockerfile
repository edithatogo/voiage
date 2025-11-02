# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt .
COPY pyproject.toml .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run the web API by default
CMD ["python", "-m", "voiage.web.main"]