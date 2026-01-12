FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy bot
COPY . .

# Env (NE kulcs!)
ENV PYTHONUNBUFFERED=1

# Run
CMD ["python", "bot.py"]
