# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY python_dash_IA.py .
COPY app.py .
COPY table_hat_game.csv .
COPY assets/ ./assets/

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Set environment variables
ENV DASH_DEBUG_MODE=False
ENV HOST=0.0.0.0
ENV PORT=7860

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "2", "--timeout", "120", "app:server"]
