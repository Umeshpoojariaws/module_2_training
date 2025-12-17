# Use a lightweight base image
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and application code
COPY deployment/app.py .
COPY model/ml_service.pkl ./model/
COPY model/ml_service.pkl /model/ml_service.pkl


# Expose the application port (assuming FastAPI runs on 9699)
EXPOSE 9699

# Command to run the service (use uvicorn or gunicorn for production)
CMD ["python", "app.py"]