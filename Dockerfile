

FROM python:3.10-slim

WORKDIR /app

# Copy all our files
COPY iris_fastapi.py .
COPY requirements.txt .
COPY model.joblib .

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8200

# Run the app
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]


