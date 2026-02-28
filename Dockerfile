# 1. Use a light weight python image
FROM python:3.11-slim

# 2. Set work directory inside container and cds into it
WORKDIR /app

# 3. Copy requirement file into container
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code into contianer
COPY . .

# 6. Expose the port FASTAPI runs on
EXPOSE 8000

# 7. Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

