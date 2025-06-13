# Gunakan image python resmi
FROM python:3.10-slim

# Set direktori kerja
WORKDIR /app

# Copy semua file dari host ke container
COPY . .

# Upgrade pip dan install dependency
RUN pip install --upgrade pip
RUN pip install -r capstone_proyek/requirements.txt

# Jalankan FastAPI dengan uvicorn
CMD ["uvicorn", "capstone_proyek.main:app", "--host", "0.0.0.0", "--port", "8000"]
