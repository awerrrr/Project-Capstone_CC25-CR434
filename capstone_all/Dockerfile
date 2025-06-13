# Gunakan base image Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy semua isi project ke dalam container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r capstone_proyek/requirements.txt

# Expose port
EXPOSE 8000

# Jalankan API
CMD ["uvicorn", "capstone_proyek.main:app", "--host", "0.0.0.0", "--port", "8000"]