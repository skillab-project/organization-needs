FROM python:3.11-slim

# Faster, cleaner Python in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# numpy/pandas/scipy/statsmodels ship manylinux wheels, so no compiler is
# normally needed. If a source build is ever required, uncomment the block below.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential gfortran && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + data + tests
COPY service.py .
COPY tests.py .
COPY mapping_of_ESCO_skills.xlsx .

EXPOSE 8000

# service.py exposes `app = FastAPI(...)`
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
