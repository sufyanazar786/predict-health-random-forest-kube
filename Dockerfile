FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN test -f diabetes_model.pkl || echo "Run train.py before building image"


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# # Dockerfile
# FROM python:3.10
# WORKDIR /app
# COPY . /app
# RUN pip install -r requirements.txt
# RUN python train.py
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

