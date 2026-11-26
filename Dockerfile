FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 FILMMATCH_MODEL=/app/models/catalog
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY filmmatch ./filmmatch
COPY --from=frontend /build/dist ./frontend/dist
COPY benchmarks ./benchmarks
COPY models ./models
RUN groupadd -r filmmatch && useradd -r -g filmmatch filmmatch
USER filmmatch
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "filmmatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
