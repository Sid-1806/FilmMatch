FROM node:24-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY filmmatch ./filmmatch
RUN python -m filmmatch.download ml-latest-small \
    && python -m filmmatch freeze --archive data/ml-latest-small.zip \
    && python -m filmmatch train --archive data/ml-latest-small.zip \
    && python -m filmmatch.content \
    && python -m zipfile -e data/ml-latest-small.zip /tmp/movielens \
    && cp /tmp/movielens/ml-latest-small/README.txt artifacts/model/License.txt \
    && rm -rf data /tmp/movielens
COPY --from=frontend /build/dist ./frontend/dist
COPY benchmarks ./benchmarks
RUN groupadd -r filmmatch && useradd -r -g filmmatch filmmatch
USER filmmatch
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "filmmatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
