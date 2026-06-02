FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt setup.py ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image so container startup is fast
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})"

COPY . .

ENV PORT=8080
EXPOSE 8080

# Single worker keeps the in-memory chat history consistent; threads add
# concurrency while sharing that memory. Generous timeout covers model/index
# load during worker boot.
CMD ["sh", "-c", "gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:${PORT} app:app"]
