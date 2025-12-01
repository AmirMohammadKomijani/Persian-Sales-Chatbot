# MegaChat - Quick Start Guide

Get your Persian sales chatbot running in 5 minutes!

## Prerequisites

- Python 3.9+
- Docker & Docker Compose
- OpenAI API key

## Step 1: Clone and Setup

```bash
cd megachat
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI + Uvicorn
- LangChain + LangGraph
- Qdrant client
- Redis
- Sentence Transformers
- Hazm (Persian NLP)
- OpenAI

## Step 3: Configure Environment

1. Copy the example env file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-key-here
```

## Step 4: Start Docker Services

```bash
docker-compose up -d
```

This starts:
- **Qdrant** (vector database) on port 6333
- **Redis** (cache) on port 6379
- **PostgreSQL** (database) on port 5432

Verify services are running:
```bash
docker ps
```

## Step 5: Ingest Sample Data

Load posts from your dataset into Qdrant:

```bash
python scripts/ingest_data.py --dataset ../dataset/sampled_posts
```

This will:
- Read all JSON files in the dataset folder
- Convert posts to products
- Generate embeddings
- Store in Qdrant

## Step 6: Start the API

```bash
python main.py
```

The API will start on `http://localhost:8000`

## Step 7: Test the Chatbot

### Using curl:

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "text": "3D'E"
  }'
```

### Test queries:

```bash
# Greeting
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "text": "3D'E"}'

# Price check
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "text": "BÌE* 'ÌF E-5HD †F/G"}'

# Availability
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "text": "EH,H/ G3*"}'
```

### Check health:

```bash
curl http://localhost:8000/api/v1/health
```

## Step 8: View API Documentation

Open your browser and visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing the Pipeline

The RAG pipeline processes requests as follows:

1. **Cache Check** ’ Looks for cached response
2. **Preprocessing** ’ Normalizes Persian text
3. **Intent Detection** ’ Classifies intent (price_check, availability, etc.)
4. **Retrieval** ’ Searches Qdrant for relevant products
5. **Reranking** ’ Scores and ranks results
6. **Generation** ’ Creates Persian response with LLM
7. **Caching** ’ Stores for future requests

## Common Commands

### Stop services:
```bash
docker-compose down
```

### View logs:
```bash
docker-compose logs -f
```

### Clear Redis cache:
```bash
docker exec -it megachat_redis redis-cli FLUSHALL
```

### Check Qdrant collections:
```bash
curl http://localhost:6333/collections
```

## Troubleshooting

### "Connection refused" errors
- Check Docker services are running: `docker ps`
- Restart services: `docker-compose restart`

### Slow first request
- First request loads ML models (embeddings, reranker)
- Subsequent requests will be much faster

### No results returned
- Make sure you ingested data: `python scripts/ingest_data.py`
- Check Qdrant has data: `curl http://localhost:6333/collections/products`

## Next Steps

1. **Add more data**: Ingest your full product catalog
2. **Tune parameters**: Adjust retrieval/reranking settings in `.env`
3. **Customize intents**: Modify intent patterns in `app/services/intent.py`
4. **Improve prompts**: Edit response templates in `app/services/generator.py`
5. **Monitor performance**: Add logging and metrics

## Need Help?

Check the full [README.md](README.md) for detailed documentation.
