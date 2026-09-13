# Bailian model failover

This project can use Bailian/DashScope for both chat models and embedding
models. When the current free quota is exhausted, run the manual Codex
automation named `Bailian manual model failover`.

## Manual flow

1. Open Chrome and sign in to Bailian.
2. Open the model usage page:
   `https://bailian.console.aliyun.com/cn-beijing?tab=model#/model-usage?modelType=Text`
3. Run the paused Codex automation manually.
4. The automation checks the current `.env.local` model names, inspects Bailian
   model usage, chooses same-type models that still have usable free quota, and
   enables the Bailian protection option named like `免费额度用完即停`.
5. The automation calls `scripts/bailian_model_failover.py` to update
   `.env.local`.

## Local switch command

The automation should call the script with explicit model names after it has
verified the page:

```bash
python scripts/bailian_model_failover.py \
  --llm-model qwen-turbo \
  --embedding-model text-embedding-v4 \
  --embedding-dimensions 1024
```

The script updates only model routing keys:

- `OPENAI_BASE_URL`
- `LLM_MODEL`
- `RAG_SYNTH_MODEL`
- `EMBEDDING_PROVIDER`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSIONS`
- `RAG_COLLECTION_NAME`

It does not change API keys. Secret values are redacted in command output.

## Embedding collection note

Changing the embedding model or dimensions creates a separate Chroma collection
name. If the new collection is empty, rebuild the index:

```bash
python rag_ingest.py --rebuild
```
