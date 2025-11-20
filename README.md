# mvp-livekit-voice-agent

Fully working Bedrock Llama 3 8B + Bedrock KB RAG + DynamoDB memory voice agent.

## Local Test
```bash
cp .env.example .env
uv sync   # or pip install -r requirements.txt
uv run python src/agent.py dev