import os
from pydantic import BaseSettings, Field, ValidationError

class Settings(BaseSettings):
    LIVEKIT_URL: str
    LIVEKIT_API_KEY: str
    LIVEKIT_API_SECRET: str

    AWS_REGION: str = "ap-south-1"
    BEDROCK_MODEL_ID: str = "meta.llama3-8b-instruct-v1:0"
    BEDROCK_KB_ID: str

    DYNAMODB_TABLE: str = "VoiceAgentSessions"

    DEEPGRAM_API_KEY: str
    ELEVENLABS_API_KEY: str
    ELEVENLABS_VOICE_ID: str = "EXAVITQu4vr4xnSDxMaL"

    VAD_SILENCE_DURATION: float = 0.6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
    print("All environment variables loaded successfully!")
except ValidationError as e:
    print("Missing or invalid environment variables:")
    for error in e.errors():
        print(f"- {error['loc'][0]}: {error['msg']}")
    raise SystemExit(1)