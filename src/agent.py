# # src/agent.py
# from livekit import agents
# from livekit.agents import VoicePipelineAgent, JobContext
# from livekit.plugins import deepgram, elevenlabs, silero
# from .utils.env import settings
# from .dynamodb_logger import load_history, save_turn
# from .rag_tool import process_user_message  # ← YOUR FULL LOGIC

# async def entrypoint(ctx: JobContext):
#     await ctx.connect_auto()

#     async def handle_session(session: agents.AgentSession):
#         room_id = session.room.name
#         history = await load_history(room_id)
#         chat_history = [{"role": m["role"], "content": m["text"]} for m in history]

#         agent = VoicePipelineAgent(
#             vad=silero.VAD.load(),
#             stt=deepgram.STT(),
#             tts=elevenlabs.TTS(voice_id=settings.ELEVENLABS_VOICE_ID),
#             allow_interruptions=True,
#         )

#         @agent.on("user_speech")
#         async def on_user_speech(text: str):
#             response = await process_user_message(text, chat_history)
#             chat_history.extend([
#                 {"role": "user", "content": text},
#                 {"role": "assistant", "content": response}
#             ])
#             await save_turn(room_id, "user", text)
#             await save_turn(room_id, "assistant", response)
#             await session.playback.say(response, allow_interruptions=True)

#         await agent.start(session)

#     ctx.on_session_started(handle_session)

# src/agent.py — FINAL VERSION THAT WORKS WITH YOUR CURRENT PACKAGES

# src/agent.py
import sys
from pathlib import Path

# Add project root to Python path (makes imports work when running directly)
sys.path.append(str(Path(__file__).parent.parent))

# Now use ABSOLUTE imports (not relative dots)
from livekit import agents, rtc
from livekit.agents import Agent
from livekit.plugins.deepgram import STT
from livekit.plugins.elevenlabs import TTS

# These are now absolute imports → WILL WORK
from src.utils.env import settings
from src.dynamodb_logger import load_history, save_turn
from src.rag_tool import process_user_message

import asyncio

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect_auto()

    async def on_participant_connected(participant: rtc.RemoteParticipant):
        room_id = ctx.room.name
        
        # Load conversation from DynamoDB
        history = await load_history(room_id)
        chat_history = [{"role": m["role"], "content": m["text"]} for m in history]

        # Agent with Deepgram VAD + STT (Silero not needed)
        agent = Agent(
            stt=STT(),
            tts=TTS(voice_id=settings.ELEVENLABS_VOICE_ID),
            chat_ctx=chat_history,
            allow_interruptions=True,
            interrupt_min_words=3,
            interrupt_silence_duration=0.7,
        )

        @agent.on("user_transcribed")
        async def on_user_transcribed(transcription):
            text = transcription.text.strip()
            if not text:
                return

            print(f"\nUSER: {text}")

            # Memory
            await save_turn(room_id, "user", text)
            chat_history.append({"role": "user", "content": text})

            # YOUR FULL ENTERPRISE RAG + INTENT + GROUNDING
            response = await process_user_message(text, chat_history)

            await save_turn(room_id, "assistant", response)
            chat_history.append({"role": "assistant", "content": response})

            print(f"ASSISTANT: {response}\n")

            await agent.say(response, allow_interruptions=True)

        await agent.start(ctx.room, participant)

    ctx.room.on("participant_connected", on_participant_connected)
    print("SPARKOUT VOICE AGENT IS LIVE AND READY!")
    print("Go to: https://aws-voice-agent-fnvtbjkm.livekit.cloud → Join room → Speak!")

if __name__ == "__main__":
    from livekit.agents.cli import run_app
    run_app(entrypoint)