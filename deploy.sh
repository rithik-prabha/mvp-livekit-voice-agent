#!/bin/bash
echo "Deploying your Sparkout RAG Voice Agent to LiveKit Cloud..."
livekit-cli deploy-agent \
  --url $LIVEKIT_URL \
  --api-key $LIVEKIT_API_KEY \
  --api-secret $LIVEKIT_API_SECRET \
  .
echo "DEPLOYED! Your agent is now live forever."