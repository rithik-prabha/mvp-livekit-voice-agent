import boto3
import json
from datetime import datetime
from typing import List, Dict
from .utils.env import settings
from .utils.logger import log

dynamodb = boto3.resource('dynamodb', region_name=settings.AWS_REGION)
table = dynamodb.Table(settings.DYNAMODB_TABLE)

async def load_history(room_id: str) -> List[Dict]:
    try:
        response = table.get_item(Key={"room_id": room_id})
        if "Item" in response:
            messages = response["Item"].get("messages", [])
            log.info("Loaded conversation history", room_id=room_id, count=len(messages))
            return messages
    except Exception as e:
        log.error("Failed to load history", error=str(e))
    return []

async def save_turn(room_id: str, role: str, text: str):
    timestamp = datetime.utcnow().isoformat()
    item = {
        "room_id": room_id,
        "timestamp": timestamp,
        "role": role,
        "text": text
    }
    try:
        table.update_item(
            Key={"room_id": room_id},
            UpdateExpression="SET messages = list_append(if_not_exists(messages, :empty), :msg), last_updated = :ts",
            ExpressionAttributeValues={
                ":msg": [item],
                ":empty": [],
                ":ts": timestamp
            }
        )
        log.info("Saved turn", role=role, room_id=room_id)
    except Exception as e:
        log.error("Failed to save to DynamoDB", error=str(e))