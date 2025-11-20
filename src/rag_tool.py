# src/rag_tool.py
import os
import json
import re
import boto3
import traceback
from botocore.exceptions import ClientError
from typing import List, Dict
from .utils.env import settings
from .utils.logger import log

# ================== YOUR EXACT BOTO3 CLIENTS ==================
bedrock = boto3.client('bedrock-runtime', region_name=settings.AWS_REGION)
bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=settings.AWS_REGION)
kb_id = settings.BEDROCK_KB_ID
MODEL_ID = settings.BEDROCK_MODEL_ID

# bedrock = boto3.client('bedrock-runtime', region_name=os.environ['REGION'])
# bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=os.environ['REGION'])
# dynamodb = boto3.resource('dynamodb', region_name=os.environ['REGION'])
# table_name = os.environ['DYNAMODB_TABLE']
# table = dynamodb.Table(table_name)
# kb_id = os.environ['KB_ID']
# max_history = 10

# ================== YOUR FULL UNCHANGED LOGIC ==================
def invoke_general_model(system_prompt, user_prompt, max_gen_len=512, temperature=0.0):
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    body = json.dumps({"prompt": formatted_prompt, "max_gen_len": max_gen_len, "temperature": temperature, "top_p": 0.9})
    try:
        response = bedrock.invoke_model(modelId=MODEL_ID, contentType='application/json', accept='application/json', body=body)
        result = json.loads(response['body'].read())
        return result['generation'].strip()
    except ClientError as e:
        log.error(f"[ERROR] InvokeModel failed: {str(e)}")
        return "I'm having trouble thinking right now."

def normalize_query(query):
    print(f"[NORMALIZE] Original query: '{query}'")
    corrections = {'saprkout': 'sparkout','sprakout': 'sparkout','sparkot': 'sparkout','menctioned': 'mentioned','mentionned': 'mentioned','studys': 'studies','studie': 'study','servies': 'services','serrvice': 'service','projet': 'project','projets': 'projects','branc': 'branch','loction': 'location','adress': 'address'}
    normalized = query.lower()
    for wrong, correct in corrections.items():
        if wrong in normalized:
            normalized = normalized.replace(wrong, correct)
            print(f"[NORMALIZE] Fixed: '{wrong}' → '{correct}'")
    normalized = re.sub(r'\.{2,}', '.', normalized)
    normalized = re.sub(r'\?+', '?', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    print(f"[NORMALIZE] Normalized query: '{normalized}'")
    return normalized

def classify_intent_with_context(message, chat_history):
    print(f"\n{'='*80}")
    print(f"[STEP 1] CONTEXT-AWARE INTENT CLASSIFICATION")
    print(f"{'='*80}")
    print(f"[STEP 1] Current message: '{message}'")
    print(f"[STEP 1] Message length: {len(message)} characters")
    print(f"[STEP 1] Chat history entries: {len(chat_history)}")
    
    context_prompt = ""
    if len(chat_history) >= 2:
        recent_history = chat_history[-6:]
        print(f"[STEP 1] Building context from last {len(recent_history)} messages...")
        context_prompt = "=== RECENT CONVERSATION ===\n"
        for idx, entry in enumerate(recent_history, 1):
            role = entry.get('role', 'unknown').capitalize()
            content = entry.get('content', '')
            intent_tag = f" [Previous Intent: {entry.get('intent', 'unknown')}]" if role == 'User' and 'intent' in entry else ""
            context_prompt += f"{role}{intent_tag}: {content}\n"
            print(f"[STEP 1]   Message {idx}: {role} - '{content[:50]}...'")
        context_prompt += "===========================\n\n"
        print(f"[STEP 1] Context built successfully")
    else:
        print(f"[STEP 1] No previous conversation")

    intent_system_prompt = """You are an expert intent classifier for Sparkout Tech Solutions voice agent.

CLASSIFICATION RULES:

1. **greetings** - Use ONLY when:
   - Pure greeting with NO question (Hi, Hello, Hey, Good morning)
   - 2-5 words maximum
   - No request for information

2. **rag** - Use when question is about Sparkout company:
   - Company services, products, projects
   - Team, clients, case studies
   - Office locations, branches, contact info
   - Company history, expertise, technology stack
   - ANY mention of "Sparkout" or "your company" or you

3. **smart_ai_assistant** - Use for general questions:
   - Technical how-to questions
   - Programming, coding help
   - General knowledge (not about Sparkout)
   - Explanations of concepts

CRITICAL CONTEXT RULE:
- If you see conversation history above, READ IT CAREFULLY
- If current message refers to or continues previous topic, MAINTAIN the same intent
- Only switch intent if the current message clearly introduces a NEW, UNRELATED topic

OUTPUT FORMAT:
Reply with EXACTLY ONE WORD ONLY: greetings OR rag OR smart_ai_assistant
No explanations, no punctuation."""

    intent_user_prompt = f"""{context_prompt}CURRENT MESSAGE: {message}

Based on the conversation history above (if any) and the current message, classify the intent.

Intent (one word only):"""
    
    print(f"[STEP 1] Calling LLM for intent classification...")
    intent_output = invoke_general_model(intent_system_prompt, intent_user_prompt, max_gen_len=50, temperature=0.0)
    
    intent = intent_output.strip().lower()
    intent = re.sub(r'^(intent:|output:|label:|classification:|answer:)\s*', '', intent).strip()
    intent = re.sub(r'[.,!?;:\'\"]', '', intent)
    intent = intent.split()[0] if intent else "smart_ai_assistant"
    intent = intent.replace(' ', '_').replace('-', '_')
    
    print(f"[STEP 1] LLM raw output: '{intent_output}'")
    print(f"[STEP 1] Parsed intent: '{intent}'")
    
    valid_intents = ["greetings", "rag", "smart_ai_assistant"]
    if intent not in valid_intents:
        print(f"[STEP 1] WARNING: Invalid intent '{intent}', defaulting to 'smart_ai_assistant'")
        intent = "smart_ai_assistant"
    
    print(f"[STEP 1] FINAL INTENT: '{intent}'")
    print(f"{'='*80}\n")
    return intent

def handle_greeting_intent(message):
    print(f"\n{'='*80}")
    print(f"[STEP 2] HANDLING GREETING INTENT")
    print(f"{'='*80}")
    greeting_system_prompt = """You are a friendly assistant at Sparkout Tech Solutions.
Respond warmly to greetings in 2-3 sentences.
Mention that you can help with information about Sparkout's services, projects, or general technical questions."""
    greeting_user_prompt = f"User said: '{message}'\n\nReply warmly and offer help:"
    response_text = invoke_general_model(greeting_system_prompt, greeting_user_prompt, max_gen_len=200, temperature=0.3)
    print(f"[STEP 2] Greeting response: '{response_text}'")
    print(f"{'='*80}\n")
    return response_text

def handle_rag_intent(message, chat_history):
    print(f"\n{'='*80}")
    print(f"[STEP 3] HANDLING RAG INTENT (KNOWLEDGE BASE)")
    print(f"{'='*80}")
    print(f"[STEP 3] Original query: '{message}'")
    
    message_cleaned = normalize_query(message)
    enhanced_query = message_cleaned
    
    vague_indicators = ['where', 'location', 'address', 'there', 'here', 'you', 'u', 'exact', 'which', 'that', 'this', 'it', 'they', 'what', 'how']
    message_lower = message_cleaned.lower()
    has_vague_word = any(word in message_lower for word in vague_indicators)
    should_enhance = len(message_cleaned.split()) <= 10 or has_vague_word
    
    is_related = False
    previous_rag_message = ""
    if len(chat_history) >= 2:
        for i in range(len(chat_history) - 1, -1, -1):
            entry = chat_history[i]
            if entry.get('role') == 'user' and entry.get('intent') == 'rag':
                previous_message = entry.get('content', '')
                previous_cleaned = normalize_query(previous_message)
                if 'case study' in message_lower:
                    break
                if len(previous_cleaned.split()) < 5:
                    continue
                related_keywords = ['branch', 'office', 'location', 'address','clients'] if any(kw in previous_cleaned for kw in ['branch', 'office', 'location']) else ['case study', 'project', 'client', 'service']
                is_related = any(kw in message_lower for kw in related_keywords) and any(kw in previous_cleaned for kw in related_keywords)
                if is_related:
                    previous_rag_message = previous_cleaned
                    break
    
    if should_enhance and is_related and previous_rag_message:
        enhanced_query = f"{previous_rag_message} {message_cleaned}"
    elif should_enhance and 'sparkout' not in message_lower:
        enhanced_query = f"Sparkout Tech Solutions {message_cleaned}"
    elif 'sparkout' not in message_lower and len(message_cleaned.split()) < 8:
        enhanced_query = f"Sparkout {message_cleaned}"
    
    conversation_context = ""
    if is_related and len(chat_history) >= 2:
        recent_history = chat_history[-2:]
        conversation_context = "Previous conversation:\n"
        for entry in recent_history:
            role = entry.get('role', '').capitalize()
            content = entry.get('content', '')
            conversation_context += f"{role}: {content}\n"
        conversation_context += f"\nCurrent question: {message}\n\n"
    
    print(f"[STEP 3] Final KB query: '{enhanced_query}'")
    
    rag_model_arn = f"arn:aws:bedrock:{settings.AWS_REGION}::foundation-model/{MODEL_ID}"
    
    try:
        bedrock_input = {
            "input": {"text": enhanced_query},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": kb_id,
                    "modelArn": rag_model_arn,
                    "retrievalConfiguration": {"vectorSearchConfiguration": {"numberOfResults": 20, "overrideSearchType": "SEMANTIC"}},
                    "generationConfiguration": {
                        "promptTemplate": {
                            "textPromptTemplate": f'''{conversation_context}You are a knowledgeable representative of Sparkout Tech Solutions.

READ THE INFORMATION BELOW CAREFULLY. If it contains the answer, YOU MUST USE IT.

MANDATORY GROUNDING RULES:
1. READ ALL THE INFORMATION BELOW - If ANY part answers the question, use it
2. Look for names, titles, roles, locations, services - they ARE the answer
3. NEVER say "I don't have information" if the answer is in the retrieved content

4. Even partial matches or related information should be used to answer
Example 1:  
Question: "Do you have branch in bangalore...?"
Retrieved Info: Coimbator and us based info
CORRECT: no we have branch in Coimbator and usa

RESPONSE RULES:
- Speak naturally as "we" or "Sparkout Tech Solutions"
- NEVER mention "documents", "search results", "retrieved", "according to", "based on"
- Be direct and confident with your answers
- Keep it conversational (2-4 sentences)

CRITICAL - STUDY THESE EXAMPLES:

Example 1:
Question: "Who is the tech architect?"
Retrieved Info: "praveen kumar - Lead Architect at Sparkout"
CORRECT: "praveen kumar is our Lead Architect."

Example 2:
Question: "Who is the COO?"
Retrieved Info: "Yokesh Sankar serves as Co-Founder and COO"
CORRECT: "Yokesh Sankar is our Co-Founder and Chief Operating Officer."



RETRIEVED INFORMATION (READ THIS CAREFULLY):
$search_results$

USER QUESTION: $query$

YOUR ANSWER (Use the retrieved information above):'''
                        },
                        "inferenceConfig": {"textInferenceConfig": {"temperature": 0.1, "topP": 0.9, "maxTokens": 1500}}
                    }
                }
            }
        }
        
        print(f"[STEP 3] Calling Bedrock retrieve_and_generate...")
        retrieve_response = bedrock_agent.retrieve_and_generate(**bedrock_input)
        
        response_text = retrieve_response.get('output', {}).get('text', '').strip()
        
        # Your exact cleaning logic (meta-phrases)
        meta_phrases = ["according to the retrieved information","according to the retrieved documents","according to the information","according to the documents","according to the search results","based on the search results","based on the retrieved information","based on the retrieved documents","based on the retrieved","based on the documents","based on the information","the retrieved documents mention","the retrieved information shows","the retrieved documents show","the documents mention","the documents show","the search results show","the information shows","i found that","i found information","it is mentioned that","it appears that","from the documents","from the retrieved information","from the search results","as per the documents","as mentioned in","so refer to","please refer to","you can refer to","refer to the"]
        response_lower = response_text.lower()
        for phrase in meta_phrases:
            if phrase in response_lower:
                idx = response_lower.find(phrase)
                response_text = response_text[:idx].strip()
                response_text = re.sub(r'[.,;:]+$', '', response_text).strip() + '.'
                response_text = response_text[0].upper() + response_text[1:]
                break
        
        if not response_text or len(response_text) < 10:
            response_text = "I don't have specific information about that."
        
        print(f"[STEP 3] Final RAG response: '{response_text}'")
        print(f"{'='*80}\n")
        return response_text
        
    except Exception as e:
        log.error("RAG failed", error=str(e))
        return "I couldn't retrieve information right now. Please try again."

def handle_smart_ai_assistant_intent(message):
    print(f"\n{'='*80}")
    print(f"[STEP 4] HANDLING SMART AI ASSISTANT")
    print(f"{'='*80}")
    smart_system_prompt = """You are a helpful technical assistant at Sparkout Tech Solutions.

1. General Technical & Project Questions:
   - Provide clear, structured explanations
   - Offer step-by-step guidance
   - Share knowledge about project architecture
   - Discuss technology trends (NO code)
   - Never provide code snippets

2. Sparkout Company Queries:
   - Encourage them to ask specific questions about services/projects

3. Out-of-Scope:
   - If unrelated to projects/architecture/technology, respond:
     "That's out of scope. Please ask about our company or project guidance."
"""
    response_text = invoke_general_model(smart_system_prompt, f"Question: {message}\n\nProvide a helpful response:", max_gen_len=1024, temperature=0.3)
    print(f"[STEP 4] Response: '{response_text[:300]}...'")
    print(f"{'='*80}\n")
    return response_text

# ================== LIVEKIT MAIN FUNCTION ==================
async def process_user_message(message: str, chat_history: List[Dict]) -> str:
    intent = classify_intent_with_context(message, chat_history)
    chat_history.append({"role": "user", "content": message, "intent": intent})
    
    if intent == "greetings":
        return handle_greeting_intent(message)
    elif intent == "rag":
        return handle_rag_intent(message, chat_history)
    else:
        return handle_smart_ai_assistant_intent(message)