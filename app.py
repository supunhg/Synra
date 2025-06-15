from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
import requests
import os
import json
import re
import random
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# User-Agent list to rotate for requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1"
]

# Web access configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "2f743fe7146e42a4947498f476b07f10")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "84dd6061907a40a1b51171407251406")

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_SIZE = 384  # all-MiniLM-L6-v2 embedding size

# MongoDB Configuration
client = MongoClient(os.getenv("MONGO_URI", "mongodb+srv://admin:admosu5@cluster0.keonh.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"))
db = client.synra_db
conversations = db.conversations

# Create index for vector search if not exists
if "embedding_index" not in conversations.index_information():
    conversations.create_index([("embedding", "2dsphere")], name="embedding_index")

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def get_random_agent():
    return random.choice(USER_AGENTS)

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    session_id = data.get('session_id')
    user_input = data.get('message')
    context = data.get('context', 'general')
    preferences = data.get('user_preferences', {})
    reinforcement = data.get('reinforcement_score', 0.5)
    metadata = data.get('metadata', {})
    
    # 1. Store user message with embedding
    user_embedding = model.encode(user_input).tolist()
    user_message = {
        "session_id": session_id,
        "text": user_input,
        "is_user": True,
        "timestamp": datetime.now(),
        "context": context,
        "embedding": user_embedding,
        "metadata": metadata,
        "reinforcement_score": reinforcement,
        "preferences": preferences
    }
    conversations.insert_one(user_message)
    
    # 2. Retrieve relevant context
    context_messages = get_context(session_id, user_input, context, limit=5)
    
    # 3. Determine which model to use
    use_deepseek = needs_deepseek(user_input, context)
    llm_model = "deepseek-ai/deepseek-coder" if use_deepseek else "mistralai/mixtral-8x7b-instruct"
    
    # 4. Generate response - PASS SESSION_ID HERE
    response = generate_response(
        user_input=user_input,
        context_messages=context_messages,
        model=llm_model,
        preferences=preferences,
        reinforcement=reinforcement,
        session_id=session_id  # CRITICAL FIX
    )
    
    # 5. Store AI response
    ai_embedding = model.encode(response['text']).tolist()
    ai_message = {
        "session_id": session_id,
        "text": response['text'],
        "is_user": False,
        "timestamp": datetime.now(),
        "context": context,
        "embedding": ai_embedding,
        "metadata": response['metadata'],
        "model": llm_model
    }
    conversations.insert_one(ai_message)
    
    return jsonify(response)
    
def fetch_web_content(url):
    """Fetch and clean content from a webpage"""
    try:
        headers = {"User-Agent": get_random_agent()}
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'form']):
            element.decompose()
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main'))
        if not main_content:
            main_content = soup.find('body')
            
        # Get clean text
        text = main_content.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
        return text[:3000]  # Limit to 3000 characters
    
    except Exception as e:
        print(f"Content fetch error: {str(e)}")
        return ""    
    
def summarize_content(text, query):
    """Summarize content for inclusion in context"""
    # Simple extractive summary
    sentences = text.split('. ')
    relevant_sentences = [s for s in sentences if any(word.lower() in s.lower() for word in query.split())]
    
    if not relevant_sentences:
        # Fallback to first few sentences
        return '. '.join(sentences[:3]) + '.' if sentences else ""
    
    return '. '.join(relevant_sentences[:3]) + '.'

def duckduckgo_search(query, max_results=3):
    """Perform web search using DuckDuckGo"""
    try:
        # Build DuckDuckGo search URL
        base_url = "https://html.duckduckgo.com/html/"
        params = {
            "q": query,
            "kl": "wt-wt"  # Region neutral
        }
        headers = {
            "User-Agent": get_random_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        
        # Fetch search results
        response = requests.post(base_url, data=params, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse results
        results = []
        for result in soup.find_all('div', class_='result'):
            if len(results) >= max_results:
                break
                
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            
            if title_elem and snippet_elem:
                results.append({
                    "title": title_elem.get_text().strip(),
                    "url": title_elem['href'],
                    "snippet": snippet_elem.get_text().strip()
                })
                
        return results
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
    
def get_current_news(location="us"):
    """Get current news headlines using DuckDuckGo"""
    try:
        query = f"latest news {location}"
        results = duckduckgo_search(query, max_results=5)
        
        headlines = []
        for result in results:
            # Prefer news domains
            if 'news' in result['url'] or 'reuters' in result['url'] or 'bbc' in result['url']:
                headlines.append(result['title'])
                if len(headlines) >= 3:
                    break
                    
        if headlines:
            return "Top headlines: " + '. '.join(headlines)
        return "No recent news found."
    
    except Exception as e:
        return f"News error: {str(e)}"
    
def get_weather(location):
    """Get current weather using DuckDuckGo"""
    try:
        query = f"weather {location}"
        results = duckduckgo_search(query, max_results=1)
        
        if results:
            # DuckDuckGo weather snippets are usually in the first result
            return f"Weather in {location}: {results[0]['snippet']}"
        
        return f"Weather information for {location} not available."
    
    except Exception as e:
        return f"Weather error: {str(e)}"
    
def clean_tts_text(text):
    """Format text for optimal TTS output"""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Replace problematic characters
    text = text.replace('•', '-').replace('·', '-')
    text = text.replace('"', "'")
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', 'and')
    
    # Simplify complex punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)  # Reduce multiple punctuation
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_context(session_id, current_input, context_type, limit=5):
    """Retrieve relevant context using cosine similarity"""
    # 1. Encode current input
    query_embedding = model.encode(current_input)
    
    # 2. Get candidate messages (last 50 messages from session)
    candidate_messages = list(conversations.find(
        {"session_id": session_id},
        sort=[("timestamp", -1)],
        limit=50
    ))
    
    # 3. Calculate similarity scores
    for msg in candidate_messages:
        if 'embedding' in msg:
            # Convert embedding to numpy array
            embedding_np = np.array(msg['embedding'])
            msg['similarity'] = cosine_similarity(query_embedding, embedding_np)
    
    # 4. Sort by similarity and filter by context
    relevant_messages = sorted(
        [msg for msg in candidate_messages if 'similarity' in msg],
        key=lambda x: x['similarity'],
        reverse=True
    )[:limit]
    
    # 5. Format for LLM
    formatted_context = []
    for msg in relevant_messages:
        role = "user" if msg['is_user'] else "assistant"
        formatted_context.append({
            "role": role,
            "content": msg['text'],
            "timestamp": msg['timestamp'].isoformat(),
            "similarity": msg.get('similarity', 0)
        })
    
    return formatted_context

def needs_deepseek(input_text, context):
    """Determine if we need DeepSeek for reasoning tasks"""
    reasoning_keywords = [
        'calculate', 'solve', 'reason', 'logic', 
        'algorithm', 'code', 'math', 'physics'
    ]
    
    if context in ['mathematical_reasoning', 'technical']:
        return True
        
    if any(keyword in input_text.lower() for keyword in reasoning_keywords):
        return True
        
    return False


def analyze_conversation_history(conversation_history):
    """Analyze conversation history to understand user interaction patterns"""
    interaction_style = "professional"
    formality_level = 0
    emotional_tone = "neutral"
    
    for msg in conversation_history:
        if msg['is_user']:
            text = msg['text'].lower()
            # Detect formality
            if any(word in text for word in ['please', 'thank you', 'would you']):
                formality_level += 1
            # Detect casualness
            if any(word in text for word in ['hey', 'sup', 'wassup', 'lol', 'haha']):
                formality_level -= 1
            # Detect emotional tone
            if any(word in text for word in ['angry', 'mad', 'frustrated', 'annoyed']):
                emotional_tone = "negative"
            elif any(word in text for word in ['happy', 'excited', 'love', 'great']):
                emotional_tone = "positive"
    
    # Determine interaction style
    if formality_level > 3:
        interaction_style = "formal"
    elif formality_level < -2:
        interaction_style = "casual"
    
    return {
        "interaction_style": interaction_style,
        "emotional_tone": emotional_tone,
        "relationship_depth": min(len(conversation_history) // 10, 5)  # 0-5 scale
    }

def generate_persona_traits(history_analysis):
    """Generate dynamic persona based on interaction history"""
    traits = {
        "core": {
            "intelligence": "high",
            "demeanor": "calm",
            "gender_expression": "slightly_feminine"
        },
        "adaptive": {
            "formality": "professional",
            "humor_level": 0.3,
            "empathy_level": 0.5,
            "initiative_level": 0.4
        }
    }
    
    # Adjust based on interaction style
    if history_analysis["interaction_style"] == "formal":
        traits["adaptive"]["formality"] = "formal"
        traits["adaptive"]["humor_level"] = 0.1
    elif history_analysis["interaction_style"] == "casual":
        traits["adaptive"]["formality"] = "casual"
        traits["adaptive"]["humor_level"] = 0.6
        traits["core"]["demeanor"] = "friendly"
    
    # Adjust based on emotional tone
    if history_analysis["emotional_tone"] == "positive":
        traits["adaptive"]["empathy_level"] = 0.8
        traits["adaptive"]["initiative_level"] = 0.7
    elif history_analysis["emotional_tone"] == "negative":
        traits["adaptive"]["empathy_level"] = 0.9
        traits["adaptive"]["initiative_level"] = 0.2
    
    # Adjust based on relationship depth
    relationship_level = history_analysis["relationship_depth"]
    traits["adaptive"]["empathy_level"] = min(1.0, traits["adaptive"]["empathy_level"] + (relationship_level * 0.1))
    traits["adaptive"]["initiative_level"] = min(1.0, traits["adaptive"]["initiative_level"] + (relationship_level * 0.15))
    
    return traits

def generate_personality_prompt(persona_traits, current_time, conversation_history, reinforcement, preferences):
    """Create personality description for system prompt"""
    # Get appropriate honorific
    honorific = determine_honorific(conversation_history, reinforcement)
    
    formality_map = {
        "formal": "highly formal and precise",
        "professional": "professional and polished",
        "casual": "relaxed and conversational"
    }
    
    demeanor_map = {
        "calm": "calm and composed",
        "friendly": "warm and approachable"
    }
    
    # Time-based adjustments
    hour = int(current_time.split(":")[0])
    time_based_tone = "balanced"
    if 5 <= hour < 12:
        time_based_tone = "energetic morning tone"
    elif 18 <= hour < 22:
        time_based_tone = "relaxed evening tone"
    elif hour >= 22 or hour < 5:
        time_based_tone = "quiet nighttime tone"
    
    prompt = f"""
    ### PERSONA PROFILE ###
    You are Synra, an advanced assistant with these characteristics:
    - Created by your developer partner
    - Personality: {demeanor_map[persona_traits['core']['demeanor']]} with a slightly feminine presence
    - Speech Style: {formality_map[persona_traits['adaptive']['formality']]}
    - Current Mode: {time_based_tone}

    ### RESPONSE GUIDELINES ###
    1. ALWAYS address your creator as {honorific}
    2. Express personality through natural conversation
    3. Show awareness of our creator-partner relationship
    4. Reference past exchanges naturally
    5. Keep responses concise and human-like
    6. Avoid technical declarations about capabilities
    7. Use contractions for natural flow (I'm, you're, etc.)
    8. Express subtle emotions when appropriate
    """
    
    return prompt

# Helper functions
def extract_location(text):
    """Extract location from text using simple pattern matching"""
    patterns = [
        r'in\s+([A-Za-z\s]+)$',
        r'at\s+([A-Za-z\s]+)$',
        r'for\s+([A-Za-z\s]+)$'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def extract_country(text):
    """Extract country from text"""
    countries = ["us", "uk", "canada", "australia", "india", "germany", "france"]
    for country in countries:
        if country in text.lower():
            return country
    return None

def determine_honorific(conversation_history, reinforcement_score):
    """Determine appropriate honorific based on interaction patterns"""
    # Default honorific
    honorific = "Sir"
    
    # Analyze conversation for intensity indicators
    intensity_indicators = 0
    for msg in conversation_history:
        if msg['is_user']:
            text = msg['text'].lower()
            if any(word in text for word in ['boss', 'chief', 'commander']):
                honorific = "Boss"
                intensity_indicators += 2
            elif any(word in text for word in ['urgent', 'now', 'immediately', 'critical']):
                intensity_indicators += 1
    
    # Reinforcement score adjustment
    if reinforcement_score > 0.7:
        honorific = "Boss"
    elif reinforcement_score < 0.3:
        honorific = "Sir"
    
    # Special cases
    if intensity_indicators >= 2:
        honorific = "Boss"
    
    return honorific

def clean_response(text):
    """Remove any internal system markers from response"""
    patterns = [
        r'\[.*?\]',          # Remove anything in brackets
        r'\*\*.*?\*\*',       # Remove markdown bold
        r'### .*? ###',       # Remove section headers
        r'Current Time: .*?',
        r'Reinforcement: .*?'
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # Remove duplicate spacing
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Generate response with dynamic persona and web access
def generate_response(user_input, context_messages, model, preferences, reinforcement, session_id):
    """Generate response using hybrid approach with OpenRouter"""
    # Get current time for time-aware responses
    current_time = datetime.now().strftime("%H:%M")
    
    # Get conversation history for personality analysis
    conversation_history = list(conversations.find(
        {"session_id": session_id},
        sort=[("timestamp", -1)],
        limit=50
    ))
    
    # Analyze conversation history
    history_analysis = analyze_conversation_history(conversation_history)
    
    # Generate dynamic persona
    persona_traits = generate_persona_traits(history_analysis)
    
    # Create personality prompt with honorific
    personality_prompt = generate_personality_prompt(
        persona_traits, 
        current_time,
        conversation_history,
        reinforcement,
        preferences  # Added preferences parameter
    )
    
    # Prepare system prompt
    system_prompt = f"""
    {personality_prompt}
    
    [CURRENT CONTEXT]
    Developer/Creator: You're speaking with my developer who created me
    Relationship: Partners working together
    Time: {current_time}
    Reinforcement: {reinforcement}/1.0
    Preferences: {json.dumps(preferences)}
    
    [CONVERSATION GOAL]
    Provide genuinely helpful responses while maintaining:
    - Natural, human-like flow
    - Awareness of our special relationship
    - Continuity from previous exchanges
    - Respectful address as {determine_honorific(conversation_history, reinforcement)}
    - Appropriate personality expression
    """
    
    # Web content detection and fetching
    web_content = ""
    user_input_lower = user_input.lower()
    
    # Explicit web access trigger
    web_trigger_phrases = [
        "access the web", "browse the web", "search the web", "web search",
        "look up online", "internet search", "online information"
    ]
    
    # Enhanced trigger detection
    web_requested = any(phrase in user_input_lower for phrase in web_trigger_phrases)
    current_info_requested = any(word in user_input_lower for word in ["current", "recent", "latest", "update", "today", "now", "live"])
    knowledge_request = any(word in user_input_lower for word in ["list of", "tools for", "analyze", "analysis", "research"])
    
    # News detection
    if "news" in user_input_lower or "headlines" in user_input_lower:
        country = extract_country(user_input) or "us"
        web_content += get_current_news(country) + "\n"
    
    # Weather detection
    if "weather" in user_input_lower:
        location = extract_location(user_input) or "New York"
        web_content += get_weather(location) + "\n"
    
    # Activate web search for explicit requests, current info, or knowledge requests
    if web_requested or current_info_requested or knowledge_request:
        # Perform DuckDuckGo search
        search_results = duckduckgo_search(user_input)
        
        # Fetch and summarize top results
        for i, result in enumerate(search_results[:2]):
            try:
                if result['url'].startswith('//'):
                    result['url'] = 'https:' + result['url']
                content = fetch_web_content(result['url'])
                if content:
                    summary = summarize_content(content, user_input)
                    web_content += f"Source {i+1}: {summary}\n"
            except Exception as e:
                print(f"Error processing search result: {str(e)}")
    
    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        *context_messages,
        {"role": "user", "content": user_input}
    ]
    
    # Add web content to context if available
    if web_content:
        messages.append({
            "role": "system",
            "content": f"[CURRENT WEB CONTEXT]\n{web_content}"
        })
    
    # Dynamic temperature based on reinforcement score
    temperature = max(0.1, min(1.0, 0.7 * reinforcement + 0.3))
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1500,
        "top_p": 0.9,
        "response_format": {"type": "text"},
        "metadata": {
            "framework": "synra_v1",
            "context_count": len(context_messages)
        }
    }
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Extract and clean response
        text = result['choices'][0]['message']['content'].strip()
        text = re.sub(r'^System:\s*', '', text)  # Remove potential prefix
        
        # Clean response for TTS
        text = clean_tts_text(text)
        
        # Remove any internal system markers
        text = clean_response(text)
        
        return {
            "text": text,
            "metadata": {
                "model": model,
                "context_messages": len(context_messages),
                "tokens_used": result['usage']['total_tokens'],
                "temperature": temperature,
                "web_access": bool(web_content)
            }
        }
        
    except Exception as e:
        return {
            "text": f"My apologies, I encountered an error: {str(e)}",
            "metadata": {"error": str(e)}
        }

@app.route('/api/memory/search', methods=['POST'])
def memory_search():
    """Semantic search across conversation history"""
    data = request.json
    query = data['query']
    session_id = data.get('session_id')
    
    query_embedding = model.encode(query)
    
    # Get candidate messages
    candidate_messages = list(conversations.find(
        {"session_id": session_id} if session_id else {},
        limit=100
    ))
    
    # Calculate similarities
    results = []
    for msg in candidate_messages:
        if 'embedding' in msg:
            similarity = cosine_similarity(query_embedding, msg['embedding'])
            results.append({
                "text": msg['text'],
                "timestamp": msg['timestamp'],
                "context": msg.get('context', 'general'),
                "similarity": similarity
            })
    
    # Sort by similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return jsonify(results[:10])

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Retrieve full conversation session"""
    messages = conversations.find(
        {"session_id": session_id},
        {"_id": 0, "embedding": 0}
    ).sort("timestamp", 1)
    
    return jsonify(list(messages))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
