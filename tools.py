"""Tool definitions for the local AI assistant project.

This module exposes small Python functions that can be called by an LLM through
OpenAI-compatible tool calling. Each function is registered in a simple schema
that the model can discover and invoke during a conversation.
"""

from datetime import datetime
from rag import search
from utilities.utilities import log
import requests
import json

def get_current_time():
    """Return the current date and time in a human-readable format."""
    return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

def get_city_weather(city):
    """Fetch the current weather for a given city from the wttr.in API."""
    url = f"https://wttr.in/{city}?format=3"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text

def get_city():
    return "Hyderabad"

def search_knowledge_base(query):
    results = search(query, 1)
    log(f" Knowledge base search: {query}")
    for r in results:
        log(f" {r['score']:.2f} - {r['source']} #{r['chunk_id']}")
    blocks = []
    for r in results:
        source_file_name = r["source"]
        chunk_id = r["chunk_id"]
        chunk_text = r["text"]
        header = f"[{source_file_name} #{chunk_id}]"
        block = f"{header}\n{chunk_text}"
        blocks.append(block)
    return "\n\n".join(blocks)


# OpenAI function-calling schema: the model sees these names and arguments.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time as a formatted string.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Returns the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to get the weather for."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"get_city",
            "description":"",
            "parameters":{
                "type":"object",
                "properties":{}
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"search_knowledge_base",
            "description":"Searches the Zorvex Technologies company knowledge base: leave, medical, HR, hiring, tech, incident, reports and travel expense policies. Use if for any question about company policies for documens. Returns the most relavent passages with their source",
            "parameters":{
                "type":"object",
                "properties":{
                    "query":{
                        "type":"string",
                        "description":"The user's question, rewritten as short, self-contained search query."
                    }
                },
                "required":["query"]
            }
        }
    }
]

# Map tool names to their Python implementations so the assistant can run them.
# Keep these keys aligned with the function names declared in TOOLS.
FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_city_weather": get_city_weather,
    "get_city": get_city,
    "search_knowledge_base":search_knowledge_base
}

def run_tool(tool_call):
    func_name = tool_call.function.name
    func_args = {}
    
    try:
        func_args = json.loads(tool_call.function.arguments or "{}")
        log(f"🔧  {func_name}({func_args})")
        fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
        result = fn(**func_args)
    except Exception as e:
        result = f"Error while running {func_name} : {e}"
    log(f"↪️ {result}")
    return result