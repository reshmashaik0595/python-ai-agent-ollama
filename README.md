# Real AI Projects

A small Python project that demonstrates a local AI assistant using function calling with OpenAI-compatible APIs. The assistant can answer questions and invoke simple tools such as getting the current time and checking weather for a city.

## Project overview

This project is built around the idea of an LLM that can use tools during a chat. Instead of only generating text, the assistant can call Python functions when needed.

Key concepts used in this repository:
- Python scripting
- OpenAI-compatible client setup
- Local `.env` configuration
- Tool calling using `tools` metadata
- Conversation memory across turns
- Local model hosting with Ollama

## Files in this project

- `simple_ai_assistance.py` — chatbot loop with tool execution support
- `tools.py` — Python functions and OpenAI-style tool schemas
- `NOTES.md` — setup notes and interview-ready ideas
- `.env` — local environment variables (not committed)

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

3. Create a `.env` file with your local model settings:

```env
BASE_URL=http://localhost:11434/v1
MODEL=llama3.2
```

> The exact model name depends on what you have installed in Ollama.

4. Start Ollama and pull or run a model:

```bash
ollama serve
ollama list
```

## Run the assistant

```bash
python simple_ai_assistance.py
```

Run the chunk-search example with the same activated environment:

```bash
python coding_snippets/chunk_search.py
```

Example prompts:
- "What time is it?"
- "What is the weather in London?"
- "Tell me a short greeting."

## Tools supported

- `get_current_time()`
- `get_city_weather(city)`

## Interview-ready explanation

This project is a practical example of tool calling in an LLM workflow. It demonstrates how a model can decide, at runtime, to call a Python function and then continue the conversation using the returned result. That is a foundational concept behind agent-style AI systems.

## Notes

- Keep secrets and model config in `.env`, not in source code.
- Use a locally hosted model if you want to work offline or test small AI workflows.
- The assistant is intentionally simple and can be extended with extra tools, memory, and workflows.
