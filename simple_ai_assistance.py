#!/usr/bin/env python3
"""Simple AI assistant that can call local Python tools during chat."""

# Standard-library import: environment variables are needed to read config values.
import os
from types import SimpleNamespace

# Load environment variables and create an OpenAI-compatible API client.
from dotenv import load_dotenv
from openai import OpenAI

# Import the tool registry, tool executor, and chat persistence helpers.
from tools import TOOLS, run_tool
from utilities.utilities import log
from memory import list_chats, load_chat, save_chat

# Configuration is loaded from .env, keeping credentials and model names out of source.
load_dotenv()

# Limit each user turn to this many model/tool cycles to prevent an infinite loop.
MAX_RUN_STEPS = 3

# Use a shared client for every chat-completion request in this process.
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

# This system message controls the assistant's tone and knowledge-source policy.
SYSTEM_MESSAGE = {
        "role": "system",
        "content": (
            "You are a helpful, humble assistant for Zorvex Technologies employees. "
            "Answer general questions (programming, science, advice) from your own knowledge. "
            "Use get_current_time, get_city_weather and get_user_city only for the current time, the weather, or the user's city. "
            "For ANY question about Zorvex policies, benefits, leave, medical cover, HR, hiring, tech processes, incident reports or travel expenses, "
            "you MUST first call search_knowledge_base. "
            "Answer company questions ONLY from the text that search_knowledge_base returns. Do not use outside knowledge for company facts. "
            "After the answer, cite the source like (leave_policy.txt #3). "
            "If the returned text does not contain the answer, say exactly: I could not find this in the knowledge base. "
            "Keep answers under 100 words."
        )
    }

# Start a fresh conversation with the system policy as the first message.
messages = [SYSTEM_MESSAGE]
chat_file_path = None

# Stream one model turn; separate visible text from reconstructed tool-call fragments.
def stream_chat(history):
    stream = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=history,
        tools=TOOLS,
        stream=True,
    )

    full_response = ""
    calls = {}
    label_printed = False

    # A stream may contain text, tool-call fragments, or neither.
    try:
        for chunck in stream:
            if not chunck.choices:
                continue

            delta = chunck.choices[0].delta

            # Print normal text incrementally instead of waiting for the whole response.
            if delta.content:
                if not label_printed:
                    print(" Assitance: ", end="", flush=True)
                    label_printed = True
                print(delta.content, end="", flush=True)
                full_response = full_response + delta.content

            # OpenAI-compatible providers can split a tool call across chunks.
            # Merge fragments by tool index, then concatenate the name and arguments.
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(
                        id=None,
                        function=SimpleNamespace(
                            name="",
                            arguments=""
                        )
                    )
                )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name = obj.function.name + tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments = obj.function.arguments + tc.function.arguments
    except KeyboardInterrupt:
        # Allow the user to stop the stream cleanly without crashing the app.
        print("\n Stopped by you!!")
        calls = {}

    if label_printed:
        print()  # Move to the next line after the assistant's response.

    # Sort by tool index so the payload preserves the model's call order.
    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])

    return full_response, calls_list

# Turn the collected tool call objects into the exact format the OpenAI API expects.
def build_tool_calls_payload(tool_calls):
    """Convert internal tool-call objects to API assistant-message dictionaries."""
    payload = []
    for tool_call in tool_calls:
        tool_call_dict = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
        }
        payload.append(tool_call_dict)

    return payload

def show_chats(chats):
    """Print saved chats as a numbered list for the resume command."""
    chat_number = 1
    for chat in chats:
        title = chat["title"]
        count = chat["count"]
        updated = chat["updated"]

        print(f"{chat_number}. {title} - {count} messages (Last updated: {updated})")


existing_chats = list_chats()

# Offer to resume the newest saved chat before entering the interactive loop.
if existing_chats:
    last_chat = existing_chats[0]
    answer = input(f"Continue your last chat \"{last_chat['title']}\"? (y/n): ")

    if answer == "y":
        data = load_chat(last_chat["path"])
        if data is not None:
            messages = [SYSTEM_MESSAGE] + data["messages"]
            chat_file_path = last_chat["path"]
            print(f"Resumed: {last_chat['title']} ({len(data['messages'])} messages)")

# Read commands and questions until the user exits or interrupts the process.
while True:
    user_input = input("\n🙋🏻RESHMA_SHAIK: ")

    # Quit the app cleanly on common exit commands.
    if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
        break

    command = user_input.strip().lower()
    if command == '/new':
        # Reset history without deleting the previously saved chat file.
        messages = [SYSTEM_MESSAGE]
        chat_file_path = None
        print("Started a new chat. The old one is still saved.")
        continue

    if command == '/chats':
        # List saved conversations without changing the current history.
        chats = list_chats()
        if chats:
            show_chats(chats)
        else:    
            print("No saved chats yet")
        continue

    if command == '/resume':
        # Replace the in-memory history with the selected saved conversation.
        chats = list_chats()
        if not chats:
            print("No chats to resume.")
            continue

        show_chats(chats)
        choice = input("Enter the chat number to resume (Press Enter to cancel): ")
        choice = choice.strip()

        if not choice:
            continue

        if not choice.isdigit() or not (1 <= int(choice) <= len(chats)):
            print("Cancelled. Invalid Choice.")
            continue

        picked_chat = chats[int(choice) - 1]
        data = load_chat(picked_chat["path"])
        if data is None:
            print("The chat could not be loaded.")
            continue

        messages = [SYSTEM_MESSAGE] + data["messages"]
        chat_file_path = picked_chat["path"]
        print(f"Resumed: {picked_chat['title']} ({len(data['messages'])} messages)")
        continue


    # Add the user's message to the conversation before sending it to the model.
    messages.append(
        {
         "role": "user", 
         "content": user_input
        }
    )

    for step in range(MAX_RUN_STEPS):
        full_response, tool_calls = stream_chat(messages)

        # An empty tool-call list means the streamed text is the final answer.
        if not tool_calls:
            break

        log(f"Step {step + 1}: model requested {len(tool_calls)} tool call(s)")

        # Record the assistant's tool requests before adding their execution results.
        messages.append({
            "role": "assistant",
            "content": full_response,
            "tool_calls": build_tool_calls_payload(tool_calls)
        })

        # Run every requested tool and pass the result back into the conversation as a tool message.
        for tool_call in tool_calls:
            result = run_tool(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_id": tool_call.id,
                    "content": str(result),
                }
            )
    else:
        # This branch runs only if the model keeps requesting tools past the safe step limit.
        fallback = "I couldn't complete your request within the allowed steps. Please try again."
        messages.append({
            "role": "assistant",
            "content": fallback,
        })
        print("ASSISTANT: " + fallback)
        chat_file_path = save_chat(chat_file_path,messages[1:])
        continue

    messages.append(
        {
            "role":"assistant",
            "content":full_response
        }
    )
    chat_file_path = save_chat(chat_file_path,messages[1:])