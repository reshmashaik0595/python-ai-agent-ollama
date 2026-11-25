import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CHAT_FOLDER = os.getenv("CHAT_FOLDER")

def save_chat(path, messages):
    """Persist a conversation atomically, creating a new file when needed."""
    os.makedirs(CHAT_FOLDER, exist_ok=True)

    if path is None:
        fileName = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = os.path.join(CHAT_FOLDER, f"{fileName}.json")

    data = {
        "title": messages[0]["content"][:40],
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages
    }

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)

    return path

def load_chat(path):
    """Read and decode one saved JSON conversation."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def list_chats():
    """Return resumable chats with title, message count, and update time."""
    if not os.path.exists(CHAT_FOLDER):
        return []

    chats = []
    for chat_file_name in os.listdir(CHAT_FOLDER):
        if not chat_file_name.endswith(".json"):
            continue

        path = os.path.join(CHAT_FOLDER, chat_file_name)
        data = load_chat(path)
        if data is None:
            print("Skipped unreadable chat file: {chat_file_name}")
            continue

        chats.append(
            {
                "path":path,
                "title":data.get("title", "(untitled)"),
                "updated": data.get("updated", ""),
                "count": len(data.get("messages", [])),
            }
        )