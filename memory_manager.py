import json
import os


MEMORY_FILE = "user_memory.json"

def load_all():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_all(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_user_memory(user_id):
    data = load_all()

    user = data.setdefault(user_id, {
        "profile": {},
        "long_memories": [],
        "short_memories": []
    })

    return user

def add_memory(user_id, mem_type, content):
    data = load_all()

    user = data.setdefault(user_id, {
        "profile": {},
        "long_memories": [],
        "short_memories": []
    })

    if mem_type == "SHORT":
        #Only one
        user["short_memories"] = [content]
    elif mem_type == "LONG":
        if content not in user["long_memories"]:
            user["long_memories"].append(content)

    save_all(data)


def clear_short_memory_on_start():
    data = load_all()
    for user in data.values():
        user["short_memories"] = []
    save_all(data)
