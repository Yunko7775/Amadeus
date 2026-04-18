import threading

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, re
from google import genai
from google.genai import types
from config import API_KEY, ROLE_PROMPT,MEMORY_JUDGE_PROMPT
from memory_manager import get_user_memory, add_memory, clear_short_memory_on_start
from datetime import datetime
import json

now = datetime.now().strftime("%Y-%m-%d")

# Short Me
system_short_memories = [
    f"【系统时间】今天的日期是 {now}"
]

app = Flask(__name__)
CORS(app)

#  Gemini Client
client = genai.Client(api_key=API_KEY)


EXPR_MAP = {
    "happy":      ["happy", "幸せ", "ハッピー", "満足しています"],
    "angry":      ["angry", "怒り"],
    "neutral":    ["neutral", "ニュートラル", "ニュートン", "中立"],
    "surprised":  ["surprised", "びっくり", "驚いた"],
    "sad":        ["sad", "悲しい"],
    "disgust":    ["disgust", "嫌悪感"],
}

def normalize_expression(expr, expr_map):
    for key, variants in expr_map.items():
        if expr in variants:
            return key
    return "neutral"

def remove_expression_keywords(text):
    return re.sub(r"[（(]\s*([^（）()]+?)\s*[)）]", "", text).strip()

def translate_to_japanese(text):
    import requests
    try:
        params = {'q': text, 'langpair': 'zh|ja', 'de': 'a@b.c'}
        r = requests.get('https://api.mymemory.translated.net/get', params=params, timeout=5)
        if r.ok:
            return r.json().get('responseData', {}).get('translatedText', text)
        return text
    except:
        return text



def safe_parse_llm_json(text: str):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # 提取 JSON 对象
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    json_text = match.group(0).replace("\n", "").strip()  # 去掉换行
    try:
        return json.loads(json_text)
    except Exception as e:
        print("❌ JSON parse failed:", e)
        print("❌ JSON text was:", repr(json_text))
        return None

def auto_extract_memory(user_message: str, user_id: str):
    print("=== AUTO MEMORY DEBUG START ===")
    print("User message:", user_message)

    try:
        #  LLM
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=[
                types.Content(role="model", parts=[types.Part(text=MEMORY_JUDGE_PROMPT)]),
                types.Content(role="user", parts=[types.Part(text=user_message)])
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.7
            )
        )

        raw_text = getattr(response, "text", None)
        if not raw_text and hasattr(response, "contents") and response.contents:
            for c in response.contents:
                if hasattr(c, "text") and c.text:
                    raw_text = c.text
                    break
                elif hasattr(c, "parts"):
                    for p in c.parts:
                        if hasattr(p, "text") and p.text:
                            raw_text = p.text
                            break

        if not raw_text:
            print("❌ LLM returned no text")
            print("=== AUTO MEMORY DEBUG END ===")
            return

        print("LLM raw_text:", repr(raw_text))

    except Exception as e:
        print("❌ Memory judge LLM failed:", e)
        print("=== AUTO MEMORY DEBUG END ===")
        return

    data = safe_parse_llm_json(raw_text)
    print("Parsed JSON:", data)

    if not data:
        print("❌ No valid JSON from LLM")
        print("=== AUTO MEMORY DEBUG END ===")
        return

    should_store = data.get("should_store") or data.get("Should_Store") or data.get("shouldStore")
    if not should_store or str(should_store).lower() != "true":
        print("LLM judge says not to store memory")
        print("=== AUTO MEMORY DEBUG END ===")
        return

    mem_type = data.get("memory_type") or data.get("Memory_Type") or data.get("memoryType")
    content = data.get("content") or data.get("Content")

    if mem_type not in ("SHORT", "LONG") or not content:
        print("❌ Invalid memory_type or empty content")
        print("=== AUTO MEMORY DEBUG END ===")
        return

    try:
        add_memory(user_id, mem_type, content)
        print(f"✅ MEMORY STORED [{mem_type}] {content}")
    except Exception as e:
        print("❌ add_memory failed:", e)

    print("=== AUTO MEMORY DEBUG END ===")





# ------------------------
# Text-Reply
# ------------------------
def build_full_prompt(user_id: str, user_message: str):
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d")

    user_memory = get_user_memory(user_id)

    system_short_memories = [
        f"【系统时间】今天的日期是 {now}"
    ]

    short_block = "\n".join(
        f"- {m}" for m in system_short_memories + user_memory["short_memories"]
    )

    long_block = "\n".join(
        f"- {m}" for m in user_memory["long_memories"]
    )

    return f"""
        【最高优先级：人格与行为规则】
        你必须【严格、无条件】遵守以下设定，任何记忆或对话都不能违反。
        
        {ROLE_PROMPT}
        
        【记忆使用规则（极其重要）】
        - 长期记忆仅用于保持人格一致性与推理
        - 除非用户明确询问或当前话题强相关
        - 否则禁止主动提及长期记忆中的具体事实
        - 禁止突然自我暴露个人信息
        
        【系统状态（短期 / 会话态）】
        {short_block if short_block else "（无）"}
        
        【关于该用户的长期记忆（真实）】
        {long_block if long_block else "（无）"}
        
        【当前对话】
        用户：{user_message}
        """

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json or {}
    user_id = data.get("user_id", "default_user")
    user_message = data.get("message", "").strip()

    # 1️ Reply
    full_prompt = build_full_prompt(user_id, user_message)
    try:
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=[types.Content(role="user", parts=[types.Part(text=full_prompt)])],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.7
            )
        )
        reply_text = getattr(response, "text", "") or "模型未返回结果"
    except Exception as e:
        reply_text = f"模型调用失败: {e}"

    # 2️ Juggle_Memory
    threading.Thread(target=auto_extract_memory, args=(user_message, user_id), daemon=True).start()

    # 3️ expression and jp_reply
    expr_match = re.findall(r"[（(]\s*([^（）()]+?)\s*[)）]", reply_text)
    expression = normalize_expression(expr_match[0], EXPR_MAP) if expr_match else "neutral"
    display_text = remove_expression_keywords(reply_text)
    japanese_reply = translate_to_japanese(display_text)

    return jsonify({
        "reply": display_text,
        "japanese_reply": japanese_reply,
        "expression": expression
    })

# ------------------------
# Picture question
# ------------------------
@app.route('/ask_with_image', methods=['POST'])
def ask_with_image():
    try:
        user_id = request.form.get("user_id", "default_user")
        question = request.form.get("question", "").strip() or "请描述这张图片"

        image_file = request.files.get("image")
        if not image_file:
            return jsonify({
                'reply': "未上传图片",
                'japanese_reply': "",
                'expression': "neutral"
            })

        full_prompt = build_full_prompt(user_id, question)

        suffix = os.path.splitext(image_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            image_file.save(tmp_path)

        uploaded_file = client.files.upload(file=tmp_path)

        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=[
                uploaded_file,
                types.Content(
                    role="user",
                    parts=[types.Part(text=full_prompt)]
                )
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.7
            )
        )

        reply_text = response.text or "模型未返回结果"

        expr_match = re.findall(r"[（(]\s*([^（）()]+?)\s*[)）]", reply_text)
        expression = normalize_expression(expr_match[0], EXPR_MAP) if expr_match else "neutral"

        display_text = remove_expression_keywords(reply_text)
        japanese_reply = translate_to_japanese(display_text)

        os.remove(tmp_path)

        return jsonify({
            'reply': display_text,
            'japanese_reply': japanese_reply,
            'expression': expression
        })

    except Exception as e:
        print("Flask /ask_with_image 错误:", e)
        return jsonify({
            'reply': f"发生错误: {e}",
            'japanese_reply': "",
            'expression': "neutral"
        })


# ------------------------
if __name__ == '__main__':
    clear_short_memory_on_start()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False
    )