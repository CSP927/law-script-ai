import os
import re
import json
import time
import base64
import subprocess
import requests
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# -------------------------
# GEMINI API
# -------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------
# GitHub 설정
# -------------------------

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")   # username/repo
GITHUB_BRANCH = "main"


def github_update_file(path, content_bytes, message):

    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("GitHub 설정 없음 → commit 생략")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}"
    }

    r = requests.get(url, headers=headers)

    sha = None
    if r.status_code == 200:
        sha = r.json()["sha"]

    encoded = base64.b64encode(content_bytes).decode()

    payload = {
        "message": message,
        "content": encoded,
        "branch": GITHUB_BRANCH
    }

    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, json=payload)

    if r.status_code not in [200, 201]:
        raise Exception(r.text)


# -------------------------
# 데이터 로드
# -------------------------

with open("scripts.json", "r", encoding="utf-8") as f:
    scripts_data = json.load(f)

texts = [d.get("text", "") for d in scripts_data]
categories = [d.get("category", "") for d in scripts_data]
filenames = [d.get("filename", "unknown") for d in scripts_data]

print("대본 개수:", len(texts))

# -------------------------
# 이하 기존 코드 전부 동일
# -------------------------

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)
    return [t for t in tokens if len(t) >= 1]


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = str(item).strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


# -------------------------
# SAVE SCRIPT (업그레이드)
# -------------------------

@app.route("/save-script", methods=["POST"])
def save_script():

    global scripts_data, texts, categories, filenames

    try:

        data = request.get_json(force=True)

        category = normalize_space(data.get("category") or "")
        text = (data.get("text") or "").strip()
        speaker = normalize_space(data.get("speaker") or "")
        title = normalize_space(data.get("filename") or "")

        if not category:
            return jsonify({"error": "category가 비어 있습니다."}), 400

        if not text:
            return jsonify({"error": "text가 비어 있습니다."}), 400

        filename = title if title else f"generated_{len(scripts_data)+1}"

        item = {
            "filename": filename,
            "category": category,
            "text": text,
            "speaker": speaker
        }

        # -------------------------
        # scripts.json append
        # -------------------------

        scripts_data.append(item)
        texts.append(text)
        categories.append(category)
        filenames.append(filename)

        with open("scripts.json", "w", encoding="utf-8") as f:
            json.dump(scripts_data, f, ensure_ascii=False, indent=2)

        print("scripts.json 저장 완료")

        # -------------------------
        # embeddings rebuild
        # -------------------------

        try:

            subprocess.run(
                ["python3", "build_index.py"],
                check=True
            )

            print("embedding rebuild 완료")

        except Exception as e:

            print("embedding rebuild 실패:", str(e))

        # -------------------------
        # GitHub commit
        # -------------------------

        try:

            with open("scripts.json", "rb") as f:

                github_update_file(
                    "scripts.json",
                    f.read(),
                    f"add script {filename}"
                )

            with open("embeddings.npy", "rb") as f:

                github_update_file(
                    "embeddings.npy",
                    f.read(),
                    "update embeddings"
                )

            print("GitHub commit 완료")

        except Exception as e:

            print("GitHub commit 실패:", str(e))

        return jsonify({
            "status": "saved",
            "message": "scripts.json 저장 + embedding rebuild + GitHub commit 완료"
        })

    except Exception as e:

        print("SAVE ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------
# 서버 실행
# -------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)