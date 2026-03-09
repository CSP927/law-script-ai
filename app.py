import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

MODEL_NAME = "jhgan/ko-sroberta-multitask"

app = Flask(__name__)
CORS(app)

# -------------------------
# GEMINI API
# -------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------
# 데이터 로드
# -------------------------

with open("scripts.json", "r", encoding="utf-8") as f:
    scripts_data = json.load(f)

texts = [d["text"] for d in scripts_data]
categories = [d["category"] for d in scripts_data]
filenames = [d["filename"] for d in scripts_data]

print("대본 개수:", len(texts))

# -------------------------
# 임베딩 로드
# -------------------------

embeddings = np.load("embeddings.npy")

# cosine similarity 최적화
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

print("embeddings shape:", embeddings.shape)

# -------------------------
# 모델 로딩 (한 번만)
# -------------------------

print("임베딩 모델 로딩 중...")
model = SentenceTransformer(MODEL_NAME)
print("임베딩 모델 로딩 완료")

# -------------------------
# 스타일 로드
# -------------------------

def load_style(style):

    if not style:
        return ""

    path = f"style_samples/{style}.txt"

    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# -------------------------
# 유사 대본 검색 (초고속)
# -------------------------

def search_similar(query, category, top_k=5):

    query_vec = model.encode([query])[0]

    query_vec = query_vec / np.linalg.norm(query_vec)

    scores = embeddings @ query_vec

    filtered_idx = [
        i for i, cat in enumerate(categories) if cat == category
    ]

    filtered_scores = [(i, scores[i]) for i in filtered_idx]

    filtered_scores.sort(key=lambda x: x[1], reverse=True)

    top_idx = [i for i, _ in filtered_scores[:top_k]]

    results = []

    for idx in top_idx:

        results.append({
            "filename": filenames[idx],
            "text": texts[idx]
        })

    return results

# -------------------------
# 서버 상태 확인
# -------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# -------------------------
# 대본 생성
# -------------------------

@app.route("/generate", methods=["POST"])
def generate():

    try:

        data = request.json

        category = data.get("category")
        keywords = data.get("keywords")
        length = data.get("length")
        style = data.get("speaker")
        structure = data.get("script_structure")

        query = f"{category} {keywords}"

        refs = search_similar(query, category)

        reference_text = "\n\n".join(
            [r["text"][:2000] for r in refs]
        )

        style_text = load_style(style)

        prompt = f"""
당신은 전문 변호사 유튜브 대본 작가입니다.

[변호사 스타일]
{style_text}

[참고 대본]
{reference_text}

카테고리: {category}
키워드: {keywords}
영상 길이: {length}
대본 구조: {structure}

JSON 형식으로만 답하세요.

{{
"hook":["","",""],
"script_title":["","","","",""],
"full_script":"",
"shorts":[
{{"title":"","segment":""}},
{{"title":"","segment":""}},
{{"title":"","segment":""}}
],
"seo_keywords":["","","","","","","","","",""]
}}
"""

        response = gemini_model.generate_content(prompt)

        raw = response.text.strip()

        if raw.startswith("```json"):
            raw = raw.replace("```json","",1)

        if raw.endswith("```"):
            raw = raw[:-3]

        result = json.loads(raw)

        return jsonify(result)

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({"error":str(e)}),500

# -------------------------
# 미리보기
# -------------------------

@app.route("/preview", methods=["POST"])
def preview():

    data = request.json

    category = data.get("category")
    keywords = data.get("keywords")

    query = f"{category} {keywords}"

    items = search_similar(query, category)

    result = []

    for item in items:

        result.append({
            "filename": item["filename"],
            "preview": item["text"][:500]
        })

    return jsonify(result)

# -------------------------
# 서버 실행
# -------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT",8000))

    app.run(host="0.0.0.0",port=port)