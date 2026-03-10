import os
import json
import numpy as np
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
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

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
# 임베딩 로드
# -------------------------
embeddings = np.load("embeddings.npy").astype(np.float32)

print("embeddings shape:", embeddings.shape)

# -------------------------
# 간단한 query vector 생성
# -------------------------
def embed_query_simple(text: str):

    vec = np.zeros(embeddings.shape[1], dtype=np.float32)

    for c in text:
        vec[ord(c) % embeddings.shape[1]] += 1

    norm = np.linalg.norm(vec)

    if norm > 0:
        vec = vec / norm

    return vec


# -------------------------
# 스타일 로드
# -------------------------
def load_style(style_name: str):

    if not style_name:
        return ""

    path = f"style_samples/{style_name}.txt"

    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# -------------------------
# 유사 대본 검색
# -------------------------
def search_similar(query: str, category: str, top_k: int = 5):

    query_vec = embed_query_simple(query)

    scores = embeddings @ query_vec

    filtered_idx = [
        i for i, cat in enumerate(categories)
        if cat == category
    ]

    filtered_scores = [(i, float(scores[i])) for i in filtered_idx]

    filtered_scores.sort(key=lambda x: x[1], reverse=True)

    top_idx = [i for i, _ in filtered_scores[:top_k]]

    return [
        {
            "filename": filenames[idx],
            "text": texts[idx]
        }
        for idx in top_idx
    ]


# -------------------------
# 서버 상태 확인
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -------------------------
# 대본 생성
# -------------------------
@app.route("/generate", methods=["POST"])
def generate():

    try:

        data = request.get_json(force=True)

        category = (data.get("category") or "").strip()
        keywords = (data.get("keywords") or "").strip()
        length = (data.get("length") or "").strip()
        style = (data.get("speaker") or "").strip()
        structure = (data.get("script_structure") or "").strip()

        query = f"{category} {keywords}"

        refs = search_similar(query, category, top_k=5)

        reference_text = "\n\n".join([r["text"][:2000] for r in refs])

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

반드시 JSON 형식으로만 답하세요.

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

        raw = (response.text or "").strip()

        raw = raw.replace("```json", "").replace("```", "").strip()

        result = json.loads(raw)

        return jsonify(result)

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({"error": str(e)}), 500


# -------------------------
# 미리보기
# -------------------------
@app.route("/preview", methods=["POST"])
def preview():

    data = request.get_json(force=True)

    category = (data.get("category") or "").strip()
    keywords = (data.get("keywords") or "").strip()

    query = f"{category} {keywords}"

    items = search_similar(query, category, top_k=5)

    result = [
        {
            "filename": item["filename"],
            "preview": item["text"][:500]
        }
        for item in items
    ]

    return jsonify(result)


# -------------------------
# 서버 실행
# -------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    app.run(host="0.0.0.0", port=port)