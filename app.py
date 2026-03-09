import os
import json
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from google import genai

MODEL_NAME = "jhgan/ko-sroberta-multitask"

app = Flask(__name__)
CORS(app)

# -------------------------
# GEMINI API
# -------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
# 데이터 로드
# -------------------------
with open("scripts.json", "r", encoding="utf-8") as f:
    scripts_data = json.load(f)

texts = [d["text"] for d in scripts_data]
categories = [d["category"] for d in scripts_data]
filenames = [d["filename"] for d in scripts_data]

print("대본 개수:", len(texts))

embeddings = np.load("embeddings.npy")

print("embeddings shape:", embeddings.shape)

# -------------------------
# 임베딩 모델
# -------------------------
model = SentenceTransformer(MODEL_NAME)

# -------------------------
# 스타일 샘플 로드
# -------------------------
def load_style(style_name):

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
def search_similar(query, category, top_k=5):

    query_vec = model.encode([query])[0]

    scores = embeddings @ query_vec

    filtered = []

    for i, cat in enumerate(categories):
        if cat == category:
            filtered.append((i, float(scores[i])))

    if not filtered:
        return []

    filtered.sort(key=lambda x: x[1], reverse=True)

    top_idx = [idx for idx, _ in filtered[:top_k]]

    results = []

    for idx in top_idx:
        results.append({
            "filename": filenames[idx],
            "text": texts[idx]
        })

    return results


# -------------------------
# 서버 상태
# -------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -------------------------
# 대본 생성
# -------------------------
@app.route("/generate", methods=["POST"])
def generate_script():

    try:

        data = request.get_json(force=True)

        category = data.get("category", "").strip()
        keywords = data.get("keywords", "").strip()
        length = data.get("length", "").strip()
        speaker = data.get("speaker", "").strip()
        script_structure = data.get("script_structure", "").strip()

        if not category:
            return jsonify({"error": "category가 비어 있습니다."}), 400

        if not keywords:
            return jsonify({"error": "keywords가 비어 있습니다."}), 400

        query = f"{category} {keywords}"

        reference_items = search_similar(query, category, 5)

        if not reference_items:
            return jsonify({"error": "참고 대본을 찾지 못했습니다"}), 404

        reference_text = "\n\n".join(
            [f"[파일명] {item['filename']}\n{item['text'][:2000]}" for item in reference_items]
        )

        style_text = load_style(speaker)

        prompt = f"""
당신은 전문 변호사 유튜브 대본 작가입니다.

[변호사 스타일]
{style_text}

[참고 대본]
{reference_text}

아래 조건에 맞는 새로운 유튜브 대본을 작성하세요.

카테고리: {category}
주제 키워드: {keywords}
영상 길이: {length}
대본 구조: {script_structure}

조건:
1 실제 변호사 말투
2 일반인이 이해하기 쉽게
3 도입 후킹 포함
4 쇼츠 구간 포함
5 SEO 키워드 포함

반드시 아래 JSON 형식으로 답하세요.

{{
"hook":["후킹1","후킹2","후킹3"],
"script_title":["제목1","제목2","제목3","제목4","제목5"],
"full_script":"전체 대본",
"shorts":[
{{"title":"쇼츠 제목1","segment":"쇼츠 내용1"}},
{{"title":"쇼츠 제목2","segment":"쇼츠 내용2"}},
{{"title":"쇼츠 제목3","segment":"쇼츠 내용3"}}
],
"seo_keywords":["키워드1","키워드2","키워드3","키워드4","키워드5","키워드6","키워드7","키워드8","키워드9","키워드10"]
}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw = response.text.strip()

        if raw.startswith("```json"):
            raw = raw.replace("```json", "", 1).strip()

        if raw.endswith("```"):
            raw = raw[:-3].strip()

        result = json.loads(raw)

        return app.response_class(
            response=json.dumps(result, ensure_ascii=False),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:

        print("ERROR:", e)

        return jsonify({"error": str(e)}), 500


# -------------------------
# 관련 대본 미리보기
# -------------------------
@app.route("/preview", methods=["POST"])
def preview_scripts():

    data = request.get_json(force=True)

    category = data.get("category", "")
    keywords = data.get("keywords", "")

    query = f"{category} {keywords}"

    items = search_similar(query, category, 5)

    results = []

    for item in items:
        results.append({
            "filename": item["filename"],
            "preview": item["text"][:800]
        })

    return jsonify(results)


# -------------------------
# 서버 실행
# -------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    app.run(host="0.0.0.0", port=port)