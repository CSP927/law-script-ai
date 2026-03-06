import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from google import genai

MODEL_NAME = "jhgan/ko-sroberta-multitask"

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

client = genai.Client(api_key=GEMINI_API_KEY)

with open("scripts.json", "r", encoding="utf-8") as f:
    scripts_data = json.load(f)

texts = [d["text"] for d in scripts_data]
categories = [d["category"] for d in scripts_data]
filenames = [d["filename"] for d in scripts_data]

print("대본 개수:", len(texts))

embeddings = np.load("embeddings.npy")
print("embeddings shape:", embeddings.shape)

model = SentenceTransformer(MODEL_NAME)


def search_similar(query: str, category: str, top_k: int = 5):
    query_vec = model.encode([query])[0]

    if embeddings.shape[1] != len(query_vec):
        raise ValueError(
            f"임베딩 차원 불일치: embeddings={embeddings.shape[1]}, query={len(query_vec)}"
        )

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


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate_script():
    try:
        data = request.get_json(force=True)

        category = data.get("category", "").strip()
        keywords = data.get("keywords", "").strip()
        length = data.get("length", "").strip()
        speaker = data.get("speaker", "").strip()

        if not category:
            return jsonify({"error": "category가 비어 있습니다."}), 400
        if not keywords:
            return jsonify({"error": "keywords가 비어 있습니다."}), 400

        query = f"{category} {keywords}"
        reference_items = search_similar(query, category, top_k=5)

        if not reference_items:
            return jsonify({"error": f"'{category}' 카테고리에서 참고 대본을 찾지 못했습니다."}), 404

        reference_text = "\n\n".join(
            [f"[파일명] {item['filename']}\n{item['text'][:2000]}" for item in reference_items]
        )

        prompt = f"""
당신은 전문 변호사 유튜브 대본 작가입니다.

다음 기존 대본 스타일을 참고하여 새로운 유튜브 대본을 작성하세요.

[참고 대본]
{reference_text}

[요청 조건]
주제 카테고리: {category}
키워드: {keywords}
영상 길이: {length}
화자 스타일: {speaker}

조건:
1. 실제 변호사가 설명하는 말투
2. 법률적으로 틀리지 않게 작성
3. 일반인이 이해하기 쉽게 설명
4. 도입 후킹 포함
5. 전체 유튜브 대본 형식
6. 쇼츠로 만들 수 있는 구간 포함

반드시 아래 JSON 형식으로만 답하세요.
설명문, 서두, 마무리 인사 없이 JSON만 출력하세요.

{{
  "hook": ["후킹1", "후킹2", "후킹3"],
  "script_title": ["제목1", "제목2", "제목3", "제목4", "제목5"],
  "full_script": "전체 유튜브 대본",
  "shorts": [
    {{
      "title": "쇼츠 제목 1",
      "segment": "쇼츠 구간 내용 1"
    }},
    {{
      "title": "쇼츠 제목 2",
      "segment": "쇼츠 구간 내용 2"
    }},
    {{
      "title": "쇼츠 제목 3",
      "segment": "쇼츠 구간 내용 3"
    }}
  ],
  "seo_keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5", "키워드6", "키워드7", "키워드8", "키워드9", "키워드10"]
}}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw_text = response.text.strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "", 1).strip()
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

        result_json = json.loads(raw_text)

        return app.response_class(
            response=json.dumps(result_json, ensure_ascii=False),
            status=200,
            mimetype="application/json"
        )

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)