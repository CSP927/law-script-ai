import os
import re
import json
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
# 데이터 로드
# -------------------------
with open("scripts.json", "r", encoding="utf-8") as f:
    scripts_data = json.load(f)

texts = [d.get("text", "") for d in scripts_data]
categories = [d.get("category", "") for d in scripts_data]
filenames = [d.get("filename", "unknown") for d in scripts_data]

print("대본 개수:", len(texts))


# -------------------------
# 공통 유틸
# -------------------------
def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def tokenize(text: str) -> List[str]:
    """
    너무 무겁지 않게 한글/영문/숫자 토큰만 추출
    """
    text = (text or "").lower()
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)
    return [t for t in tokens if len(t) >= 1]


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = item.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def extract_json_object(raw: str) -> Dict[str, Any]:
    """
    Gemini가 설명을 섞거나 코드블록을 섞어도
    첫 { 부터 마지막 } 까지 잘라서 JSON 파싱
    """
    raw = (raw or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini 응답에서 JSON 객체를 찾지 못했습니다.")

    json_text = raw[start:end + 1]
    return json.loads(json_text)


def ensure_list_of_strings(value: Any, target_len: int) -> List[str]:
    if not isinstance(value, list):
        value = []

    cleaned = []
    for item in value:
        if item is None:
            continue
        cleaned.append(str(item).strip())

    cleaned = unique_keep_order(cleaned)

    while len(cleaned) < target_len:
        cleaned.append("")

    return cleaned[:target_len]


def ensure_shorts(value: Any, target_len: int = 3) -> List[Dict[str, str]]:
    if not isinstance(value, list):
        value = []

    result = []
    for item in value:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "") or "").strip()
        segment = str(item.get("segment", "") or "").strip()

        result.append({
            "title": title,
            "segment": segment
        })

    while len(result) < target_len:
        result.append({"title": "", "segment": ""})

    return result[:target_len]


def normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wix에서 항상 같은 구조로 받을 수 있게 강제 보정
    """
    normalized = {
        "hook": ensure_list_of_strings(result.get("hook"), 3),
        "script_title": ensure_list_of_strings(result.get("script_title"), 5),
        "full_script": str(result.get("full_script", "") or "").strip(),
        "shorts": ensure_shorts(result.get("shorts"), 3),
        "seo_keywords": ensure_list_of_strings(result.get("seo_keywords"), 10),
    }

    return normalized


# -------------------------
# 스타일 로드
# -------------------------
def load_style(style_name: str) -> str:
    if not style_name:
        return ""

    path = f"style_samples/{style_name}.txt"

    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# -------------------------
# 유사 대본 검색 (키워드 기반)
# -------------------------
def score_document(query_tokens: List[str], doc_text: str, filename: str) -> float:
    doc_text_l = (doc_text or "").lower()
    filename_l = (filename or "").lower()

    score = 0.0

    for token in query_tokens:
        if not token:
            continue

        # 텍스트 본문에 있으면 점수
        count_in_text = doc_text_l.count(token)
        if count_in_text > 0:
            score += min(count_in_text, 10) * 2.0

        # 파일명에 있으면 가중치
        if token in filename_l:
            score += 3.0

    # 너무 짧은 문서는 감점
    if len(doc_text.strip()) < 300:
        score -= 1.5

    return score


def search_similar(query: str, category: str, top_k: int = 5) -> List[Dict[str, str]]:
    query_tokens = unique_keep_order(tokenize(query))

    candidates = []

    for i, cat in enumerate(categories):
        if cat != category:
            continue

        text = texts[i] or ""
        filename = filenames[i] or "unknown"

        score = score_document(query_tokens, text, filename)

        # 아주 최소 점수라도 없으면 뒤로 밀리게 처리
        candidates.append((i, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    top_idx = [i for i, _ in candidates[:top_k]]

    return [
        {
            "filename": filenames[idx],
            "text": texts[idx]
        }
        for idx in top_idx
    ]


# -------------------------
# 프롬프트 생성
# -------------------------
def build_prompt(
    category: str,
    keywords: str,
    length: str,
    style: str,
    structure: str,
    style_text: str,
    reference_text: str
) -> str:
    return f"""
당신은 전문 변호사 유튜브 대본 작가입니다.

이번 작업의 최우선 목표는
1. 화자 이름과 자기소개를 절대 바꾸지 않는 것
2. 참고 대본의 말투와 설명 방식을 최대한 유지하는 것
3. 반드시 JSON만 출력하는 것입니다.

[화자 규칙]
- 이번 대본의 화자는 반드시 "{style}" 입니다.
- 화자 이름을 다른 이름으로 바꾸면 안 됩니다.
- 자기소개가 들어갈 경우 반드시 "{style}" 이름을 사용해야 합니다.
- 기존 참고 대본의 말투, 문장 길이, 설명 방식, 호흡을 최대한 따르세요.

[참고 대본 스타일]
아래 텍스트는 실제 화자의 말투와 설명 방식입니다.
설명 순서, 사례 풀어가는 방식, 문장 톤, 전문직다운 어조를 반드시 참고하세요.
완전히 다른 스타일로 새로 쓰지 말고, 아래 스타일을 최대한 유지해서 새 주제로 확장하세요.

{style_text}

[관련 참고 대본]
{reference_text}

[이번 작성 조건]
- 카테고리: {category}
- 핵심 키워드: {keywords}
- 영상 길이: {length}
- 대본 구조: {structure}

[대본 작성 규칙]
- 일반인이 이해하기 쉽게 써야 합니다.
- 변호사답게 정확하고 차분하게 설명해야 합니다.
- 유튜브 영상용 대본처럼 자연스럽게 써야 합니다.
- 후킹 멘트는 반드시 3개 작성합니다.
- 제목 추천은 반드시 5개 작성합니다.
- 쇼츠 제목/쇼츠 내용은 반드시 3개씩 작성합니다.
- SEO 키워드는 반드시 10개 작성합니다.
- JSON 외의 어떤 설명도 쓰면 안 됩니다.
- 마크다운 코드블록도 쓰면 안 됩니다.
- 모든 필드는 반드시 채워야 합니다.
- shorts의 segment는 절대 비워두면 안 됩니다.
- 내용이 비어 있으면 안 됩니다.

반드시 아래 형식 그대로 JSON만 출력하세요.

{{
  "hook": [
    "후킹1",
    "후킹2",
    "후킹3"
  ],
  "script_title": [
    "제목1",
    "제목2",
    "제목3",
    "제목4",
    "제목5"
  ],
  "full_script": "전체 대본",
  "shorts": [
    {{
      "title": "쇼츠 제목1",
      "segment": "쇼츠 내용1"
    }},
    {{
      "title": "쇼츠 제목2",
      "segment": "쇼츠 내용2"
    }},
    {{
      "title": "쇼츠 제목3",
      "segment": "쇼츠 내용3"
    }}
  ],
  "seo_keywords": [
    "키워드1",
    "키워드2",
    "키워드3",
    "키워드4",
    "키워드5",
    "키워드6",
    "키워드7",
    "키워드8",
    "키워드9",
    "키워드10"
  ]
}}
""".strip()


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

        category = normalize_space(data.get("category") or "")
        keywords = normalize_space(data.get("keywords") or "")
        length = normalize_space(data.get("length") or "")
        style = normalize_space(data.get("speaker") or "")
        structure = normalize_space(data.get("script_structure") or "")

        if not category:
            return jsonify({"error": "category가 비어 있습니다."}), 400

        if not keywords:
            return jsonify({"error": "keywords가 비어 있습니다."}), 400

        query = f"{category} {keywords}"
        refs = search_similar(query, category, top_k=5)

        if not refs:
            return jsonify({"error": "유사 대본을 찾지 못했습니다."}), 404

        reference_text = "\n\n".join(
            [f"[파일명] {r['filename']}\n{(r['text'] or '')[:2000]}" for r in refs]
        )

        style_text = load_style(style)

        prompt = build_prompt(
            category=category,
            keywords=keywords,
            length=length,
            style=style,
            structure=structure,
            style_text=style_text,
            reference_text=reference_text
        )

        response = gemini_model.generate_content(prompt)
        raw = (response.text or "").strip()

        parsed = extract_json_object(raw)
        result = normalize_result(parsed)

        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------
# 미리보기
# -------------------------
@app.route("/preview", methods=["POST"])
def preview():
    try:
        data = request.get_json(force=True)

        category = normalize_space(data.get("category") or "")
        keywords = normalize_space(data.get("keywords") or "")

        if not category:
            return jsonify({"error": "category가 비어 있습니다."}), 400

        if not keywords:
            return jsonify({"error": "keywords가 비어 있습니다."}), 400

        query = f"{category} {keywords}"
        items = search_similar(query, category, top_k=5)

        result = [
            {
                "filename": item["filename"],
                "preview": (item["text"] or "")[:500]
            }
            for item in items
        ]

        return jsonify(result)

    except Exception as e:
        print("PREVIEW ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------
# 서버 실행
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)