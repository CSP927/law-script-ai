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
GITHUB_REPO = os.getenv("GITHUB_REPO")  
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")


def github_update_file(path: str, content_bytes: bytes, message: str):
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("GitHub 설정 없음 → commit 생략")
        return

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

    # 기존 파일 sha 조회
    r = requests.get(url, headers=headers)
    sha = None
    if r.status_code == 200:
        sha = r.json().get("sha")
    elif r.status_code not in [200, 404]:
        raise Exception(f"GitHub sha 조회 실패: {r.status_code} {r.text}")

    encoded = base64.b64encode(content_bytes).decode("utf-8")

    payload = {
        "message": message,
        "content": encoded,
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, json=payload)
    if r.status_code not in [200, 201]:
        raise Exception(f"GitHub 업로드 실패: {r.status_code} {r.text}")


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


def extract_json_object(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini 응답에서 JSON 객체를 찾지 못했습니다.")

    json_text = raw[start:end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print("===== JSON PARSE ERROR =====")
        print(json_text)
        print("============================")
        raise e


def ensure_list_of_strings(value: Any, target_len: int) -> List[str]:
    if isinstance(value, str):
        value = [value]

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
    if isinstance(value, dict):
        value = [value]

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


def normalize_meta_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hook": ensure_list_of_strings(result.get("hook"), 3),
        "script_title": ensure_list_of_strings(result.get("script_title"), 5),
        "short_titles": ensure_list_of_strings(result.get("short_titles"), 3),
        "seo_keywords": ensure_list_of_strings(result.get("seo_keywords"), 10),
    }


def safe_generate_content(prompt: str, retries: int = 3, sleep_sec: float = 1.2) -> str:
    last_error = None

    for _ in range(retries):
        try:
            response = gemini_model.generate_content(prompt)
            raw = (response.text or "").strip()
            raw = raw.replace("\r", " ").replace("\n", " ")
            return raw
        except Exception as e:
            last_error = e
            time.sleep(sleep_sec)

    raise last_error


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

        count_in_text = doc_text_l.count(token)
        if count_in_text > 0:
            score += min(count_in_text, 10) * 2.0

        if token in filename_l:
            score += 3.0

    if len(doc_text.strip()) < 300:
        score -= 1.5

    return score


def make_reference_excerpt(text: str, head_len: int = 500, tail_len: int = 300) -> str:
    text = (text or "").strip()

    if len(text) <= head_len + tail_len + 30:
        return text

    head = text[:head_len].strip()
    tail = text[-tail_len:].strip()

    return f"{head}\n...\n{tail}"


def search_similar(query: str, category: str, top_k: int = 3) -> List[Dict[str, str]]:
    query_tokens = unique_keep_order(tokenize(query))
    candidates = []

    for i, cat in enumerate(categories):
        if cat != category:
            continue

        text = texts[i] or ""
        filename = filenames[i] or "unknown"
        score = score_document(query_tokens, text, filename)
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
# 구조 가이드
# -------------------------
def structure_guide(structure: str) -> str:
    structure = (structure or "").strip()

    if structure == "설명형":
        return """
- 법 개념과 원칙을 먼저 설명합니다.
- 왜 중요한지 설명합니다.
- 실무상 주의점과 결론으로 마무리합니다.
"""

    if structure == "사례형":
        return """
- 실제로 있을 법한 사례나 상황으로 시작합니다.
- 그 사례가 왜 문제인지 설명합니다.
- 법적 판단과 실무적 대응을 설명합니다.
"""

    if structure == "판례형":
        return """
- 실제 판례 또는 판결 취지 소개로 시작합니다.
- 사건의 쟁점, 법원의 판단, 실무적 의미 순서로 설명합니다.
- 단순 설명형처럼 쓰지 말고 판례 해설 중심으로 작성합니다.
"""

    if structure == "Q&A형":
        return """
- 사람들이 자주 묻는 질문으로 시작합니다.
- 질문에 답하는 방식으로 설명합니다.
- 답변은 짧고 명확하게, 이어서 이유를 설명합니다.
"""

    return """
- 일반적인 유튜브 설명형 대본 구조로 작성합니다.
"""


# -------------------------
# 전체 대본 생성 프롬프트
# -------------------------
def build_full_script_prompt(
    category: str,
    keywords: str,
    length: str,
    style: str,
    structure: str,
    style_text: str,
    reference_text: str,
    draft_text: str,
    article_text: str
) -> str:
    expand_block = ""

    if not draft_text and not article_text:
        expand_block = """
[작성 방식]
- 참고자료 없이 작성하되, 반드시 참고 대본 스타일을 따르세요.
"""

    elif draft_text and not article_text:
        expand_block = f"""
[초안 기반 작성]
- 아래 초안을 반드시 반영하여 확장하세요.
- 초안의 핵심 취지와 논점을 유지할 것

[초안]
{draft_text}
"""

    elif article_text and not draft_text:
        expand_block = f"""
[기사 기반 작성]
- 아래 기사 내용을 바탕으로 대본을 작성하세요.
- 기사 내용은 참고자료일 뿐이며, 최종 대본 구조는 반드시 "{structure}"을 따르세요.
- 단순 요약이 아니라 선택한 구조에 맞게 재구성하세요.

[기사]
{article_text}
"""

    else:
        expand_block = f"""
[초안 + 기사 기반 작성]
- 초안을 중심으로 유지하고 기사 내용으로 보강하세요.
- 기사 내용을 그대로 복사하지 말고 재구성할 것
- 최종 대본 구조는 반드시 "{structure}"을 따르세요.

[초안]
{draft_text}

[기사]
{article_text}
"""

    return f"""
당신은 전문 변호사 유튜브 대본 작가입니다.

이번 작업의 최우선 목표는
1. 화자 이름과 자기소개를 절대 바꾸지 않는 것
2. 참고 대본의 말투와 설명 방식을 최대한 유지하는 것
3. 내용은 새로 작성하되, 법률적으로 어색하거나 잘못된 표현 없이 쓰는 것

[화자 규칙]
- 이번 대본의 화자는 반드시 "{style}" 입니다.
- 화자 이름을 다른 이름으로 바꾸면 안 됩니다.
- 자기소개가 들어갈 경우 반드시 "{style}" 이름을 사용해야 합니다.
- 기존 참고 대본의 말투, 문장 길이, 설명 방식, 호흡을 최대한 따르세요.

[참고 대본 스타일]
{style_text}

[관련 참고 대본]
{reference_text}

{expand_block}

[이번 작성 조건]
- 카테고리: {category}
- 핵심 키워드: {keywords}
- 영상 길이: {length}
- 대본 구조: {structure}

[대본 구조 규칙]
{structure_guide(structure)}

[대본 작성 규칙]
- 일반인이 이해하기 쉽게 써야 합니다.
- 변호사답게 정확하고 차분하게 설명해야 합니다.
- 유튜브 영상용 대본처럼 자연스럽게 써야 합니다.
- 참고 대본은 말투와 전개 방식만 참고하고, 내용은 새로 작성해야 합니다.
- 현행 법률과 일반적인 실무 기준에 맞는 방향으로 작성하세요.
- 인삿말 이후 곧바로 핵심 설명으로 들어가세요.
- 전체 대본만 출력하세요.
- JSON으로 출력하지 말고, 전체 대본 본문만 출력하세요.
""".strip()


# -------------------------
# 문장 분리 / 쇼츠 구간 추출
# -------------------------
def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\r", " ", text or "")
    text = re.sub(r"\n+", " ", text)
    parts = re.split(r'(?<=[\.\!\?다요])\s+', text)
    result = [p.strip() for p in parts if p.strip()]
    return result


def is_greeting_sentence(sentence: str) -> bool:
    s = sentence.strip()
    greeting_patterns = [
        r"^안녕하세요",
        r"^오늘은",
        r"^이번에는",
        r"변호사입니다",
        r"채널.*입니다",
    ]
    return any(re.search(p, s) for p in greeting_patterns)


def build_segment_from_sentences(
    sentences: List[str],
    start_idx: int,
    min_chars: int = 180,
    max_chars: int = 420
) -> str:
    chunk = []
    total = 0

    for i in range(start_idx, len(sentences)):
        s = sentences[i].strip()
        if not s:
            continue

        if not chunk and is_greeting_sentence(s):
            continue

        if total + len(s) > max_chars and total >= min_chars:
            break

        chunk.append(s)
        total += len(s)

        if total >= min_chars:
            continue

    return " ".join(chunk).strip()


def pick_short_segments(
    full_script: str,
    count: int = 3,
    min_chars: int = 180,
    max_chars: int = 420
) -> List[str]:
    sentences = split_sentences(full_script)

    if not sentences:
        return ["", "", ""]

    start_points = [
        max(0, len(sentences) // 6),
        max(0, len(sentences) // 2 - 1),
        max(0, (len(sentences) * 5) // 6 - 1),
    ]

    segments = []
    for sp in start_points:
        seg = build_segment_from_sentences(
            sentences,
            sp,
            min_chars=min_chars,
            max_chars=max_chars
        )
        if seg:
            segments.append(seg)

    idx = 0
    while len(segments) < count and idx < len(sentences):
        seg = build_segment_from_sentences(
            sentences,
            idx,
            min_chars=min_chars,
            max_chars=max_chars
        )
        if seg and seg not in segments:
            segments.append(seg)
        idx += 1

    while len(segments) < count:
        segments.append("")

    return segments[:count]


# -------------------------
# 메타데이터 생성 프롬프트
# -------------------------
def build_meta_prompt(full_script: str, shorts_segments: List[str]) -> str:
    seg1 = shorts_segments[0] if len(shorts_segments) > 0 else ""
    seg2 = shorts_segments[1] if len(shorts_segments) > 1 else ""
    seg3 = shorts_segments[2] if len(shorts_segments) > 2 else ""

    return f"""
아래 full_script를 바탕으로 메타데이터만 생성하세요.

중요 규칙:
- hook은 full_script 안에서 실제 문장을 그대로 발췌해야 합니다.
- hook은 새로 쓰지 말고, full_script에 있는 문장을 그대로 복사하세요.
- hook에는 인삿말(안녕하세요, 오늘은 등)이 들어가면 안 됩니다.
- script_title은 제목 추천 5개를 작성하세요.
- short_titles는 아래 3개 쇼츠 구간 각각에 맞는 제목 1개씩만 작성하세요.
- seo_keywords는 10개 작성하세요.
- JSON만 출력하세요.
- 빈 문자열을 넣지 마세요.

[full_script]
{full_script}

[쇼츠 구간 1]
{seg1}

[쇼츠 구간 2]
{seg2}

[쇼츠 구간 3]
{seg3}

반드시 아래 JSON 형식으로만 출력하세요.

{{
  "hook": [
    "full_script에서 그대로 발췌한 문장1",
    "full_script에서 그대로 발췌한 문장2",
    "full_script에서 그대로 발췌한 문장3"
  ],
  "script_title": [
    "제목1",
    "제목2",
    "제목3",
    "제목4",
    "제목5"
  ],
  "short_titles": [
    "쇼츠 제목1",
    "쇼츠 제목2",
    "쇼츠 제목3"
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
# 대본 생성 / 초안 확장 포함
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
        draft_text = (data.get("draft_text") or "").strip()
        article_text = (data.get("article_text") or "").strip()

        if not category:
            return jsonify({"error": "category가 비어 있습니다."}), 400

        if not keywords:
            return jsonify({"error": "keywords가 비어 있습니다."}), 400

        query = f"{category} {keywords}"
        refs = search_similar(query, category, top_k=3)

        if not refs:
            return jsonify({"error": "유사 대본을 찾지 못했습니다."}), 404

        reference_text = "\n\n".join(
            [f"[파일명] {r['filename']}\n{make_reference_excerpt(r['text'])}" for r in refs]
        )

        style_text = load_style(style)

        full_script_prompt = build_full_script_prompt(
    category=category,
    keywords=keywords,
    length=length,
    style=style,
    structure=structure,
    style_text=style_text,
    reference_text=reference_text,
    draft_text=draft_text,
    article_text=article_text
)

        full_script = safe_generate_content(full_script_prompt).strip()

        shorts_segments = pick_short_segments(
            full_script=full_script,
            count=3,
            min_chars=180,
            max_chars=420
        )

        meta_prompt = build_meta_prompt(full_script, shorts_segments)
        meta_raw = safe_generate_content(meta_prompt)
        meta_parsed = extract_json_object(meta_raw)
        meta = normalize_meta_result(meta_parsed)

        result = {
            "hook": meta["hook"],
            "script_title": meta["script_title"],
            "full_script": full_script,
            "shorts": [
                {
                    "title": meta["short_titles"][0],
                    "segment": shorts_segments[0]
                },
                {
                    "title": meta["short_titles"][1],
                    "segment": shorts_segments[1]
                },
                {
                    "title": meta["short_titles"][2],
                    "segment": shorts_segments[2]
                }
            ],
            "seo_keywords": meta["seo_keywords"]
        }

        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------
# 생성 대본 저장
# scripts.json 저장 + embeddings 재생성 + GitHub 업로드
# -------------------------
@app.route("/save-script", methods=["POST"])
def save_script():
    global scripts_data, texts, categories, filenames

    try:
        data = request.get_json(force=True)

        category = normalize_space(data.get("category") or "")
        text = (data.get("text") or "").strip()
        speaker = normalize_space(data.get("speaker") or "")
        title = normalize_space(data.get("title") or data.get("filename") or "")

        if not category:
            return jsonify({"error": "category가 비어 있습니다."}), 400

        if not text:
            return jsonify({"error": "text가 비어 있습니다."}), 400

        filename = title if title else f"generated_{len(scripts_data) + 1}"

        item = {
            "filename": filename,
            "category": category,
            "text": text,
            "speaker": speaker
        }

        scripts_data.append(item)
        texts.append(text)
        categories.append(category)
        filenames.append(filename)

        with open("scripts.json", "w", encoding="utf-8") as f:
            json.dump(scripts_data, f, ensure_ascii=False, indent=2)

        print("scripts.json 저장 완료")

        build_ok = True
        build_error = ""

        try:
            subprocess.run(
                ["python3", "build_index.py"],
                check=True
            )
            print("embedding rebuild 완료")
        except Exception as e:
            build_ok = False
            build_error = str(e)
            print("embedding rebuild 실패:", build_error)

        github_ok = True
        github_error = ""

        try:
            if os.path.exists("scripts.json"):
                with open("scripts.json", "rb") as f:
                    github_update_file(
                        "scripts.json",
                        f.read(),
                        f"add script {filename}"
                    )

            if os.path.exists("embeddings.npy"):
                with open("embeddings.npy", "rb") as f:
                    github_update_file(
                        "embeddings.npy",
                        f.read(),
                        "update embeddings"
                    )

            print("GitHub commit 완료")
        except Exception as e:
            github_ok = False
            github_error = str(e)
            print("GitHub commit 실패:", github_error)

        return jsonify({
            "status": "saved",
            "message": "대본이 저장되었습니다.",
            "build_index": "success" if build_ok else "failed",
            "build_error": build_error,
            "github_upload": "success" if github_ok else "failed",
            "github_error": github_error
        })

    except Exception as e:
        print("SAVE ERROR:", str(e))
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