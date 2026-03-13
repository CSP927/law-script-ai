import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

print("모델 로딩 중...")
model = SentenceTransformer(MODEL_NAME)

print("scripts.json 로딩 중...")
with open("scripts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []

for item in data:
    text = item.get("text", "").strip()
    if text:
        texts.append(text[:1000])

print("임베딩 생성할 문서 수:", len(texts))
print("임베딩 생성 시작...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.clip(norms, 1e-12, None)

embeddings = embeddings.astype(np.float32)

print("embeddings shape:", embeddings.shape)
print("embeddings size (MB):", embeddings.nbytes / 1024 / 1024)

np.save("embeddings.npy", embeddings)

print("embeddings.npy 저장 완료")