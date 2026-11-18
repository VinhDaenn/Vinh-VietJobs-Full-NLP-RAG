"""NLP skill extractor + semantic retrieval + optional local-LLM RAG.

Self-contained inference module for use by Supervisor / FastAPI.
All model downloads happen only when the corresponding function is first used.
"""
import os
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np

DEFAULT_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=2)
def get_embedder(model_name=DEFAULT_EMBEDDING):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_llm(model_name=DEFAULT_LLM):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return tokenizer, model, device


class VinhAgent:
    """Loaded model + retrieval module; paths can point to a persisted Drive folder."""

    def __init__(self, model_dir=None, chroma_dir=None):
        root = Path(model_dir or os.environ.get("VINH_MODEL_DIR", "outputs"))
        model_path = root / "vietjobs_skill_extractor.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing trained classifier: {model_path}")
        # ONLY load artifacts that you created/trust: joblib is Python pickle.
        self.artifact = joblib.load(model_path)
        self.vectorizer = self.artifact["vectorizer"]
        self.classifier = self.artifact["classifier"]
        self.classes = list(self.artifact["classes"])
        self.mapping = self.artifact["skill_mapping"]
        self.threshold = float(self.artifact["threshold"])
        self.chroma_dir = Path(chroma_dir or os.environ.get("VINH_CHROMA_DIR", str(root / "chroma_db")))
        self._collection = None

    def extract_skills(self, text, top_k=12, threshold=None):
        if not isinstance(text, str) or not text.strip():
            return []
        probs = self.classifier.predict_proba(self.vectorizer.transform([text]))[0]
        cutoff = self.threshold if threshold is None else float(threshold)
        order = np.argsort(-probs)[:max(1, min(int(top_k), len(self.classes)))]
        return [
            {"skill_id": self.mapping[self.classes[i]],
             "skill_name": self.classes[i],
             "confidence": round(float(probs[i]), 4)}
            for i in order if probs[i] >= cutoff
        ]

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            try:
                self._collection = client.get_collection(name="vietjobs_jds")
            except Exception as exc:
                raise RuntimeError("ChromaDB chưa có index. Hãy chạy cell tạo embeddings trước.") from exc
        return self._collection

    def search_jobs(self, query, top_k=5, category=None):
        if not isinstance(query, str) or not query.strip():
            return []
        collection = self._get_collection()
        count = collection.count()
        if count < 1:
            return []
        embedder = get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True).tolist()
        args = {"query_embeddings": q_vec,
                "n_results": min(count, max(int(top_k) * 5, int(top_k))),
                "include": ["documents", "metadatas", "distances"]}
        if category:
            args["where"] = {"category": str(category)}
        results = collection.query(**args)
        seen = set()
        rows = []
        for chunk_id, doc, meta, distance in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0]
        ):
            job_id = str(meta.get("job_id") or chunk_id)
            if job_id in seen:
                continue
            seen.add(job_id)
            rows.append({
                "job_id": job_id,
                "chunk_id": chunk_id,
                "title": str(meta.get("title", "")),
                "category": str(meta.get("category", "")),
                "location": str(meta.get("location", "")),
                "snippet": str(doc)[:800],
                "distance": round(float(distance), 4),
            })
            if len(rows) >= int(top_k):
                break
        return rows

    def ask_rag(self, question, top_k=5, use_llm=False):
        hits = self.search_jobs(question, top_k=top_k)
        if not hits:
            return {"question": question, "answer": "Không tìm thấy JD phù hợp trong chỉ mục.",
                    "mode": "no_results", "sources": []}
        sources = [{"ref": f"[{i}]", **hit} for i, hit in enumerate(hits, 1)]
        evidence = "\n\n".join(
            f"[{i}] {h['title']} | {h['location']}\n{h['snippet'][:650]}"
            for i, h in enumerate(hits, 1)
        )
        if not use_llm:
            answer = "Các JD liên quan được tìm thấy:\n" + "\n".join(
                f"[{i}] {h['title']} ({h['location'] or 'không rõ địa điểm'}). "
                f"Thông tin: {h['snippet'][:180]}..." for i, h in enumerate(hits, 1)
            )
            return {"question": question, "answer": answer,
                    "mode": "retrieval_only", "sources": sources}
        import torch
        tokenizer, model, device = get_llm()
        messages = [
            {"role": "system", "content": (
                "Bạn là trợ lý phân tích tuyển dụng Việt Nam. Chỉ dùng các đoạn JD được cung cấp. "
                "Nếu thiếu bằng chứng, nói rõ không đủ dữ liệu. Trích dẫn tài liệu theo [1], [2]. "
                "Không bịa nguồn, số liệu, lương hoặc nhà tuyển dụng.")},
            {"role": "user", "content": (
                f"Câu hỏi: {question}\n\nCác JD truy xuất:\n{evidence[:3500]}\n\n"
                "Trả lời ngắn gọn bằng tiếng Việt, kèm trích dẫn [số] cho từng nhận định.")},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = messages[0]["content"] + "\n" + messages[1]["content"]
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3000)
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=220,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {"question": question, "answer": answer,
                "mode": "local_llm_rag", "sources": sources}
