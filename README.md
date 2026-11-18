# Vinh – NLP & RAG Specialist · Google Colab

Dự án tự chứa notebook **`Vinh_VietJobs_FULL_Training_RAG_API.ipynb`** để làm phần của Vinh trong sơ đồ 12 bước: **Bước 4: NLP/Skill Extraction Agent** và **Bước 9: RAG Agent**, cùng API cung cấp cho Supervisor (Việt).

## Chạy trên Google Colab (không cần API key)

1. Truy cập <https://colab.research.google.com/> → **File → Upload notebook** → chọn `Vinh_VietJobs_FULL_Training_RAG_API.ipynb`.
2. Vào **Runtime → Change runtime type**: CPU cho train và truy xuất nhỏ; GPU T4 nếu dự định bật LLM.
3. Chọn **Runtime → Run all**. Notebook tự tải `VietJobs.csv` công khai từ <https://huggingface.co/datasets/dinhieufam/VietJobs> (khoảng 103 MB). Có thể upload file `VietJobs.csv.gz` (~22,85 MB) qua Files ở Colab hoặc ở cell upload dự phòng nếu mạng chặn Hugging Face.
4. Đợi các ô **TRAIN**, **VALIDATION/TEST**, **ChromaDB**, **API smoke tests** hoàn tất. Các chỉ số sẽ được tính **tại runtime của bạn**; notebook không cài sẵn kết quả F1.
5. Kết quả tại `/content/vinh_vietjobs/outputs/`. Ô cuối tạo `Vinh_NLP_RAG_Artifacts.zip`. Đặt `AUTO_DOWNLOAD_ZIP=True` để tự tải xuống hoặc dùng thanh Files của Colab.

### Cấu hình quan trọng

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ONLY_IT` | `True` | Chỉ JD CNTT, không phải 48.092 JD là IT |
| `MAX_TRAIN_ROWS` | `None` | Dùng tất cả JD IT có nhãn sau lọc |
| `MIN_TRAIN_POS` | `12` | Số ví dụ dương tối thiểu của nhãn ở tập train |
| `MAX_LABELS` | `35` | Số nhãn kỹ năng được mô hình học |
| `RAG_LIMIT` | `1000` | Giới hạn số JD để lập chỉ mục trên Colab |
| `ENABLE_LLM` | `False` | `True` dùng Qwen2.5-0.5B-Instruct local, không dùng API |
| `EXPORT_CHROMA` | `False` | Có đóng gói ChromaDB cùng artifacts hay không |

**Lưu ý:** `Run all` với `ENABLE_LLM=False` đã có **Agent retrieval** thực sự và API, nhưng phần câu trả lời ở chế độ `retrieval_only` **không** phải câu trả lời do LLM sinh. Đặt `ENABLE_LLM=True` để tải model nguồn mở và chạy RAG có sinh văn bản, nên dùng GPU. Mô hình embedding và LLM chỉ được sử dụng pretrained, **không fine-tune**.

## Đầu ra

```text
/content/vinh_vietjobs/
├── data/
│   ├── VietJobs.csv                 # tự tải, không cần commit Git
│   └── retrieval_annotation_template.csv
├── outputs/
│   ├── vietjobs_skill_extractor.joblib
│   ├── skill_id_mapping.csv
│   ├── metrics.json
│   ├── per_skill_report.csv
│   ├── rag_build_report.json
│   └── chroma_db/                   # Persistent ChromaDB
├── src/
│   ├── vinh_agents.py
│   ├── api.py
│   └── __init__.py
└── Vinh_NLP_RAG_Artifacts.zip
```

`skill_id_mapping.csv` chứa **48 kỹ năng trong taxonomy v1**, với ID ổn định; cột `trained` cho biết kỹ năng có trong tối đa 35 lớp được học hay không. Số mẫu đã qua lọc và F1 được ghi trực tiếp trong `metrics.json` sau khi chạy.

## API cho Supervisor Agent

Sau khi sao chép `src/`, model và **cả thư mục ChromaDB** sang máy/server, cài dependencies trong `requirements-colab.txt` (kể cả trên server local có Python 3.11). Trên PowerShell, tại thư mục dự án có `src/`:

```powershell
$env:VINH_MODEL_DIR="E:\path\to\outputs"
$env:VINH_CHROMA_DIR="E:\path\to\outputs\chroma_db"
python -m pip install -r requirements-colab.txt
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000/docs`. API nhận:

```http
POST /skills/extract
Content-Type: application/json

{"text":"Tuyển Python Backend Developer, Docker, SQL", "top_k":8}
```

```http
POST /jobs/search
Content-Type: application/json

{"query":"Tuyển Python backend Docker", "top_k":5}
```

```http
POST /rag/ask
Content-Type: application/json

{"question":"Những yêu cầu cho Python backend?", "top_k":5, "use_llm":false}
```

Chạy `use_llm=true` có thể tải thêm mô hình Qwen và yêu cầu RAM/GPU. Colab `TestClient` kiểm thử HTTP in-process, **không** tự xuất API ra internet. Nếu không sao chép ChromaDB sang máy/server, chỉ endpoint `/skills/extract` hoạt động; API truy xuất sẽ trả lỗi.

## Đánh giá và hạn chế

- **Skill Extraction:** TF-IDF + One-vs-Rest Logistic Regression, train/validation/test 80/10/10. `technical_skills` chỉ tạo nhãn (Y), **không** đưa vào input (X). Chỉ số Precision/Recall/F1 đo khả năng khớp **nhãn có sẵn của VietJobs đã chuẩn hóa**, không phải nhãn do con người kiểm chứng.
- **Embedding/RAG:** mô hình `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` pretrained, ChromaDB sử dụng cosine. Mặc định lập chỉ mục 1.000 JD IT và tối đa 3 đoạn/JD (có thể tăng giới hạn để chạy quy mô lớn).
- **Retrieval evaluation:** có self-retrieval proxy để kiểm tra hệ thống. Kết quả này **không** đủ làm Recall@K độc lập. Muốn đánh giá đúng, tạo `data/retrieval_gold.csv` với cột `question,relevant_job_ids` và các ID được gán thủ công (cách nhau `|`).
- Cần đánh giá độc lập thêm trên JD thu thập từ VietnamWorks/CareerViet trước khi tuyên bố khả năng tổng quát hóa. Chưa có đồng bộ PostgreSQL thực tế với module của Vũ; `src/api.py` chỉ cung cấp giao diện thống nhất cho Việt tích hợp.

## Đưa lên GitHub

Có thể commit `Vinh_VietJobs_FULL_Training_RAG_API.ipynb`, `src/`, `requirements-colab.txt`, `README.md`, và tài liệu trong `data/`. `.gitignore` mặc định loại bỏ mô hình `.joblib`, chỉ mục ChromaDB và CSV gốc. Nếu nhóm muốn đăng dataset, hãy kiểm tra quyền phân phối lại và kích thước file; **không cần đăng dataset** vì notebook tự tải nguồn công khai.

**Nguồn tài liệu:** [VietJobs dataset](https://huggingface.co/datasets/dinhieufam/VietJobs), [Chroma Cookbook](https://cookbook.chromadb.dev/core/), [Sentence Transformers](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).
