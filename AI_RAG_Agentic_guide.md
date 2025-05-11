# 🧠 Agentic RAG Assistant

Một hệ thống hỏi đáp tài liệu có khả năng:
- Trích xuất tài liệu từ `.pdf`, `.docx`, `.csv`
- Xử lý câu hỏi tự nhiên từ người dùng
- Truy xuất dữ liệu vector (FAISS)
- Trả lời câu hỏi, phân tích CSV, tóm tắt nội dung
- Tích hợp UI với **Streamlit**

---

## 📦 Yêu cầu thư viện (`requirements.txt`)

```txt
streamlit
python-dotenv
langchain
langchain-openai
langchain-community
faiss-cpu
sentence-transformers
pymupdf
docx2txt
pandas
```

---

## 📁 Cấu trúc thư mục

```text
├── app.py                       # Entry chính - khởi chạy Streamlit và AI Agent
├── modules/
│   ├── document_processor.py    # Load và phân tích tài liệu (.pdf, .docx, .csv)
│   └── vector_store.py          # Xử lý embeddings và FAISS vectorstore
├── data/                        # Thư mục chứa tài liệu đầu vào
├── .env                         # Biến môi trường
├── requirements.txt             # Danh sách thư viện cần cài
```

---

## 🔁 Luồng xử lý chính

1. **Tải tài liệu:** `load_all_documents()` đọc tất cả `.pdf`, `.docx`, `.csv` trong `data/`
2. **Tạo vectorstore:** Dùng FAISS và mô hình `sentence-transformers` để sinh embedding
3. **Khởi tạo LLM:** Dùng `OpenAI` hoặc `ChatOllama`
4. **Tạo Agent:** Sử dụng `initialize_agent()` với các công cụ:
   - Truy xuất tài liệu
   - Trả lời từ tài liệu
   - Phân tích CSV
   - Tóm tắt tài liệu
5. **Giao diện:** Người dùng nhập câu hỏi → Agent xử lý → Hiển thị kết quả

---

## 🔧 Các hàm chính dùng từ LangChain

| Hàm / Lớp | Mô tả |
|----------|-------|
| `load_qa_chain` | Tạo chuỗi QA đơn giản, dùng `stuff` để nạp toàn bộ document |
| `LLMChain` | Tạo chuỗi prompt cho tác vụ tóm tắt |
| `Tool` | Định nghĩa các công cụ cho agent (trả lời, tóm tắt, phân tích CSV,...) |
| `initialize_agent` | Tạo agent dạng Zero-shot ReAct sử dụng các Tool đã khai báo |
| `ConversationBufferMemory` | Lưu lại hội thoại để tạo trải nghiệm tương tác mượt mà |
| `RecursiveCharacterTextSplitter` | Chia nhỏ tài liệu đầu vào để tạo embedding hiệu quả |
| `FAISS.from_documents()` | Dùng để xây dựng vectorstore từ document đã nhúng |
| `HuggingFaceEmbeddings` | Sinh vector embedding từ model HuggingFace |

---

## 📌 Ví dụ Tool Agent hoạt động

Ví dụ: người dùng hỏi  
> "Có bao nhiêu người trên 50 tuổi trong file insurance.csv?"

Agent sử dụng tool `AnalyzeAnyCSV`, gọi đến `analyze_csv_by_filename()` và áp điều kiện lọc.

---

## ✅ Tính năng nổi bật

- 🔎 Truy xuất chính xác nội dung tài liệu
- 📑 Hiểu cả file PDF, Word, CSV
- 📊 Tự động phân tích dữ liệu dạng bảng
- 💬 Giao diện thân thiện, dễ dùng với Streamlit