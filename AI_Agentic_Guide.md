
# Hướng Dẫn Triển Khai Ứng Dụng AI Agentic Với LangChain + OpenAI

## 📁 Cấu Trúc Dự Án
```
├── app.py
├── .env
├── requirements.txt
├── data/               # Thư mục chứa các tài liệu .csv, .docx, .pdf
├── modules/
│   ├── document_processor.py
│   └── vector_store.py
```

---

## ⚙️ 1. Cài Đặt Môi Trường

### 📦 requirements.txt
```txt
streamlit
langchain>=0.1.14
langchain-openai
python-dotenv
huggingface_hub
faiss-cpu
sentence-transformers
pandas
docx2txt
pymupdf
python-docx
```

### 📄 .env (OpenAI)
```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 📄 .env (Azure OpenAI)
```env
OPENAI_API_TYPE=azure
OPENAI_API_BASE=https://<your-resource-name>.openai.azure.com/
OPENAI_API_KEY=<your-key>
OPENAI_API_VERSION=2023-12-01-preview
```

---

## 🧠 2. Mô Tả Chức Năng Từng File

### 🔹 `app.py`
Ứng dụng chính chạy với `streamlit`, cho phép:
- Tải dữ liệu
- Hiển thị danh sách tài liệu
- Người dùng chọn tài liệu
- Nhập câu hỏi
- Nhận câu trả lời từ AI
- Dừng phản hồi khi cần

Các thành phần chính:
```python
load_vectorstore_and_docs() → load tài liệu và tạo vector store
build_llm_chain() → cấu hình LLM từ OpenAI/Azure
init_memory() → khởi tạo bộ nhớ hội thoại
```

Luồng xử lý:
```python
1. Người dùng chọn tài liệu từ dropdown
2. Nhập câu hỏi
3. Hệ thống lấy context từ tài liệu đã chọn
4. Gửi context + câu hỏi vào prompt → AI phản hồi
```

---

### 🔹 `modules/document_processor.py`
Chứa các hàm để xử lý và chuyển đổi tài liệu thành `Document`:

```python
def load_pdf(path):
    → Sử dụng PyMuPDF để load file PDF, thêm metadata['filename']

def load_docx(path):
    → Load file DOCX bằng docx2txt và gán filename

def load_csv_as_docs(path, filename):
    → Chia nhỏ file CSV thành các đoạn và convert thành Document

def load_all_documents(folder_path):
    → Tự động load toàn bộ các file trong thư mục /data
```

---

### 🔹 `modules/vector_store.py`
Dùng FAISS để:
```python
def prepare_vectorstore(docs):
    → Tạo embeddings từ SentenceTransformer
    → Dùng FAISS để index dữ liệu
```

---

## 🤖 3. Mô Tả Chi Tiết AI Agent

### 🧱 Prompt Template (trong `build_llm_chain()`)
```python
prompt = PromptTemplate(
    input_variables=["context", "question", "filename"],
    template="""
    Bạn là một trợ lý thông minh. Trả lời câu hỏi dựa trên tài liệu dưới đây.

    Tên tài liệu: {filename}
    Nội dung tài liệu: {context}
    Câu hỏi: {question}
    Trả lời:
    """
)
```

### 🤖 LLM Chain
```python
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
```

### 🔄 Memory
```python
ConversationBufferMemory
→ Lưu hội thoại giữa người dùng và AI để giữ ngữ cảnh
```

---

## 🚀 4. Chạy Ứng Dụng
```bash
streamlit run app.py
```
> Hệ thống sẽ hiển thị giao diện gồm dropdown tài liệu, ô nhập câu hỏi và phản hồi từ AI.

---

## 🧪 5. Mở Rộng Gợi Ý
- Thêm tuỳ chọn nhiều tài liệu cùng lúc
- Export hội thoại ra file .txt
- Thêm đánh giá phản hồi

---

## 📞 Liên hệ hỗ trợ
Nếu bạn cần hỗ trợ tích hợp Azure, Ollama hoặc private model, hãy liên hệ nhóm phát triển.
