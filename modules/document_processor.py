from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.schema import Document
import pandas as pd
import os

def load_pdf(path):
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
        docs = PyMuPDFLoader(path).load()
        filename = os.path.basename(path)
        for doc in docs:
            doc.metadata["filename"] = filename
        return docs
    except Exception as e:
        print(f"❌ Lỗi khi load PDF '{path}': {e}")
        return []
    
    
def load_docx(path):
    docs = Docx2txtLoader(path).load()
    filename = os.path.basename(path)
    for doc in docs:
        doc.metadata["filename"] = filename
    return docs

def load_csv_as_docs(path, filename, batch_size=5):
    docs = []
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python")
    except Exception as e:
        print(f"⚠️ Không thể đọc file CSV '{filename}': {e}")
        return docs

    columns_info = f"File này có các cột: {', '.join(df.columns)}.\n"
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size]
        rows = [f"Dòng {i+j+1}: {', '.join([f'{col}={row[col]}' for col in df.columns])}"
                for j, row in chunk.iterrows()]
        content = columns_info + "\n".join(rows)
        content = content.replace("```", "").replace('"""', "").replace("#", "")
        docs.append(Document(
            page_content=content,
            metadata={"filename": filename, "start_row": i+1}
        ))
    return docs

def load_all_documents(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        try:
            if file.endswith(".pdf"):
                docs.extend(load_pdf(full_path))
            elif file.endswith(".docx"):
                docs.extend(load_docx(full_path))
            elif file.endswith(".csv"):
                docs.extend(load_csv_as_docs(full_path, file))
        except Exception as e:
            print(f"❌ Lỗi khi load '{file}': {e}")
    return docs
