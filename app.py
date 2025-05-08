import streamlit as st
from modules.document_processor import load_all_documents
from modules.vector_store import prepare_vectorstore
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAI, AzureOpenAI
from langchain.chat_models import ChatOpenAI 
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import StuffDocumentsChain
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Khởi tạo bộ nhớ
def init_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result"  # Thêm output_key để chỉ định rõ khóa cần lưu
    )

# Tạo prompt câu hỏi mở rộng
follow_up_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""
    Bạn là trợ lý AI. Dựa trên tài liệu và hội thoại trước, hãy đặt một câu hỏi mở rộng:

    Tài liệu:
    {context}

    Lịch sử hội thoại:
    {chat_history}

    Hãy đặt một câu hỏi mở rộng:
    """
)

def build_rag_chain(vectorstore):
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    
    llm = ChatOpenAI(
        openai_api_base="https://models.inference.ai.azure.com",
        openai_api_key=github_token,
        model="gpt-4o",
        temperature=0.3
    )
    
    # llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", huggingfacehub_api_token= huggingface_token)

    memory = init_memory()

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Sử dụng thông tin sau để trả lời câu hỏi. Nếu không biết thì nói không biết.

        Tài liệu: {context}
        Câu hỏi: {question}
        Trả lời:
        """
    )

    llm_chain = LLMChain(llm=llm, prompt=qa_prompt)

    # ✅ Bọc vào StuffDocumentsChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"  # << khớp với tên trong PromptTemplate
    )

    # ✅ Dùng RetrievalQA thủ công
    main_chain = RetrievalQA(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        combine_documents_chain=combine_documents_chain,
        return_source_documents=True,
        memory=memory,
        input_key="question",  # Rất quan trọng!
        output_key="result"    # Chỉ định output_key để khớp với memory
    )

    # ✅ Follow-up chain
    follow_up_chain = LLMChain(
        llm=llm,
        prompt=follow_up_prompt,
        output_key="text"
    )

    return main_chain, follow_up_chain, memory


def main():
    st.title("🤖 AI Agent Thông Minh")

    # Khởi tạo session state nếu chưa có
    if "main_chain" not in st.session_state:
        with st.spinner("🔄 Đang khởi tạo..."):
            try:
                documents = load_all_documents("data/")
                vectordb = prepare_vectorstore(documents)
                st.session_state.main_chain, st.session_state.follow_up_chain, st.session_state.memory = build_rag_chain(vectordb)
                st.session_state.conversation = []
                st.success("✅ Sẵn sàng!")
            except Exception as e:
                st.error(f"Lỗi khởi tạo: {str(e)}")
                st.exception(e)
                return

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "query" not in st.session_state:
        st.session_state.query = ""

    # Hiển thị hội thoại
    for msg in st.session_state.conversation:
        role = "👤 Người dùng" if msg["role"] == "user" else "🤖 AI"
        st.markdown(f"**{role}:** {msg['content']}")

    # Nhập câu hỏi
    query = st.text_input("💬 Nhập câu hỏi hoặc gõ 'tiếp' để nhận câu hỏi mở rộng:", key="query")

    if query:
        with st.spinner("🤖 Đang xử lý..."):

            chat_history_text = "\n".join([
                f"Người dùng: {m['content']}" if m["role"] == "user" else f"AI: {m['content']}"
                for m in st.session_state.conversation
            ])

            try:
                if query.lower() == "tiếp":
                    docs = st.session_state.main_chain.retriever.get_relevant_documents("")

                    if docs:
                        context_text = "\n".join([doc.page_content[:1000] for doc in docs[:3]])

                        follow_up_response = st.session_state.follow_up_chain.invoke({
                            "context": context_text,
                            "chat_history": chat_history_text
                        })
                        follow_up_text = follow_up_response.get("text", "Bạn muốn biết thêm về chủ đề này?")
                        st.session_state.conversation.append({"role": "ai", "content": follow_up_text})
                else:
                    # Lưu câu hỏi
                    st.session_state.conversation.append({"role": "user", "content": query})
                    st.session_state.memory.chat_memory.add_user_message(query)

                    # Truy vấn hệ thống chính
                    response = st.session_state.main_chain({"question": query})
                    answer = response.get("result", response.get("answer", "Không tìm thấy câu trả lời"))
                    st.session_state.memory.chat_memory.add_ai_message(answer)

                    # Hiển thị kết quả
                    source_docs = response.get("source_documents", [])
                    sources = [doc.metadata.get('filename', '') for doc in source_docs]

                    st.session_state.conversation.append({
                        "role": "ai",
                        "content": f"{answer}\n\n🔗 Nguồn: {sources}"
                    })

                    # Gợi ý follow-up nếu có tài liệu
                    if source_docs:
                        context_text = "\n".join([doc.page_content[:1000] for doc in source_docs[:3]])

                        follow_up_response = st.session_state.follow_up_chain.invoke({
                            "context": context_text,
                            "chat_history": chat_history_text + f"\nNgười dùng: {query}\nAI: {answer}"
                        })
                        follow_up_text = follow_up_response.get("text", "Bạn muốn biết thêm về chủ đề này?")

                        st.session_state.conversation.append({
                            "role": "ai",
                            "content": f"💡 Bạn có muốn hỏi tiếp về: '{follow_up_text}'? (Gõ 'tiếp' để xem)"
                        })

            except Exception as e:
                st.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
                st.exception(e)

        # Reset ô nhập sau khi xử lý
        st.session_state.query = ""


if __name__ == "__main__":
    main()