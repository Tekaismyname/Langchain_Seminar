import streamlit as st
from modules.document_processor import load_all_documents
from modules.vector_store import prepare_vectorstore
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from collections import defaultdict
from dotenv import load_dotenv
import os

@st.cache_resource(show_spinner="🔄 Đang tải và vector hóa tài liệu...")
def load_vectorstore_and_docs():
    docs = load_all_documents("data/")
    vectorstore = prepare_vectorstore(docs)
    return vectorstore, docs

# Khởi tạo bộ nhớ
def init_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="result"
    )

def build_llm_chain():
    llm = Ollama(model="mistral")
    
    # load_dotenv()
    
    # llm = OpenAI(
    # base_url="https://models.inference.ai.azure.com",
    # api_key= os.getenv("GITHUB_TOKEN"),
    # model="gpt-4o",
    # temperature=0.1,
    # )
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # print(f"🔑 OPENAI_API_KEY: {openai_api_key}")
    # if not openai_api_key:
    #     st.error("❌ OPENAI_API_KEY chưa được cấu hình trong .env hoặc biến môi trường!")
    #     st.stop()

    # llm = ChatOpenAI(
    #     openai_api_key = openai_api_key,
    #     model_name="gpt-3.5-turbo",
    #     temperature=0.3
    # )
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Trả lời câu hỏi dựa trên tài liệu sau. Nếu không có thông tin phù hợp, bạn vẫn có thể sử dụng kiến thức của mình để đưa ra câu trả lời hợp lý.
        Tên tài liệu được tham chiếu: {filename}
        Tài liệu: {context}
        Câu hỏi: {question}
        Trả lời:
        """
    )
    return LLMChain(llm=llm, prompt=prompt)

def main():
    st.title("🤖 Trợ lý AI")

    if "docs" not in st.session_state:
        vectorstore, docs = load_vectorstore_and_docs()
        st.session_state.vectorstore = vectorstore
        st.session_state.docs = docs
        st.session_state.llm_chain = build_llm_chain()
        st.session_state.memory = init_memory()
        st.session_state.conversation = []
        st.session_state.stop_generation = False
        st.success("✅ Hệ thống đã sẵn sàng!")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False

    # Hiển thị danh sách tài liệu đã load
    with st.expander("📄 Danh sách tài liệu đã load"):
        filenames = sorted(list({
            doc.metadata["filename"]
            for doc in st.session_state.docs
            if "filename" in doc.metadata and doc.metadata["filename"]
        }))
        st.markdown("\n".join(f"- {f}" for f in filenames))

    col1, col2 = st.columns([5, 1])

    with col1:
        selected_file = st.selectbox("📂 Chọn tài liệu để tham chiếu:", filenames)
        with st.form("form_input"):
            query = st.text_input("💬 Câu hỏi của bạn:", key="query_input")
            submitted = st.form_submit_button("Gửi")

    with col2:
        if st.button("⏹ Stop Generating"):
            st.session_state.stop_generation = True

    if submitted and query:
        with st.spinner("🤖 Đang xử lý..."):
            try:
                if st.session_state.stop_generation:
                    st.warning("⛔ Phản hồi đã bị dừng trước khi bắt đầu.")
                    st.session_state.stop_generation = False
                    return

                st.session_state.conversation.append({"role": "user", "content": query})
                st.session_state.memory.chat_memory.add_user_message(query)

                matched_docs = [doc.page_content for doc in st.session_state.docs if doc.metadata.get("filename") == selected_file]
                context = f"Tên file: {selected_file}" + "".join(matched_docs)

                response = st.session_state.llm_chain.invoke({
                    "filename": selected_file,
                    "context": context,
                    "question": query
                })

                if st.session_state.stop_generation:
                    st.warning("⛔ Đã dừng phản hồi sau khi gửi.")
                    st.session_state.stop_generation = False
                    return

                answer = response.get("text", "Không có câu trả lời.")
                st.session_state.memory.chat_memory.add_ai_message(answer)
                st.session_state.conversation.append({"role": "ai", "content": answer})

            except Exception as e:
                st.error(f"Lỗi khi xử lý: {e}")

        st.session_state.pop("query_input", None)
        st.session_state.stop_generation = False

    # Hiển thị hội thoại gần nhất
    for msg in st.session_state.conversation:
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"**{icon}**: {msg['content']}")

if __name__ == "__main__":
    main()
