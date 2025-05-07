import streamlit as st
from modules.document_processor import load_all_documents
from modules.vector_store import prepare_vectorstore
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAI, AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import StuffDocumentsChain
import os
from dotenv import load_dotenv

# Khởi tạo bộ nhớ
def init_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Tạo prompt câu hỏi mở rộng
follow_up_prompt = PromptTemplate(
    input_variables=["context", "question"],
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
    api_key = os.getenv("GITHUB_TOKEN")

    try:
        llm = AzureOpenAI(
            api_key=api_key,
            api_version="2023-05-15",
            azure_endpoint="https://models.inference.ai.azure.com",
            deployment_name="gpt-4o",
            temperature=0.7
        )
    except Exception:
        llm = OpenAI(
            api_key=api_key,
            base_url="https://models.inference.ai.azure.com",
            model_name="gpt-4o",
            temperature=0.7
        )

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
        input_key="question"  # Rất quan trọng!
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

    if "conversation" not in st.session_state:
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

    # Hiển thị lịch sử
    for msg in st.session_state.conversation:
        role = "👤 Người dùng" if msg["role"] == "user" else "🤖 AI"
        st.markdown(f"**{role}:** {msg['content']}")

    query = st.text_input("💬 Nhập câu hỏi hoặc gõ 'tiếp' để nhận câu hỏi mở rộng:")

    if query:
        if query.lower() == "tiếp":
            with st.spinner("🤖 Đang đặt câu hỏi mở rộng..."):
                chat_history_text = "\n".join([
                    f"Người dùng: {m['content']}" if m["role"] == "user" else f"AI: {m['content']}"
                    for m in st.session_state.conversation
                ])
                try:
                    docs = st.session_state.main_chain.retriever.get_relevant_documents("")
                    context_text = "\n".join([doc.page_content for doc in docs[:3]])

                    follow_up_response = st.session_state.follow_up_chain.invoke({
                        "context": context_text,
                        "chat_history": chat_history_text
                    })

                    follow_up_text = follow_up_response.get("text", "Bạn muốn biết thêm về chủ đề này?")
                    st.session_state.conversation.append({"role": "ai", "content": follow_up_text})
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Lỗi khi tạo câu hỏi mở rộng: {str(e)}")
                    st.exception(e)
        else:
            with st.spinner("🤖 Đang trả lời..."):
                try:
                    st.session_state.memory.chat_memory.add_user_message(query)

                    response = st.session_state.main_chain({"question": query})

                    st.session_state.conversation.append({"role": "user", "content": query})
                    answer_content = response.get('result', response.get('answer', "Không tìm thấy câu trả lời"))

                    st.session_state.memory.chat_memory.add_ai_message(answer_content)

                    source_docs = response.get('source_documents', [])
                    sources = [doc.metadata.get('filename', '') for doc in source_docs]

                    st.session_state.conversation.append({
                        "role": "ai",
                        "content": f"{answer_content}\n\n🔗 Nguồn: {sources}"
                    })

                    chat_history_text = "\n".join([
                        f"Người dùng: {m['content']}" if m["role"] == "user" else f"AI: {m['content']}"
                        for m in st.session_state.conversation
                    ])
                    context_text = "\n".join([doc.page_content for doc in source_docs[:3]])

                    follow_up_response = st.session_state.follow_up_chain.invoke({
                        "context": context_text,
                        "chat_history": chat_history_text
                    })

                    follow_up_text = follow_up_response.get("text", "Bạn muốn biết thêm về chủ đề này?")
                    st.session_state.conversation.append({
                        "role": "ai",
                        "content": f"💡 Bạn có muốn hỏi tiếp về: '{follow_up_text}'? (Gõ 'tiếp' để xem)"
                    })
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()
