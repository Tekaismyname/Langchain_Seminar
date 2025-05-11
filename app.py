import os
import streamlit as st
from dotenv import load_dotenv
from modules.document_processor import load_all_documents
from modules.vector_store import prepare_vectorstore
from langchain.chains.question_answering import load_qa_chain
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOllama
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load biến môi trường từ .env
load_dotenv()

# Tải tài liệu và tạo vectorstore từ FAISS
@st.cache_resource(show_spinner="🔄 Đang tải tài liệu và tạo vector...")
def load_docs_and_vectorstore():
    docs = load_all_documents("data/")
    vectorstore = prepare_vectorstore(docs)
    return docs, vectorstore

# Cấu hình LLM từ OpenAI (có thể dùng Azure)
def init_llm():
    # return ChatOllama(model="mistral") 
    return OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key= "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",
    model="gpt-4o",
    temperature=0.1,
    )



# Tạo agent RAG với các công cụ riêng biệt
def build_agent(llm, retriever, memory):
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff")

    def retrieve_docs_fn(q):
        docs = retriever.get_relevant_documents(q)
        return "\n\n".join([doc.page_content for doc in docs])

    def answer_from_docs_fn(q):
        docs = retriever.get_relevant_documents(q)
        return qa_chain.run(input_documents=docs, question=q)
    
    def generic_csv_analysis_fn(query):
        import re

        match = re.search(r"trong file (\w+\.csv)", query)
        if not match:
            return "❌ Không tìm thấy tên file CSV trong câu hỏi."

        filename = match.group(1)

        # Mẫu điều kiện: tìm người trên 50 tuổi
        def condition_fn(df):
            if "age" not in df.columns:
                return f"❌ File '{filename}' không có cột 'age'."
            count = df[df["age"] > 50].shape[0]
            return f"✅ File '{filename}' có {count} người trên 50 tuổi."

        from modules.document_processor import analyze_csv_by_filename
        return analyze_csv_by_filename(filename, condition_fn)


    summarize_prompt = PromptTemplate(
        input_variables=["text"],
        template="Tóm tắt nội dung sau một cách ngắn gọn và logic:\n\n{text}\n\nTóm tắt:"
    )
    summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

    def summarize_fn(q):
        docs = retriever.get_relevant_documents(q)
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        return summarize_chain.run(text=combined_text)

    tools = [
        Tool(
            name="RetrieveDocuments",
            func=retrieve_docs_fn,
            description="Use this tool to retrieve relevant documents from the uploaded data. Always reply in ReAct format like: Thought: ... Action: RetrieveDocuments Action Input: your query"
        ),
        Tool(
            name="AnswerFromDocs",
            func=answer_from_docs_fn,
            description="Use this tool to answer questions using retrieved documents. Always reply in ReAct format like: Thought: ... Action: AnswerFromDocs Action Input: your question"
        ),
        Tool(
            name="SummarizeContent",
            func=summarize_fn,
            description="Use this tool to summarize documents relevant to the query. Always reply in ReAct format like: Thought: ... Action: SummarizeContent Action Input: your query"
        ),
        Tool(
            name="AnalyzeAnyCSV",
            func=generic_csv_analysis_fn,
            description="Dùng để phân tích file CSV cụ thể. Ví dụ: 'Có bao nhiêu người trên 50 tuổi trong file insurance.csv?'"
        )

    ]

    return initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=False,  # ✅ Tắt verbose để không log Thought/Action
    handle_parsing_errors=True,
    return_intermediate_steps=False,  # ✅ Chỉ trả Final Answer
    agent_kwargs={
        "prefix": (
            "Bạn là trợ lý AI. Nếu người dùng chỉ chào hỏi ('hi', 'hello', 'xin chào'), "
            "bạn sẽ trả lời thẳng mà không cần thực hiện Action nào.\n\n"
            "Nếu câu hỏi cần dữ liệu, bạn sẽ dùng Thought → Action → Observation."
        )
    }
)




# Giao diện người dùng

def main():
    st.set_page_config(page_title="Agentic RAG", page_icon="🤖")
    st.title("🧠 Agentic RAG Assistant")

    if "agent" not in st.session_state:
        docs, vectorstore = load_docs_and_vectorstore()
        llm = init_llm()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        agent = build_agent(llm, retriever, memory)

        st.session_state.docs = docs
        st.session_state.vectorstore = vectorstore
        st.session_state.llm = llm
        st.session_state.memory = memory
        st.session_state.agent = agent
        st.session_state.conversation = []

        st.success("✅ Agent đã sẵn sàng!")

    # Form nhập câu hỏi
    query = st.text_input("💬 Nhập câu hỏi:")
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("⏹ Dừng"):
            st.session_state.stop_generation = True

    # Xử lý câu hỏi
    if query:
        with st.spinner("🤖 Đang xử lý..."):
            if st.session_state.get("stop_generation"):
                st.warning("⛔ Đã dừng phản hồi.")
                st.session_state.stop_generation = False
                return
            try:
                st.session_state.conversation.append({"role": "user", "content": query})
                result = st.session_state.agent.invoke({"input": query})
                answer = result.get("output", "")  # Chỉ lấy Final Answer gọn gàng
                st.session_state.conversation.append({"role": "ai", "content": answer})
            except Exception as e:
                st.error(f"❌ Lỗi khi xử lý: {e}")
            st.session_state.stop_generation = False

    # Hiển thị hội thoại
    for msg in st.session_state.conversation[-10:]:
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"**{icon}**: {msg['content']}")

if __name__ == "__main__":
    main()
