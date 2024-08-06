import streamlit as st
from langchain.memory import ConversationBufferMemory
from filereader_utils import qa_agent

st.title("📑 AI智能文件阅读助手")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API key](https://platform.openai.com/account/api-keys)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

# 允许用户上传最多100个文件
uploaded_files = st.file_uploader("上传你的文件（支持PDF、Docx、xlsx、txt、pptx、csv、json、jpg、png）", type=["pdf", "docx", "xlsx", "txt", "pptx", "csv", "json", "jpg", "png"], accept_multiple_files=True)

question = st.text_input("对文件内容进行提问", disabled=not uploaded_files)

if uploaded_files and question and not openai_api_key:
    st.info("请输入你的OpenAI API密钥")

if uploaded_files and question and openai_api_key:
    with st.spinner("AI正在思考中，请稍等..."):
        response = qa_agent(openai_api_key, st.session_state["memory"],
                            uploaded_files, question)
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()