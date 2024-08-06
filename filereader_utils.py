from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

def qa_agent(openai_api_key, memory, uploaded_files, question):
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
    all_texts = []

    # 遍历所有上传的文件
    for uploaded_file in uploaded_files:
        temp_file_path = f"temp_{uploaded_file.name}"
        file_content = uploaded_file.read()

        # 将文件写入临时文件
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file_content)

        # 根据文件类型加载内容
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(temp_file_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(temp_file_path)

        elif uploaded_file.type == "text/csv":
            loader = CSVLoader(temp_file_path)
        elif uploaded_file.type == "application/json":
            loader = JSONLoader(temp_file_path)

        else:
            continue  # 如果文件类型不支持，跳过

        docs = loader.load()
        all_texts.extend(docs)  # 将加载的文档添加到所有文本中

    # 将所有文本分割为小块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ".", "!", "?", ",", ";", ":", "-", "—", "(", ")", "[", "]", "{", "}"])
    texts = text_splitter.split_documents(all_texts)

    # 创建向量数据库
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.from_documents(texts, embeddings_model)

    # 创建检索器和问答链
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    # 获取回答
    response = qa.invoke({"chat_history": memory, "question": question})
    return response