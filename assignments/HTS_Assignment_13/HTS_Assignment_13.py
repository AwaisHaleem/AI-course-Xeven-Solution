import os
import docx

import streamlit as st


from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from streamlit_chat import message
from langchain.llms import OpenAIChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback

open_ai_api = st.secrets["OPENAI_API_KEY"]


def main():
    st.set_page_config(page_title="Talk to your document")
    st.header("Document Gpt")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "process_complete" not in st.session_state:
        st.session_state.process_complete = None

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "UPLOAD FILES HERER", type=["docx", "pdf"], accept_multiple_files=True)
        proceed = st.button("Proceed")
    if proceed:
        file_text = get_file_text(uploaded_files)
        st.write("Files Uploaded")
        # text_chunks
        text_chunks = get_text_chunks(file_text)
        st.write("text_chunks made")
        # create embeddings
        vector_store = get_vector_store(text_chunks)
        st.write("vector store created")
        # create conversation chain
        st.session_state.conversation = get_conversation_chain(
            vector_store, open_ai_api)
        st.session_state.process_complete = True
    if st.session_state.process_complete == True:
        user_question = st.chat_input("Ask Questions about your files")
        if user_question:
            handle_user_input(user_question)


def get_file_text(files):
    text = ""
    for file in files:
        file_type = os.path.splitext(file.name)[1]
        if file_type == ".pdf":
            text += read_pdf_file(file)
        elif file_type == ".docx":
            text += read_docx_file(file)
    return text


def read_pdf_file(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def read_docx_file(file):
    text = ""
    document = docx.Document(file)
    for para in document.paragraphs:
        text += para.text
    return text


def get_vector_store(chunks):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding)
    return vector_store


def get_text_chunks(text):
    chunk_spliter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                   chunk_overlap=20,
                                                   length_function=len,)
    chunks = chunk_spliter.split_text(text=text)
    return chunks


def get_conversation_chain(store, key):
    llm = OpenAIChat(openai_api_key=key,
                     temperature=0.5,
                     model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(input):
    with get_openai_callback() as cb:

        response = st.session_state.conversation({"question": input})
        st.write(input)
    st.session_state.chat_history = response["chat_history"]

    # layout in/out response container
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if (i % 2 == 0):
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))


if __name__ == main():
    main()
