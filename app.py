import os
import streamlit as st
from langchain.llms import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

def load_docs(url):
    """
    Load a web document that will provide additional context to LLM
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs

def get_text_chunks(text):
    """
    Split document into smaller sized chunks for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, 
                                                   chunk_overlap=100)
    chunks = text_splitter.split_documents(text) 
    return chunks   

def get_vector_store(text_chunks):
    """
    Embed and store document chunks in a vector store.
    We can retrieve from this vectorstore based on similarity search
    """
    vectorstore = Chroma.from_documents(documents=text_chunks,
                                        embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

def get_rag_chain(retriever):
    """
    Create a chain that takes a question, retrieves relevant documents, 
    constructs a prompt, passes that to a model"""
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    template = """SYSTEM: You are a question answer bot. 
                  Be factual in your response.
                  Respond to the following question: {question} only from 
                  the below context: {context}. 
                  If you don't know the answer, just say that you don't know.
                """
    rag_prompt_custom = PromptTemplate.from_template(template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )
    return rag_chain

def gen_answer(url, question):
    data = load_docs(url)
    chunks = get_text_chunks(data)
    vector = get_vector_store(chunks)
    ragchain = get_rag_chain(vector)
    answer = ragchain.invoke(question)
    return answer

st.title("Q&A App for your website")
web_input = st.text_input("Enter a website link")
user_question = st.text_input("Ask a question from the website")
st.button("Submit")
ans = gen_answer(web_input, user_question)
st.write(ans)
