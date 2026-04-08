import pdfplumber
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# AZURE_OPENAI_API_KEY="FC6nsAyYXnzuFlXHdr8yMnmCNFV2aB5TSmCHQSHtvIVd0q0ahHjcJQQJ99BJACYeBjFXJ3w3AAABACOGpoPg"

# load_dotenv()
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# ##api_key = os.getenv("AZURE_OPENAI_API_KEY")
# embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
# chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# azure_embeddings = AzureOpenAIEmbeddings(
#     azure_endpoint=azure_endpoint,
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_deployment=embed_deployment,
#     api_version=api_version
# )
# st.header("My first Chatbot")

# with st.sidebar:
#     st.title("Your Documents")
#     file = st.file_uploader("Upload a pdf file and Start asking questions", type="pdf")

# if file is not None:
#     with pdfplumber.open(file) as pdf:
#         text = ""
#         for page in pdf.pages:
#             text+=page.extract_text() + "\n"
#     #st.write(text)

#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", ". "," ", ""],
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     chunks= text_splitter.split_text(text)
#     st.write(chunks)

#     # embeddings = AzureOpenAIEmbeddings(
#     #     azure_endpoint=
#     #     model= "My-text-embeddings",
#     #     azure_openai_api_key= AZURE_OPENAI_API_KEY
#     # )

#     vector_store = FAISS.from_texts(chunks,azure_embeddings)
    
#     user_question = st.text_input("Type your question here")
    
#     def format_docs(docs):
#         return "\n\n".join([doc.page_content for doc in docs])

#     retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k":4}
#     )

#     #define the LLM and prompts
#     llm = AzureChatOpenAI(
#         azure_endpoint=azure_endpoint,
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         azure_deployment="gpt-5",
#         temperature=0.3,
#         max_tokens=1000
        
#     )

#     #provide the prompts
#     prompt = ChatPromptTemplate.from_messages([
#         ("system",
#          "You are a helpful assistant answering questions about a PDF document.\n\n"
#          "Guidelines:\n"
#          "1. Provide complete, well-explained answers using the context below.\n"
#          "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
#          "3. If the context mentions related information, include it to give fuller picture.\n"
#          "4. Only use information from the provided context - do not use outside knowledge.\n"
#          "5. Summarize long information, ideally in bullets where needed\n"
#          "6. If the information is not in the context, say so politely.\n\n"
#          "Context:\n{context}"),
#         ("human", "{question}")
#     ])


#     chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#     )



#     if user_question:
#         response = chain.invoke(user_question)
#         st.write(response)

        
    



# import streamlit as st
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.messages import SystemMessage, HumanMessage



 # Example of passing credentials through environment file
load_dotenv()
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Azureembeddings
azure_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    api_key=api_key,
    deployment=embed_deployment,
    api_version=api_version,
    )

# Upload pdf files

st.header("My First Chat Bot")

with st.sidebar:
    st.title("Your documents")
    file=st.file_uploader("upload a file and start asking questions", type=["pdf","doc"])

# Extract the text

if file is not None:
    with pdfplumber.open(file) as pdf:
        text= ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        
#     with pdfplumber.open(file) as pdf:
#         text = ""
#         for page in pdf.pages:
#             text+=page.extract_text() + "\n"


    # Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks=text_splitter.split_text(text)

    # generate embeddings

    result = azure_embeddings.embed_query("Hello world!")

    # Assuming 'chunks' is your list of text chunks, as in your code:
    vectorstore = FAISS.from_texts(chunks, embedding=azure_embeddings)

    # added from frida
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # get user question
    user_question = st.text_input("Type your question here:")

    # do similarity search
    if user_question:   
        # Define LLM
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            openai_api_key=api_key,
            azure_deployment=chat_deployment,
            openai_api_version=api_version,
            )

         # --- Step 4: Retrieve relevant documents ---
        docs = retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # --- Step 5: Compose prompt and get answer ---
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=f"Use the following context to answer the question. \n\nContext:\n{context}\n\nQuestion: {user_question}\nAnswer:")
        ]
        response = llm.invoke(messages)
        st.write(response.content)


