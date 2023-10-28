from __future__ import absolute_import
from dotenv import load_dotenv
import datetime
from streamlit_chat import message
import os
from datetime import datetime
from llama_index import (GPTVectorStoreIndex, ServiceContext,
                         SimpleDirectoryReader)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import openai
import streamlit as st
from llama_index import StorageContext, load_index_from_storage
import llama_index

# Load environment variables
dotenv_path = '.env'
load_dotenv(dotenv_path)

openai.api_key = os.getenv("OPENAI_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages= ["Hello, Ask me any questions related to covid-19"]

client = QdrantClient(":memory:")

documents = SimpleDirectoryReader('data/').load_data()

service_context = ServiceContext.from_defaults(chunk_size=512)

vector_store = QdrantVectorStore(client=client, collection_name="Covid19_latest_guidelines")


st.set_page_config(
        page_title="Covid19 Bot🤖",
        page_icon="🤖"
    )

    
st.header("Covid19 Bot🤖 ")

start = datetime.now()
print("Started loading content...", start)

index = GPTVectorStoreIndex.from_documents(documents,vector_store=vector_store,service_context=service_context,show_progress=True)

print("Finished loading content...", datetime.now() - start)

query_engine = index.as_query_engine(similarity_top_k = 2)


user_input = st.sidebar.text_input("Enter your query:")
submit=st.sidebar.button("Generate_Response")

if user_input:
    st.session_state.messages.append(user_input)
    with st.spinner("loading..."):
        response = query_engine.query(user_input)
    st.session_state.messages.append(response.response)
    
messages= st.session_state.messages

for i,msg in enumerate(messages):
    if i%2==0:
        message(msg, is_user= False)
    else:
        message(msg, is_user= True)