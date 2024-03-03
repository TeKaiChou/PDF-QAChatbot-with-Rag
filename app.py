import os
import tempfile
import streamlit as st
from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from dotenv import load_dotenv
load_dotenv()

# Initialize streamlit app
st.set_page_config(page_title="PDF_RAG")
st.title("QA Chabot with RAG for PDF")
st.write("This integrated QA Chatbot applying retrieval augmented generation (RAG) considers the external data sources to answer specific questions based on the knowledge from the context of provided documents.")
st.divider()

# Load gemini pro LLM and embedding model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_pro = Gemini(models='gemini-pro', api_key=GOOGLE_API_KEY)
gemini_embed = GeminiEmbedding(models='embedding-001', api_key=GOOGLE_API_KEY)

# Load and process data into vectorindex
def process_pdf():
    # Load data, Create Embedding, Store into VectorStore
    if st.session_state['source_docs']:
        # 1. Load the original data into session_state['loaded_documents'] 
        with tempfile.TemporaryDirectory() as tempdirpath: # Create temp directory (will close after the script finishes)
            # Write PDF files into temp directory
            for uploaded_file in st.session_state['source_docs']:
                uploaded_file_path = os.path.join(tempdirpath, uploaded_file.name)
                with open(uploaded_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

            # Read and Load PDF files with llamaindex data connector
            st.session_state['loaded_documents'] = SimpleDirectoryReader(input_dir=tempdirpath).load_data()

        # 2. Embedding and create vectorstoreindex
        ## Configure Service Context
        service_context = ServiceContext.from_defaults(
            llm = gemini_pro,
            embed_model = gemini_embed,
            chunk_size = 800
        )
        ## Initialize session state for loaded_documents if it doesn't exist
        if 'loaded_documents' not in st.session_state:
            st.session_state['loaded_documents'] = []

        ## Embedding: convert documents into indexes
        index = VectorStoreIndex.from_documents(st.session_state['loaded_documents'], service_context=service_context)
        ## persist index in data
        index.storage_context.persist()
        st.session_state['chat_engine'] = index.as_chat_engine(chat_mode="context")

# Set up sidebar
with st.sidebar:
    st.session_state['source_docs'] = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    st.text_area(
        "Tools",
        "LLM Model: Gemini-pro \nEmbed Model: Gemini Embedding \nFramework: LlamaIndex", 
        disabled=True
    )
    

# Generate a response from given query
def query_LLM(query):
    response = st.session_state['chat_engine'].chat(query).response
    st.session_state['messages'].append({"query": query, "response": response}) # for history display
    return response


def run_app():
    # Remind user to upload files
    if not st.session_state['source_docs']:
        st.warning(':orange[Please upload files in the left sidebar! Remember to click "Upload Documents".]')

    # Create on_click to trigger process_pdf function
    with st.sidebar:
        st.button('Upload Documents', on_click=process_pdf)

    # Initialize session state for messages if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # Display the chat history
    for message in st.session_state['messages']:
        st.chat_message('User').markdown(message["query"])
        st.chat_message('Model').markdown(message["response"])

    # Send query to query_engine and generate response
    if query := st.chat_input("You:"):
        st.chat_message('User').markdown(query)
        response = query_LLM(query)
        st.chat_message('Model').markdown(response)
        # st.write(st.session_state['chat_engine'].chat_history)
    

if __name__ == '__main__':
    run_app()