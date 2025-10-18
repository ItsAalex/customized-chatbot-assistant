from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import streamlit as st
from typing import List, Dict
import tempfile
import os
import pandas as pd


# Initialize the Ollama model
@st.cache_resource  # saves the model in (cache) memory for the session
def init_model():
    return OllamaLLM(model='llama3.2')


# Initialize embeddings (used to convert text to vectors)
# This is cached to avoid reloading the model every time
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Initialize a basic conversation chain (without document context)
@st.cache_resource
def init_basic_chain(_model):
    template = """
    You are florinpay's helpful assistant. Answer the following question:
    Question: {question}
    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt | _model


# Process documents and create vector store
def process_document(file, file_type):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Initialize text content
        text_content = ""

        # Process based on file type
        if file_type in ['xlsx', 'xls']:
            # Load Excel file
            df = pd.read_excel(tmp_path)
            text_content = df.to_string()

        elif file_type == 'pdf':
            # Load PDF file
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            # Combine all pages into one text
            text_content = "\n".join([page.page_content for page in pages])

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        chunks = text_splitter.split_text(text_content)

        # Create vector store
        embeddings = init_embeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Clean up temp file
        os.unlink(tmp_path)

        return vectorstore
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e


# Function to get model response with or without document context
def get_model_response(prompt: str, chain, is_doc_chain=False) -> str:
    try:
        if is_doc_chain:
            return chain({"question": prompt, "chat_history": []})["answer"]
        else:
            return chain.invoke({"question": prompt})
    except Exception as e:
        return f"Error: Unable to get response from model. {str(e)}"


# APPLICATION

# App Title
st.title("FlorinPay's Personal Assistant")

# Initialize model
model = init_model()

# Initialize basic chain for when no document is uploaded
basic_chain = init_basic_chain(model)

# File uploader in sidebar with multiple file types
st.sidebar.title("Document Upload (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload Document for Context", type=['xlsx', 'xls', 'pdf'])

# Initialize or update vector store when file is uploaded
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    with st.spinner(f"Processing {file_type.upper()} file..."):
        try:
            vectorstore = process_document(uploaded_file, file_type)
            st.session_state.doc_chain = ConversationalRetrievalChain.from_llm(
                llm=model,
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            st.session_state.using_document = True
            st.sidebar.success(f"File '{uploaded_file.name}' processed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
else:
    st.session_state.using_document = False

# Initialize session state for messages if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("How can I help you?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            if st.session_state.using_document and 'doc_chain' in st.session_state:
                response = get_model_response(prompt, st.session_state.doc_chain, is_doc_chain=True)
                # Add a note that response is based on document
                response += "\n\n_Response based on the uploaded document._"
            else:
                response = get_model_response(prompt, basic_chain, is_doc_chain=False)
        message_placeholder.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add buttons to sidebar
with st.sidebar:
    # Clear document button
    if st.session_state.get('using_document', False) and st.button("Clear Document"):
        if 'doc_chain' in st.session_state:
            del st.session_state.doc_chain
        st.session_state.using_document = False
        st.success("Document removed from context")
        st.rerun()

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()