import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Document Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .message-user {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .message-assistant {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 1rem 0;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        margin-bottom: 0.5rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .metric-card p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize embeddings
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 AI Document Chat</h1>
    <p>Upload PDFs and have intelligent conversations with your documents</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Key input
    st.markdown("### 🔑 API Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="Enter your Groq API key...",
        help="Get your API key from https://console.groq.com"
    )
    
    # Session management
    st.markdown("### 💬 Chat Session")
    session_id = st.text_input(
        "Session ID",
        value="default_session",
        help="Unique identifier for your chat session"
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        if 'store' in st.session_state and session_id in st.session_state.store:
            del st.session_state.store[session_id]
        if 'messages' in st.session_state:
            st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()
    
    # Statistics
    st.markdown("### 📊 Session Stats")
    if 'store' in st.session_state and session_id in st.session_state.store:
        msg_count = len(st.session_state.store[session_id].messages)
        st.metric("Messages", msg_count)
    else:
        st.metric("Messages", 0)

# Main content area
if groq_api_key:
    # Initialize LLM
    llm = ChatGroq(api_key=groq_api_key, model_name="Gemma2-9b-It")
    
    # Initialize session state
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    # File upload section
    st.markdown("## 📄 Document Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            accept_multiple_files=False,
            help="Upload a PDF document to chat with its content"
        )
    
    with col2:
        if uploaded_file:
            st.markdown(f"""
            <div class="status-success">
                <strong>✅ File Uploaded</strong><br>
                {uploaded_file.name}<br>
                Size: {uploaded_file.size:,} bytes
            </div>
            """, unsafe_allow_html=True)
    
    # Document processing
    if uploaded_file:
        with st.spinner("🔄 Processing document..."):
            # Create temp file
            temp_pdf = f"./temp_{int(time.time())}.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.read())
            
            # Load and process document
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            
            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                chunk_overlap=500
            )
            splits = splitter.split_documents(docs)
            
            # Create vector store
            vectorstore = Chroma.from_documents(
                splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            retriever = vectorstore.as_retriever()
            
            # Clean up temp file
            os.remove(temp_pdf)
        
        # Setup RAG chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise.\n\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        # Chat interface
        st.markdown("## 💬 Chat with your Document")
        
        # Initialize chat messages in session state first
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history from LangChain's message history
        if session_id in st.session_state.store:
            langchain_messages = st.session_state.store[session_id].messages
            if langchain_messages and not st.session_state.messages:
                # Initialize from existing LangChain history if session state is empty
                for i, message in enumerate(langchain_messages):
                    role = "user" if i % 2 == 0 else "assistant"
                    st.session_state.messages.append({"role": role, "content": message.content})
        
        # User input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask a question about your document:",
                placeholder="What is this document about?",
                key="user_question_input"
            )
            submit_button = st.form_submit_button("Send 📤")
        
        if submit_button and user_input:
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("🤔 Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                
                # Add assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        
        # Display chat messages from session state
        if st.session_state.messages:
            st.markdown("### 💬 Conversation")
            
            # Create a container for messages
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.write(message["content"])
    
    else:
        # Upload prompt
        st.markdown("""
        <div class="upload-area">
            <h3>📄 Upload a PDF Document</h3>
            <p>Get started by uploading a PDF file to begin your intelligent conversation</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # API key prompt
    st.markdown("""
    <div class="status-warning">
        <strong>⚠️ API Key Required</strong><br>
        Please enter your Groq API key in the sidebar to get started.
        <br><br>
        <small>Don't have an API key? Get one from <a href="https://console.groq.com" target="_blank">Groq Console</a></small>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("## ✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI-Powered</h3>
            <p>Advanced language models for intelligent document analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>💬 Conversational</h3>
            <p>Natural chat interface with memory of previous interactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📄 PDF Support</h3>
            <p>Transform your PDFs into interactive, conversational knowledge sources</p>
        </div>
        """, unsafe_allow_html=True)