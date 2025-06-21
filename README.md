# DocuChat AI

DocuChat AI lets you chat with your PDF documents using advanced AI. Upload any PDF and ask questions, get summaries, or extract insights instantly—all through a modern conversational interface powered by Streamlit, LangChain, Chroma, and Groq LLMs.

Effortlessly chat with and extract insights from your PDF documents using Conversational RAG (Retrieval-Augmented Generation) and Streamlit.

## 🚀 Features
- 📄 Upload and analyze PDF documents with ease
- 💬 Conversational AI chat interface with memory
- 🤖 Powered by LangChain, Chroma, and Groq LLMs
- 🔒 Session management and chat history
- 🎨 Modern, responsive UI

## 🛠️ Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/docuchat-ai.git
   cd docuchat-ai
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables:**
   - Create a `.env` file and add your HuggingFace token:
     ```env
     HF_TOKEN=your_huggingface_token
     ```

## ☁️ Deployment (Streamlit Cloud)
1. Push your code to a public GitHub repository (e.g., `docuchat-ai`).
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and click "New app".
3. Select your repo, branch, and set the main file path to `app.py`.
4. Add your environment variables (like `HF_TOKEN`) in the app's settings under "Secrets".
5. Click "Deploy" and share your app's URL!

## 💡 Usage
- Enter your Groq API key in the sidebar.
- Upload a PDF document.
- Start chatting and ask questions about your document!

## 📄 License
This project is licensed under the MIT License. 
