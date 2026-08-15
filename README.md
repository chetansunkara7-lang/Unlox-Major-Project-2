Domain-Specific RAG Chatbot
This project is a Retrieval-Augmented Generation (RAG) chatbot built to answer questions from uploaded PDF documents without hallucinating facts.
Features
Extracts text and page data from PDFs
Chunks text
Creates semantic embeddings. Stores them in a local FAISS database.
Uses a local, offline LLM to generate answers.
Enforces strict guardrails to refuse to answer if the information is not in the text.
How to Run
Install dependencies: pip install -r requirements.txt
Run the application: streamlit run app.py
Upload your PDFs, click "Process Documents", and ask a question.
