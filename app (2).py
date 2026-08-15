import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="Domain RAG Chatbot")

@st.cache_resource
def load_ai_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    #LLM: Free, local, open-source model (No API keys needed)
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)
    llm = HuggingFacePipeline(pipeline=pipe)

    return embeddings, llm

embeddings, llm = load_ai_models()


st.title("Domain-Specific RAG Chatbot")
st.write("Upload a PDF and ask questions. The AI will strictly answer from the document.")

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
    process_btn = st.button("Process Documents")

    if st.button("Clear Chat / Reset"):
        if os.path.exists("faiss_index"):
            import shutil
            shutil.rmtree("faiss_index")
        st.success("Cleared! Please upload new documents.")


if process_btn and uploaded_files:
    with st.spinner("Processing Documents..."):
        os.makedirs("temp_docs", exist_ok=True)
        all_splits = []

        for file in uploaded_files:
            file_path = os.path.join("temp_docs", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            #Extract text
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            #Chunking: 800 chars with 120 overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=120
            )
            splits = text_splitter.split_documents(docs)
            all_splits.extend(splits)

        if all_splits:
            #Store in FAISS
            vector_store = FAISS.from_documents(all_splits, embeddings)
            vector_store.save_local("faiss_index")
            st.success("Documents processed and vector database created!")
        else:
            st.error("No text found in PDFs.")


if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template = """You are a document question-answering assistant.
Answer only from the supplied context. If the answer is not available, say:
"I could not find this information in the uploaded documents." Do not invent facts.
Mention the source document and page number when available.

Context: {context}
Question: {question}
Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    st.divider()
    st.header("2. Ask Questions")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        with st.spinner("Searching for answer..."):
            result = qa_chain.invoke({"query": user_question})
            answer = result["result"]
            source_docs = result["source_documents"]

            st.write("**Answer:**")
            st.info(answer)

            st.write("**Sources used:**")
            for i, doc in enumerate(source_docs):
                source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page_num = doc.metadata.get('page', 0) + 1
                st.caption(f"Source {i+1}: {source_file}, Page {page_num}")
