import streamlit as st
import tempfile
import os
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.preview.generative_models as generative_models
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub

# Initialize Vertex AI
vertexai.init(project="", location="us-central1")

@st.cache_resource
def initialize_rag(model_choice):
    # Initialize embedding model
    embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@latest")

    # Initialize Chroma vector store
    persist_directory = "chroma_db"
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Initialize LLM based on user choice
    if model_choice == "Gemini":
        llm = VertexAI(
            model_name="gemini-1.5-pro-001",
            max_output_tokens=8192,
            temperature=0,
            top_p=0.95,
            verbose=True,
        )
    elif model_choice in ["Mistral", "OpenLLAMA"]:
        api_token = "<enter token>"
        if not api_token:
            st.error("Hugging Face API token is required for Mistral and OpenLLAMA models.")
            st.stop()
        
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" if model_choice == "Mistral" else "openlm-research/open_llama_3b"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.5, "max_length": 512},
            huggingfacehub_api_token=api_token
        )
    else:
        raise ValueError(f"Unsupported model choice: {model_choice}")

    # Create RetrievalQA chain
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )

    # Custom prompt template
    custom_prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}

    Please provide a comprehensive, detailed, and very long response:
    """

    PROMPT = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    qa_chain.combine_documents_chain.llm_chain.prompt = PROMPT

    return vectorstore, qa_chain

def process_pdf_document(file, vectorstore):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Add to vectorstore
        vectorstore.add_documents(texts)
        vectorstore.persist()
        
        return f"Processed and added {len(texts)} text chunks to the vectorstore."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def generate_answer(qa_chain, query):
    try:
        result = qa_chain({"query": query})
        answer = result['result']
        source_documents = result['source_documents']
        
        return answer, [doc.metadata for doc in source_documents]
    except Exception as e:
        return f"Error during generation: {str(e)}", []

def main():
    st.title("PDF Question Answering with RAG")

    # Model selection
    model_choice = st.selectbox("Choose a language model:", ["Gemini", "Mistral"])

    # Initialize RAG components
    vectorstore, qa_chain = initialize_rag(model_choice)

    # Mode selection
    mode = st.radio("Select mode:", ("Single Document", "Multiple Documents"))

    if mode == "Single Document":
        # Single document mode
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            result = process_pdf_document(uploaded_file, vectorstore)
            st.success(result)

            # Query input for single document
            query = st.text_input("Enter your question about the uploaded document:")
            if st.button("Ask"):
                if query:
                    with st.spinner("Generating answer..."):
                        answer, sources = generate_answer(qa_chain, query)
                        st.write("Answer:")
                        st.write(answer)
                        if sources:
                            st.write("Sources:")
                            for source in sources:
                                st.write(source)
                else:
                    st.warning("Please enter a question.")

    else:
        # Multiple documents mode
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                result = process_pdf_document(file, vectorstore)
                st.success(result)

            # Query input for multiple documents
            query = st.text_input("Enter your question about the uploaded documents:")
            if st.button("Ask"):
                if query:
                    with st.spinner("Generating answer..."):
                        answer, sources = generate_answer(qa_chain, query)
                        st.write("Answer:")
                        st.write(answer)
                        if sources:
                            st.write("Sources:")
                            for source in sources:
                                st.write(source)
                else:
                    st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
