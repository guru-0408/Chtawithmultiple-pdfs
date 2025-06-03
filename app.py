import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai # type: ignore
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re
import fitz  # PyMuPDF (better for extracting text with spaces)
import io
# we have using this for spacing between each word or sentence............
def fix_spacing(text):
    """
    Fix spacing issues where words are merged together.
    """
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # Add space between lowercase and uppercase words
    text = re.sub(r"(?<=[a-zA-Z])(?=[0-9])", " ", text)  # Add space before numbers
    text = re.sub(r"(?<=[.,])(?=[A-Za-z])", " ", text)  # Add space after punctuation if missing
    return " ".join(text.split())  # Normalize spaces

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text_chunks = []
    metadata = []

    for pdf in pdf_docs:
        pdf_stream = io.BytesIO(pdf.read())  # Read uploaded PDF as a byte stream
        doc = fitz.open(stream=pdf_stream, filetype="pdf")  # Open using PyMuPDF
        for i, page in enumerate(doc):
            extracted_text = page.get_text("text")  # Extract text properly
            
            if extracted_text:
                clean_text = fix_spacing(extracted_text)  # Apply spacing fix
                text_chunks.append(clean_text)
                metadata.append({"page": i + 1})  # Store page numbers
    return  text_chunks, metadata
    

def get_vector_store(text_chunks, metadata):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search_with_score(user_question)  # Retrieve results with scores

    chain = get_conversational_chain()
    response = chain({"input_documents": [doc[0] for doc in docs], "question": user_question}, return_only_outputs=True)

    for doc, score in docs:
        page_number = doc.metadata.get("page", "Unknown")
        st.write(f"Page {page_number}: {doc.page_content}")

    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                
                text_chunks, metadata = get_pdf_text(pdf_docs)
                get_vector_store(text_chunks, metadata)
                st.success("Done")



if __name__ == "__main__":
    main()
