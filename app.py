import os
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

json_files = ["faq_data.json"]  # List of JSON file paths
all_docs = []

for json_file in json_files:
    with open(json_file, 'r') as f:
        
        def extract_content(data):
            
            return str(data)
            
        loader = JSONLoader(
            file_path=json_file,
            jq_schema='.', 
            content_key=None,  
            text_content=False,
            json_lines=False
        )
        data = loader.load()
        all_docs.extend(data)

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(all_docs)

# Create embeddings and vector store
gemini_embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=docs, embedding=gemini_embeddings, persist_directory="chroma_db")
vectorstore_disk = Chroma(
    persist_directory="chroma_db", embedding_function=gemini_embeddings
)
retriever = vectorstore_disk.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Retrieve top 10 contexts

llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'),model="llama-3.3-70b-versatile",temperature=0.0,max_retries=2,)

def rag_pipeline(query):
        retrieved_docs = retriever.get_relevant_documents(query)
        contexts = [doc.page_content for doc in retrieved_docs]

    
        system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. You can also answer the questions which are not in given context."
        "Use five sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Prepare the context string for the LLM
        context_str = "\n\n".join(contexts)
        response = rag_chain.invoke({"input": query, "context": context_str})
        answer = response["answer"]
        return answer

# API endpoint for FAQ
@app.route("/")
def hello():
     return "server is live"

@app.route("/faq", methods=["POST"])
def faq():
    try:
        query = request.json.get("query", "")
        if not query:
            return jsonify({"error": "No query provided."}), 400

        # Call the RAG pipeline
        answer = rag_pipeline(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()