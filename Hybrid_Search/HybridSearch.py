from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

documents = [Document(
    page_content = "Employ 1 works in AI department"
),
    Document(
    page_content = "Employ 2works in HR department , lives in delhi "
),
    Document(
        page_content = "EMP003 works in Health department"
)]
# for doc in documents:
#     print(doc.page_content)

embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
vector_db = FAISS.from_documents(
    documents,
    embeddings
)

vector_retriever = vector_db.as_retriever()

query = "where EMP002 lives "

result = vector_retriever.invoke(query)
print(result)
