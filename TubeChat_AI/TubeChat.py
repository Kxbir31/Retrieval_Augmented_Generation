from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter



st.set_page_config(page_title="Tube Chat AI", page_icon="🎥", layout="wide")

st.title("🎥 Tube Chat AI")
st.caption("Ask questions about any YouTube video using its transcript.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Video")
    video_id = st.text_input("YouTube Video ID")
    if st.button("📥 Load Video", use_container_width=True):
        if not video_id:
            st.warning("Enter a video ID.")
            st.stop()

        with st.spinner("Loading transcript..."):
            transcript = YouTubeTranscriptApi().fetch(video_id=video_id, languages=["en"])

            document = Document(
                page_content=" ".join(chunk.text for chunk in transcript),
                metadata={"source": "youtube", "video_id": video_id},
            )

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents([document])

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vectorstore = Chroma.from_documents(
                docs,
                embeddings,
                persist_directory=f"youtube_db_{video_id}",
            )
            st.session_state.loaded = True

        st.image(f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg", use_container_width=True)
        st.success("Video loaded successfully!")

if not st.session_state.loaded:
    st.info("Load a video from the sidebar to begin.")
    st.stop()

question = st.chat_input("Ask a question about the video...")

if question:
    st.session_state.messages.append(("user", question))

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)

            prompt = ChatPromptTemplate.from_template("""
You are Tube Chat AI.

Answer ONLY using the transcript context.

If the answer is unavailable, say:
"The requested information is not available in the provided transcript."

Transcript:
{context}

Question:
{question}

Provide:
- A clear explanation
- Bullet points when useful
- Bold important terms
- No unnecessary repetition
""")

            llm = ChatGoogleGenerativeAI(
                model="models/gemini-3-flash-preview"
            )
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})

        st.markdown(answer)
        st.session_state.messages.append(("assistant", answer))

if st.session_state.messages:
    st.divider()
    st.subheader("Conversation")
    for role, msg in st.session_state.messages:
        with st.chat_message(role):
            st.write(msg)

st.divider()
st.caption("Built with ❤️ using Streamlit • LangChain • Gemini • Chroma")