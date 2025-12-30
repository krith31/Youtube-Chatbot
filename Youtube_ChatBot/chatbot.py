#////////////////USING FAISS STORE///////////////////////

import os
import re
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- ENV ----------------
load_dotenv()

# ---------------- FUNCTIONS ----------------

def extract_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1]
    return None


def load_transcript(video_id: str) -> str:
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched = ytt_api.fetch(
            video_id,
            languages=["hi", "en"]
        )
        return " ".join(snippet.text for snippet in fetched.snippets)
    except TranscriptsDisabled:
        return None
    except Exception:
        return None


def split_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )
    return splitter.create_documents([text])


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


def get_answer(vectorstore, question: str) -> str:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = PromptTemplate(
        template="""
You are a helpful AI assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    return llm.invoke(final_prompt).content


# ---------------- MAIN ENTRY POINT ----------------

def answer_question(youtube_url: str, question: str) -> str:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return "Invalid YouTube URL."

    transcript = load_transcript(video_id)
    if transcript is None:
        return "Transcript not available for this video."

    chunks = split_text(transcript)
    vectorstore = get_vectorstore(chunks)

    return get_answer(vectorstore, question)


#////////////////USING PINECONE STORE///////////////////////

# import os
# import re
# from dotenv import load_dotenv

# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEmbeddings
# from pinecone import Pinecone, ServerlessSpec
# from langchain_pinecone import PineconeVectorStore

# # ---------------- ENV ----------------
# load_dotenv()
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# # ---------------- FUNCTIONS ----------------

# def extract_video_id(url: str) -> str:
#     if "v=" in url:
#         return url.split("v=")[-1].split("&")[0]
#     elif "youtu.be/" in url:
#         return url.split("youtu.be/")[-1]
#     return None


# def load_transcript(video_id: str) -> str:
#     try:
#         ytt_api = YouTubeTranscriptApi()
#         fetched = ytt_api.fetch(
#             video_id,
#             languages=["hi"]
#         )
#         return " ".join(snippet.text for snippet in fetched.snippets)
#     except TranscriptsDisabled:
#         return None
#     except Exception:
#         return None


# def split_text(text: str):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     return splitter.create_documents([text])


# def get_vectorstore(chunks, index_name: str):
#     embeddings = HuggingFaceEmbeddings(
#         model="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     pc = Pinecone(api_key=PINECONE_API_KEY)

#     if index_name not in [i["name"] for i in pc.list_indexes()]:
#         pc.create_index(
#             name=index_name,
#             dimension=384,
#             metric="cosine",
#             spec=ServerlessSpec(
#                 cloud="aws",
#                 region="us-east-1"
#             )
#         )

#     return PineconeVectorStore.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         index_name=index_name
#     )


# def get_answer(vectorstore, question: str) -> str:
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 4,
#             "fetch_k": 20,
#             "lambda_mult": 0.5
#         }
#     )

#     docs = retriever.invoke(question)
#     context = "\n\n".join(doc.page_content for doc in docs)

#     prompt = PromptTemplate(
#         template="""
#         You are a helpful AI assistant.
#         Answer ONLY from the provided transcript context.
#         If the context is insufficient, say "I don't know".

#         Context:
#         {context}

#         Question:
#         {question}
#         """,
#         input_variables=["context", "question"]
#     )

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0.2
#     )

#     final_prompt = prompt.invoke({
#         "context": context,
#         "question": question
#     })

#     return llm.invoke(final_prompt).content


# # ---------------- MAIN ENTRY POINT ----------------

# def answer_question(youtube_url: str, question: str) -> str:
#     video_id = extract_video_id(youtube_url)
#     if not video_id:
#         return "Invalid YouTube URL."

#     transcript = load_transcript(video_id)
#     if transcript is None:
#         return "Transcript not available for this video."

#     chunks = split_text(transcript)
#     safe_video_id = re.sub(r'[^a-z0-9-]', '-', video_id.lower())
#     index_name = f"yt-{safe_video_id}"

#     vectorstore = get_vectorstore(chunks, index_name)
#     return get_answer(vectorstore, question)



