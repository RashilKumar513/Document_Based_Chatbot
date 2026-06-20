import streamlit as st
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import uuid
import json
import re

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Document-Based Chatbot",
    page_icon="🎯",
    layout="wide"
)

st.title("Document- Based Chatbot")
st.markdown("Upload PDFs and ask complex, multi-part questions. The AI will strictly use your documents and will explicitly tell you if a piece of information is missing.")

# =====================================
# SESSION STATE & INITIALIZATION
# =====================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# =====================================
# CHROMA DB SETUP
# =====================================
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="chroma_db")

client = get_chroma_client()
collection = client.get_or_create_collection(name="documents")
chat_collection = client.get_or_create_collection(name="chat_history")

# =====================================
# EMBEDDING MODEL
# =====================================
@st.cache_resource
def load_model():
    # Using all-MiniLM-L6-v2 for fast, highly accurate semantic search
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()

# =====================================
# SIDEBAR SETTINGS
# =====================================
st.sidebar.title("⚙️ Settings & Storage")

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    try:
        client.delete_collection("chat_history")
    except:
        pass
    chat_collection = client.get_or_create_collection(name="chat_history")
    st.rerun()

if st.sidebar.button("🗑️ Clear Vector Database"):
    try:
        client.delete_collection("documents")
    except:
        pass
    collection = client.get_or_create_collection(name="documents")
    st.session_state.processed_files.clear()
    st.sidebar.success("Database Cleared Successfully")
    st.rerun()

try:
    st.sidebar.metric(label="Stored Document Chunks", value=collection.count())
except:
    pass

# =====================================
# PDF UPLOAD & INGESTION
# =====================================
st.sidebar.subheader("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    new_files = False
    for uploaded_file in uploaded_files:
        if uploaded_file.name in st.session_state.processed_files:
            continue
            
        try:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                text = ""
                pdf = PdfReader(uploaded_file)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                # Precision Chunking Configurations 
                # Smaller chunks (800) with decent overlap (200) ensure highly specific text blocks
                chunk_size = 800
                overlap = 200
                chunks = []

                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)

                if chunks:
                    embeddings = embedding_model.encode(chunks).tolist()
                    ids = [str(uuid.uuid4()) for _ in chunks]
                    metadatas = [{"source": uploaded_file.name} for _ in chunks]

                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=chunks,
                        metadatas=metadatas
                    )
                
                st.session_state.processed_files.add(uploaded_file.name)
                new_files = True

        except Exception as e:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")
            
    if new_files:
        st.sidebar.success("✅ Files successfully processed into local vector space!")
        st.rerun()

# =====================================
# DISPLAY CHAT HISTORY
# =====================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================
# HIGH-PRECISION RETRIEVAL & GROUNDING SYSTEM
# =====================================
question = st.chat_input("Ask a question from your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    chat_collection.add(
        ids=[str(uuid.uuid4())],
        documents=[f"USER: {question}"]
    )

    try:
        if collection.count() == 0:
            answer = "No documents found. Please upload PDFs first using the sidebar."
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.stop()

        # STEP 1: Intelligent Sub-Query Decomposition via LLM
        # Accurately splits compound inputs into logical standalone parts
        with st.spinner("Analyzing and routing query..."):
            decomposition_prompt = f"""Break down the following user question into a flat JSON list of independent, standalone sub-questions targeting distinct concepts. 
Each item in the list must be a complete sentence that can be searched independently in a database.

User Question: "{question}"

You must respond ONLY with a raw, valid JSON array of strings. Do not include markdown formatting, explanations, or any other text.
Example format: ["sub-question 1", "sub-question 2"]"""
            
            decomp_response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": decomposition_prompt}]
            )
            
            raw_text = decomp_response["message"]["content"].strip()
            
            # Robust JSON extraction using Regex to prevent crashes if the LLM includes markdown
            sub_questions = [question] # Default fallback
            match = re.search(r'\[.*\]', raw_text, re.DOTALL)
            if match:
                try:
                    sub_questions = json.loads(match.group(0))
                    if not isinstance(sub_questions, list):
                        sub_questions = [question]
                except json.JSONDecodeError:
                    sub_questions = [question]

        all_docs = []
        sources = []
        
        # STEP 2: Multi-Query Parallelized Database Retrieval
        for sub_q in sub_questions:
            query_embedding = embedding_model.encode(sub_q).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,  # Fetch top 5 highly relevant chunks per sub-question
                include=["documents", "metadatas"]
            )

            if results and "documents" in results and results["documents"]:
                all_docs.extend(results["documents"][0])
                for meta in results["metadatas"][0]:
                    if meta and "source" in meta:
                        sources.append(meta["source"])

        # Deduplicate retrieved artifacts to save context window space
        unique_docs = list(dict.fromkeys(all_docs))
        sources = list(set(sources))
        
        # Join chunks with clear delimiters
        context = "\n\n---\n\n".join(unique_docs[:15])

        # STEP 3: Rigid Context Isolation System Instructions
        system_rules = (
            "You are a strict, highly precise document QA engine. Your internal weights and historical training knowledge are completely offline. "
            "You can ONLY access and read facts present in the 'Document Context' section below.\n\n"
            "CRITICAL OPERATIONAL LAWS:\n"
            "1. Answer the user's question sequentially by evaluating each part of their request against the provided text context.\n"
            "2. If information regarding a specific question or part is explicitly present in the context, provide a detailed, highly accurate summary.\n"
            "3. If a specific concept, question part, or fact cannot be point-blank found or verified inside the provided context text, do NOT make assumptions or guess. Instead, explicitly state: 'Information regarding [topic] not found in database.'\n"
            "4. Only provide a blanket 'Information not found in database.' response if absolutely zero percent of the user's query can be solved by the text context."
        )

        user_prompt = f"""--- START OF DOCUMENT CONTEXT ---
{context}
--- END OF DOCUMENT CONTEXT ---

User Question: {question}

Answer:"""

        # STEP 4: Native Stream Generation Layer
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            response_stream = ollama.chat(
                model="llama3.2",
                messages=[
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True
            )
            
            for chunk in response_stream:
                content = chunk.get("message", {}).get("content", "")
                full_response += content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)

            # Display source files used during processing
            if sources and "Information not found in database" not in full_response:
                with st.expander("📄 Sources Document(s) Used"):
                    for source in sources:
                        st.write(f"- {source}")

        # Update persistent logs
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        chat_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[f"ASSISTANT: {full_response}"]
        )

    except Exception as e:
        st.error(f"Error executing generation thread: {e}")