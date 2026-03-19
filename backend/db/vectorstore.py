import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from db.crud import get_all_internships
from utils.config import (
    embeddings,
    embedding_type,
    VECTORSTORE_BASE_DIR,
    STUDENT_VECTORSTORE_BASE_DIR,
)

# Keep separate FAISS stores per embedding type to avoid dimension mismatch
VECTORSTORE_PATH = os.path.join(VECTORSTORE_BASE_DIR, f"faiss_index_{embedding_type}")  # Folder, not .pkl
STUDENT_VECTORSTORE_DIR = os.path.join(STUDENT_VECTORSTORE_BASE_DIR, f"student_faiss_{embedding_type}")  # Base folder for per-student FAISS indexes

def build_vectorstore():
    internships = get_all_internships()
    docs = []

    for i in internships:
        text = f"JobID: {i.job_id}\nTitle: {i.title}\nDescription: {i.description}\nSkills: {i.skills_required}\nLocation: {i.location}"
        docs.append(Document(page_content=text, metadata={"job_id": i.job_id}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # If no internship docs exist, avoid building an empty index
    if not split_docs:
        print("⚠️ No internships found to index; skipping vector store build.")
        return None

    # Ensure directory exists
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    # Build and persist the vectorstore
    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(VECTORSTORE_PATH)  # ✅ Save using FAISS
        print("✅ Vector store built/updated")
        return vectorstore
    except Exception as e:
        print(f"⚠️ Failed to build vectorstore: {e}")
        return None

def load_vectorstore():
    """Load an existing FAISS vectorstore or build it if missing.

    Returns:
        FAISS object or None if loading/building fails.
    """
    # Check if the directory exists and contains the required FAISS index files
    index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
    if not os.path.exists(VECTORSTORE_PATH) or not os.path.exists(index_file):
        print("FAISS index not found, building vectorstore...")
        built = build_vectorstore()
        if not built:
            print("⚠️ Vectorstore build failed or produced no index. Returning None.")
            return None

    try:
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)  # ✅ Load using FAISS
    except Exception as e:
        print(f"⚠️ Failed to load vectorstore: {e}")
        return None





def build_or_update_student_vectorstore(student_id: str, resume_text: str):
	"""Create or update a FAISS vectorstore for a specific student.

	- Splits the provided resume text into chunks
	- Embeds and stores them in a per-student FAISS index under STUDENT_VECTORSTORE_DIR/{student_id}
	- Re-usable later via load_student_vectorstore(student_id)
	"""
	# Ensure base directory exists
	os.makedirs(STUDENT_VECTORSTORE_DIR, exist_ok=True)

	student_store_path = os.path.join(STUDENT_VECTORSTORE_DIR, str(student_id))

	# Prepare documents from resume text
	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	chunks = splitter.split_documents([Document(page_content=resume_text, metadata={"student_id": str(student_id)})])

	# If the student's index exists, load and append. Otherwise, create new
	index_file = os.path.join(student_store_path, "index.faiss")
	if os.path.exists(student_store_path) and os.path.exists(index_file):
		try:
			vectorstore = FAISS.load_local(student_store_path, embeddings, allow_dangerous_deserialization=True)
			vectorstore.add_documents(chunks)
		except Exception as e:
			print(f"⚠️ Error loading existing student vectorstore for {student_id}: {e}")
			print("Creating new vectorstore...")
			vectorstore = FAISS.from_documents(chunks, embeddings)
	else:
		vectorstore = FAISS.from_documents(chunks, embeddings)

	# Persist to disk
	vectorstore.save_local(student_store_path)
	print(f"✅ Student vector store built/updated for student_id={student_id}")


def load_student_vectorstore(student_id: str):
	"""Load the FAISS vectorstore for a specific student. Returns None if not found."""
	student_store_path = os.path.join(STUDENT_VECTORSTORE_DIR, str(student_id))
	index_file = os.path.join(student_store_path, "index.faiss")
	if not os.path.exists(student_store_path) or not os.path.exists(index_file):
		return None
	return FAISS.load_local(student_store_path, embeddings, allow_dangerous_deserialization=True)

