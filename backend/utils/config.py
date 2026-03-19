import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from google.api_core.exceptions import ResourceExhausted
import logging

# Load environment variables from .env if present
load_dotenv()

# API Keys - Load from environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN", "")

# Database and deployment-related settings
# Prefer env-provided DATABASE_URL (e.g., postgres) with a local SQLite fallback for dev
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./internships.db")

# CORS origins (comma-separated). Example: "http://localhost:5173,https://your-frontend.vercel.app"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if o.strip()]

# Vector store base directories (use volumes in production)
VECTORSTORE_BASE_DIR = os.getenv("VECTORSTORE_DIR", ".")
STUDENT_VECTORSTORE_BASE_DIR = os.getenv("STUDENT_VECTORSTORE_DIR", ".")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_cached_embeddings = None
_cached_provider = None

def get_embeddings_with_fallback():
    """Get embeddings with automatic fallback to Hugging Face API if Gemini fails."""
    global _cached_embeddings, _cached_provider
    if _cached_embeddings is not None:
        return _cached_embeddings, _cached_provider

    # Try Gemini first only if API key is present
    google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_api_key:
        try:
            primary_embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=google_api_key,
            )
            # Smoke test
            _ = primary_embeddings.embed_query("test")
            logger.info("✅ Using Gemini embeddings")
            _cached_embeddings, _cached_provider = primary_embeddings, "gemini"
            return _cached_embeddings, _cached_provider
        except ResourceExhausted:
            logger.warning("⚠️ Gemini quota exceeded, switching to Hugging Face API embeddings")
        except Exception as e:
            logger.warning(f"⚠️ Gemini error, using Hugging Face API fallback: {e}")

    # Fallback: Hugging Face API
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACE_API_TOKEN", "")
    fallback_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    _cached_embeddings, _cached_provider = fallback_embeddings, "huggingface"
    return _cached_embeddings, _cached_provider

# Default embeddings (will be set dynamically)
embeddings, embedding_type = get_embeddings_with_fallback()
