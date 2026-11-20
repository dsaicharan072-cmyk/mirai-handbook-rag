"""Configuration settings and environment variable loading."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "mirai_academic_and_hostel_handbook_50p.pdf"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")

# API Keys & Model configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
# Default model: fallback to gemini-3.8-flash if gemini-2.0-flash is retired in modern environment
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Embedding configuration
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "handbook"

# Chunking configuration (~800-1000 chars, ~150-200 chars overlap)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180

# Retrieval configuration
TOP_K = 5
DISTANCE_THRESHOLD = 0.75  # Cosine distance cutoff for relevance
