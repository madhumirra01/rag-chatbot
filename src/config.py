"""All settings in one place. Other files import from here."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DOCS_DIR = ROOT / "data" / "docs"      # put your PDF / TXT / CSV files here
CHROMA_DIR = ROOT / "chroma_db"        # the vector database is saved here
UPLOAD_DIR = ROOT / "data" / "uploads"  # files uploaded through the chat app are saved here

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # small, free, runs on your laptop
COLLECTION = "documents"

CHUNK_TOKENS = 300      # max size of a text chunk, measured in tokens
CHUNK_OVERLAP = 50      # tokens shared between neighbouring chunks
MIN_CHUNK_CHARS = 40    # text chunks shorter than this are dropped (page numbers, stray headings)

MAX_CSV_ROWS = 2000     # CSVs are limited to this many rows until CSV routing is added (phase 6)

TOP_K = 5               # how many chunks to retrieve per question
SIM_THRESHOLD = 0.5     # best similarity must reach this, otherwise we refuse to answer

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = "openai/gpt-oss-120b"