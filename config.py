import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHUNKS_FILE = DATA_DIR / "chunks.json"
INDEX_CACHE_FILE = DATA_DIR / "vector_index.pkl"

PDF_DOCS = [
    {
        "filename": "infosys-ar-25.pdf",
        "title": "Infosys Integrated Annual Report 2024-25",
        "fiscal_year": "2024-25",
        "theme": "AI Your Enterprise",
        "path": BASE_DIR / "infosys-ar-25.pdf"
    },
    {
        "filename": "infosys-ar-26.pdf",
        "title": "Infosys Integrated Annual Report 2025-26",
        "fiscal_year": "2025-26",
        "theme": "AI-First Enterprise & $20B Milestone",
        "path": BASE_DIR / "infosys-ar-26.pdf"
    }
]

# Load environment variables from .env if present
env_file = BASE_DIR / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str and not line_str.startswith("#") and "=" in line_str:
                k, v = line_str.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


MODELS = [
    {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-OSS 120B (High Reasoning)",
        "description": "Deep reasoning model with comprehensive analysis for complex financial and corporate queries."
    },
    {
        "id": "qwen/qwen3.8-27b",
        "name": "Qwen 3.8 27B (Ultra Fast)",
        "description": "Blazing fast inference with precise factual accuracy."
    },
    {
        "id": "openai/gpt-oss-20b",
        "name": "GPT-OSS 20B (Balanced)",
        "description": "Lightweight model optimized for rapid conversational responses."
    }
]

DEFAULT_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
