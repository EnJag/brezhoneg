# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration Cache HuggingFace ---
CACHE_DIR = os.environ.get("HF_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "hf")) # "D:/cache"
TRANSFORMERS_CACHE_PATH = os.environ.get("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))
# ... (other cache paths) ...
HF_DATASETS_CACHE_PATH = os.environ.get("HF_DATASETS_CACHE", os.path.join(CACHE_DIR, "datasets"))
HF_METRICS_CACHE_PATH = os.environ.get("HF_METRICS_CACHE", os.path.join(CACHE_DIR, "metrics"))
HF_MODULES_CACHE_PATH = os.environ.get("HF_MODULES_CACHE", os.path.join(CACHE_DIR, "modules"))

def setup_cache():
    # ... (setup_cache remains the same) ...
    print(f"--- Setting up Hugging Face Cache ---")
    print(f"Main Cache Directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(TRANSFORMERS_CACHE_PATH, exist_ok=True)
    os.makedirs(HF_DATASETS_CACHE_PATH, exist_ok=True)
    os.makedirs(HF_METRICS_CACHE_PATH, exist_ok=True)
    os.makedirs(HF_MODULES_CACHE_PATH, exist_ok=True)
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE_PATH
    os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE_PATH
    os.environ["HF_METRICS_CACHE"] = HF_METRICS_CACHE_PATH
    os.environ["HF_MODULES_CACHE"] = HF_MODULES_CACHE_PATH
    print("Cache environment variables set.")
    print("-" * 30)


# --- Translator Model Names ---
MODEL_NAME_NLLB = "facebook/nllb-200-distilled-600M"
MODEL_NAME_HELSINKI = "Helsinki-NLP/opus-mt-fr-en"
MODEL_NAME_TRANSLATOR_SENTENCE_TRANSFORMER = 'distiluse-base-multilingual-cased-v1'

# --- RAG / Utils Configuration ---
ZILLIZ_URI = "https://in03-944f34cea0c1859.serverless.gcp-us-west1.cloud.zilliz.com" # os.environ.get("ZILLIZ_URI")
ZILLIZ_TOKEN = "e9202f865e5ec16f40f0362f08a09902e8e6676b0880ab72dc10e33c353922e37b29f53200e3a9506dcba6386b17f6ac5393fef5" # os.environ.get("ZILLIZ_TOKEN")
RAG_COLLECTION_NAME = "traductions_francais_breton"
MODEL_NAME_RAG_ENCODER = 'paraphrase-multilingual-mpnet-base-v2'

# --- Path Configuration ---
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# REMOVED: PROMPT_FILE_PATH = os.path.join(DATA_DIR, "prompt_50.txt")

# --- Sanity Checks ---
def check_config():
    """Checks if essential configuration (like Zilliz creds) is set."""
    config_ok = True
    if not ZILLIZ_URI:
        print("ERROR: ZILLIZ_URI environment variable not set.")
        config_ok = False
    if not ZILLIZ_TOKEN:
        print("ERROR: ZILLIZ_TOKEN environment variable not set.")
        config_ok = False
    # REMOVED: Check for prompt file
    # if not os.path.exists(PROMPT_FILE_PATH):
    #      print(f"WARNING: Prompt file not found at configured path: {PROMPT_FILE_PATH}")
    return config_ok