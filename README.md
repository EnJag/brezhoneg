# Breton Translator - Gradio Interface

This project provides a Gradio web interface for translating French text using different Hugging Face models, with optional prompt assistance via RAG (using Zilliz) or random examples from Zilliz. Zilliz examples are taken from https://github.com/Ofis-publik-ar-brezhoneg/breton-french-corpus.

## Features

-   Translate French text using:
    -   `facebook/nllb-200-distilled-600M` (targeting Breton)
    -   `Helsinki-NLP/opus-mt-fr-en` (targeting English)
    -   `meta/llama3.2-1B` (targeting Breton)
-   Optional prompt enhancement:
    -   **RAG**: Retrieve `k` examples **similar** to the input text from a Zilliz Cloud vector database (collection: `traductions_francais_breton`) using the `paraphrase-multilingual-mpnet-base-v2` model to guide the translation.
    -   **Prompt prédéfini**: Retrieve `k` **random** examples from the Zilliz collection (by searching for a random vector) to provide varied context.
-   Configurable cache directory for Hugging Face models.
-   Structured codebase suitable for version control and deployment.
-   Secure handling of Zilliz credentials via environment variables.

## Project Structure
```bash
breton_translator/
├── src/                     # Source code package
│   ├── init.py
│   ├── config.py            # Configuration (cache, models, Zilliz endpoint)
│   ├── translator.py        # Core BretonTraducteur class
│   ├── utils.py             # RAG/Prompt helper functions (Zilliz connection, searches)
│   └── app.py               # Gradio application logic & initialization
├── .env                     # Local environment variables (e.g., Zilliz credentials - DO NOT COMMIT IF PUBLIC)
├── requirements.txt         # Python dependencies
├── .gitignore               # Files ignored by Git
└── README.md                # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd breton_translator
    ```
<!-- 
2.  **Configure Zilliz Credentials:**
    -   Create a file named `.env` in the project root (`breton_translator/.env`).
    -   Add your Zilliz credentials to this file:
        ```dotenv
        # breton_translator/.env
        ZILLIZ_URI="https://YOUR_ZILLIZ_URI_HERE.cloud.zilliz.com"
        ZILLIZ_TOKEN="YOUR_ZILLIZ_TOKEN_HERE"
        ```
    -   **Important:** Ensure `.env` is listed in your `.gitignore` file.
    -   Alternatively, set these as system environment variables. -->

2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
<!-- 
5.  **Configure Cache (Optional):**
    -   The default cache location is `D:/cache`. Change in `src/config.py` or via the `HF_CACHE_DIR` environment variable. -->

4.  **(facultative) Ensure Zilliz Collection Exists:**
    -   Both RAG modes require a Zilliz Cloud collection named `traductions_francais_breton`.
    -   This collection must contain vectors generated using the `paraphrase-multilingual-mpnet-base-v2` model and have fields named `embedding`, `francais`, and `breton`. You need to create and populate this collection separately.


5.  **Running the Application**

```bash
python src/app.py 
```
or
```bash
python -m src.app
```

6.  **Go on your browser**

http://127.0.0.1:7860