# src/utils.py
import os
import numpy as np
import random # Keep for potential future use, though not directly needed for random vector
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility, MilvusException

# Import config values from the config module
from . import config

# --- Global Variables (Initialized by initialize_utils) ---
RAG_ENCODER = None
ZILLIZ_COLLECTION = None
UTILS_INITIALIZED = False
RAG_ENCODER_DIMENSION = None # Store dimension after loading model

# --- Initialization Function ---
def initialize_utils():
    """
    Connects to Zilliz Cloud and loads the Sentence Transformer model for RAG.
    Should be called once at application startup AFTER config is loaded.
    """
    global RAG_ENCODER, ZILLIZ_COLLECTION, UTILS_INITIALIZED, RAG_ENCODER_DIMENSION
    if UTILS_INITIALIZED:
        print("Utils: Already initialized.")
        return True

    print("--- Initializing Utilities (Zilliz Connection & RAG Encoder) ---")

    if not config.check_config(): # check_config needs update to remove file check
         print("ERROR: Utils initialization failed due to missing configuration.")
         return False

    try:
        print(f"Utils: Loading RAG encoder model: {config.MODEL_NAME_RAG_ENCODER}...")
        RAG_ENCODER = SentenceTransformer(config.MODEL_NAME_RAG_ENCODER, cache_folder=config.TRANSFORMERS_CACHE_PATH)
        # Get and store the model's embedding dimension
        RAG_ENCODER_DIMENSION = RAG_ENCODER.get_sentence_embedding_dimension()
        print(f"Utils: RAG encoder model loaded (Dimension: {RAG_ENCODER_DIMENSION}).")
    except Exception as e:
        print(f"ERROR: Failed to load RAG encoder model '{config.MODEL_NAME_RAG_ENCODER}': {e}")
        RAG_ENCODER = None
        RAG_ENCODER_DIMENSION = None

    try:
        if not connections.has_connection("default"):
            print(f"Utils: Connecting to Zilliz Cloud: {config.ZILLIZ_URI}")
            connections.connect("default", uri=config.ZILLIZ_URI, token=config.ZILLIZ_TOKEN)
            print("Utils: Connected to Zilliz.")
        else:
             print("Utils: Already have a Zilliz connection.")

        if utility.has_collection(config.RAG_COLLECTION_NAME):
            print(f"Utils: Accessing Zilliz collection '{config.RAG_COLLECTION_NAME}'...")
            collection = Collection(config.RAG_COLLECTION_NAME)
            ZILLIZ_COLLECTION = collection
            print(f"Utils: Loading collection '{config.RAG_COLLECTION_NAME}' for search...")
            # load_state = utility.get_loading_progress(config.RAG_COLLECTION_NAME)
            # if load_state.get('loading_progress', 0) < 100:
            # print("Utils: Collection not fully loaded, attempting load...")
            collection.load()
            utility.wait_for_loading_complete(config.RAG_COLLECTION_NAME, timeout=60)
            print(f"Utils: Collection '{config.RAG_COLLECTION_NAME}' loading complete.")
            # else:
            #      print(f"Utils: Collection '{config.RAG_COLLECTION_NAME}' is already loaded.")
        else:
            print(f"ERROR: Zilliz collection '{config.RAG_COLLECTION_NAME}' not found.")
            ZILLIZ_COLLECTION = None

    except MilvusException as me:
        print(f"ERROR: Milvus/Zilliz specific error during connection/loading: {me}")
        ZILLIZ_COLLECTION = None
    except Exception as e:
        print(f"ERROR: General error during Zilliz connection/loading: {e}")
        ZILLIZ_COLLECTION = None

    print("-" * 30)
    UTILS_INITIALIZED = True
    # Initialization considered successful if Zilliz collection and encoder are ready
    return (RAG_ENCODER is not None) and (ZILLIZ_COLLECTION is not None)


# --- RAG Search Function (Similarity) ---
def find_similar_examples_zilliz(text: str, k: int) -> tuple[str, list[dict]]:
    """
    Finds k similar examples using the initialized Zilliz connection and RAG encoder.
    Formats a prompt string with the examples.

    Args:
        text: The input text to search for.
        k: The number of similar examples to retrieve.

    Returns:
        A tuple containing:
        - A formatted prompt string including examples, or a fallback prompt.
        - A list of the retrieved example dictionaries [{'french':..., 'breton':...}].
    """
    fallback_prompt = "Traduire en breton (RAG indisponible):\n\n" + text
    if not UTILS_INITIALIZED or ZILLIZ_COLLECTION is None or RAG_ENCODER is None:
        print("WARN: find_similar_examples_zilliz cannot run: Not initialized, Zilliz disconnected or RAG model not loaded.")
        return fallback_prompt, []

    print(f"--- RAG Similarity: Finding {k} similar examples for '{text[:50]}...' ---")
    try:
        query_embedding = RAG_ENCODER.encode([text])[0]
        search_params = {"metric_type": "COSINE", "params": {"level": 2}}
        print(f"Utils: Searching Zilliz collection '{config.RAG_COLLECTION_NAME}' for similar...")
        results = ZILLIZ_COLLECTION.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["francais", "breton"]
        )
        print(f"Utils: Zilliz similarity search completed.")
    except (MilvusException, Exception) as e:
        print(f"ERROR: Error during Zilliz similarity search: {e}")
        return fallback_prompt, []

    similar_examples = []
    if results and len(results[0]) > 0:
        print(f"Utils: Found {len(results[0])} similar results.")
        for hit in results[0]:
            french_text = hit.entity.get('francais', '[français manquant]') if hit.entity else '[entité manquante]'
            breton_text = hit.entity.get('breton', '[breton manquant]') if hit.entity else '[entité manquante]'
            similar_examples.append({'french': french_text, 'breton': breton_text, 'distance': hit.distance})
    else:
        print("Utils: No similar examples found in Zilliz.")

    if not similar_examples:
        prompt_header = "Traduire en breton (aucun exemple similaire trouvé):\n\n"
    else:
        prompt_header = "Traduire en breton en utilisant ces exemples :\n\n"

    example_string = ""
    for ex in similar_examples:
        example_string += f"Français : {ex['french']}\n"
        example_string += f"Breton : {ex['breton']}\n\n"

    final_prompt = f"{prompt_header}{example_string}Texte à traduire en breton:\n{text}"
    print(f"--- RAG Similarity: Generated prompt with {len(similar_examples)} examples. ---")
    return final_prompt, similar_examples

# --- Function to get RANDOM examples from Zilliz ---
def get_random_examples_zilliz(k: int) -> list[dict]:
    """
    Retrieves k 'random' examples from Zilliz by searching for a random vector.

    Args:
        k: The number of random examples to retrieve.

    Returns:
        A list of example dictionaries [{'french': ..., 'breton': ...}],
        or an empty list if retrieval fails.
    """
    if not UTILS_INITIALIZED or ZILLIZ_COLLECTION is None or RAG_ENCODER is None or RAG_ENCODER_DIMENSION is None:
        print("WARN: get_random_examples_zilliz cannot run: Not initialized, Zilliz disconnected or RAG model/dimension not loaded.")
        return []

    if k <= 0:
        return []

    print(f"--- RAG Random: Getting {k} random examples from Zilliz ---")
    try:
        # 1. Generate a random vector
        random_vector = np.random.rand(RAG_ENCODER_DIMENSION).astype(np.float32)
        # Normalize the vector for COSINE search (optional but good practice)
        norm = np.linalg.norm(random_vector)
        if norm > 0:
            random_vector /= norm

        # 2. Prepare search parameters
        search_params = {
            "metric_type": "COSINE", # Or "L2"
            "params": {"level": 2}   # Adjust as needed
        }

        # 3. Perform search with the random vector
        print(f"Utils: Searching Zilliz collection '{config.RAG_COLLECTION_NAME}' with random vector...")
        results = ZILLIZ_COLLECTION.search(
            data=[random_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["francais", "breton"]
        )
        print(f"Utils: Zilliz random search completed.")

    except (MilvusException, Exception) as e:
        print(f"ERROR: Error during Zilliz random search: {e}")
        return []

    # 4. Extract results
    random_examples = []
    if results and len(results[0]) > 0:
        print(f"Utils: Found {len(results[0])} random results.")
        for hit in results[0]:
            french_text = hit.entity.get('francais', '[français manquant]') if hit.entity else '[entité manquante]'
            breton_text = hit.entity.get('breton', '[breton manquant]') if hit.entity else '[entité manquante]'
            random_examples.append({'french': french_text, 'breton': breton_text})
            # We don't usually care about the distance for random examples
    else:
        print("Utils: No random examples found (search returned empty).")

    return random_examples