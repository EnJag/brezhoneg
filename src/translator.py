# src/translator.py
import os
import ollama
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .config import (
    MODEL_NAME_NLLB, MODEL_NAME_HELSINKI, MODEL_NAME_LLAMA,
    MODEL_NAME_TRANSLATOR_SENTENCE_TRANSFORMER,
    TRANSFORMERS_CACHE_PATH, CACHE_DIR
)
# Import the specific functions needed from utils
from .utils import find_similar_examples_zilliz, get_random_examples_zilliz # <-- Updated import

class BretonTraducteur:
    def __init__(self):
        # ... (init remains the same) ...
        print("--- Initialising Translator Class ---")
        st_model_name = MODEL_NAME_TRANSLATOR_SENTENCE_TRANSFORMER
        print(f"Loading Translator's SentenceTransformer model: {st_model_name}...")
        try:
            self.translator_encoder = SentenceTransformer(st_model_name, cache_folder=CACHE_DIR)
            print("Translator's SentenceTransformer loaded.")
        except Exception as e:
            print(f"WARN: Failed loading Translator's SentenceTransformer '{st_model_name}': {e}")
            self.translator_encoder = None
        print(f"Loading NLLB model: {MODEL_NAME_NLLB}...")
        try:
            self.tokenizer_nllb = AutoTokenizer.from_pretrained(MODEL_NAME_NLLB, cache_dir=TRANSFORMERS_CACHE_PATH)
            self.model_nllb = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_NLLB, cache_dir=TRANSFORMERS_CACHE_PATH)
            print("NLLB model loaded.")
        except Exception as e:
            print(f"ERROR loading NLLB model {MODEL_NAME_NLLB}: {e}")
            raise
        print(f"Loading Helsinki model: {MODEL_NAME_HELSINKI}...")
        try:
            self.tokenizer_helsinki = AutoTokenizer.from_pretrained(MODEL_NAME_HELSINKI, cache_dir=TRANSFORMERS_CACHE_PATH)
            self.model_helsinki = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_HELSINKI, cache_dir=TRANSFORMERS_CACHE_PATH)
            print("Helsinki model loaded.")
        except Exception as e:
            print(f"ERROR loading Helsinki model {MODEL_NAME_HELSINKI}: {e}")
            raise
        print("--- Translator Class Initialisation Complete ---")


    def translate(self, text: str, model_name: str, use_rag: int = 0, use_prompt: int = 0) -> tuple[int, int, str, str]:
        """
        Translate text using the selected model.
        - use_rag > 0: Adds SIMILAR examples found via Zilliz vector search.
        - use_prompt > 0: Adds RANDOM examples retrieved from Zilliz.
        """
        question_to_ask = None # Initialize
        prompt_generated = False # Flag to see if RAG/Prompt logic ran

        # --- RAG/Prompt Logic ---
        if use_rag > 0:
            prompt_generated = True
            # print(f"RAG Similarity requested (k={use_rag}). Calling utils.find_similar_examples_zilliz...")
            # find_similar_examples_zilliz now returns (prompt_string, examples_list)
            question_to_ask, _ = find_similar_examples_zilliz(text, k=use_rag)
            # Use the fully formatted prompt returned by the function

        elif use_prompt > 0:
            prompt_generated = True
            # print(f"Random Examples requested (k={use_prompt}). Calling utils.get_random_examples_zilliz...")
            random_examples = get_random_examples_zilliz(k=use_prompt)

            if not random_examples:
                print("WARN: Failed to retrieve random examples from Zilliz.")
                # Fallback prompt if random retrieval fails
                question_to_ask = "Traduire en breton (exemples aléatoires indisponibles):\n\n" + text
            else:
                # print(f"Constructing prompt with {len(random_examples)} random examples.")
                prompt_header = "Voici quelques paires français-breton aléatoires de la base de données:\n\n"
                example_string = ""
                for ex in random_examples:
                    example_string += f"Français : {ex['french']}\n"
                    example_string += f"Breton : {ex['breton']}\n\n"
                # Combine header, examples, and the text to translate
                question_to_ask = f"{prompt_header}{example_string}Traduire le texte suivant en breton:\n{text}"

        # Default prompt if no RAG/Prompt mode was selected
        if not prompt_generated:
            # print("Using default prompt generation logic (no RAG/Prompt requested)...")
            if model_name == "helsinki":
                 question_to_ask = "Translate to English:\n\n" + text
            else:
                 question_to_ask = "Traduire en breton:\n\n" + text
            # print("Default prompt set.")
        # If RAG/Prompt ran but failed, question_to_ask already holds the fallback.

        # --- Model Selection and Translation ---
        # ... (rest of the translate method remains the same) ...
        selected_tokenizer = None
        selected_model = None
        full_model_name = "Unknown"
        target_lang_code = None

        if model_name == "nllb":
            selected_tokenizer = self.tokenizer_nllb
            selected_model = self.model_nllb
            full_model_name = MODEL_NAME_NLLB
            target_lang_code = "bre_Latn"
            prompt_already_specifies_breton = "Traduire en breton" in question_to_ask # Simple check
        elif model_name == "helsinki":
            selected_tokenizer = self.tokenizer_helsinki
            selected_model = self.model_helsinki
            full_model_name = MODEL_NAME_HELSINKI
        elif model_name == "llama":
            full_model_name = MODEL_NAME_LLAMA
        else:
            error_msg = f"Error: Model '{model_name}' unknown."
            print(error_msg)
            return use_prompt, use_rag, question_to_ask or text, error_msg

        # print(f"Using translator model: {full_model_name}")
        if not isinstance(question_to_ask, str):
             print(f"ERROR: Generated prompt is not a string ({type(question_to_ask)}). Using raw text.")
             question_to_ask = text # Fallback

        # print(f"Final prompt sent to model:\n------\n{question_to_ask}\n------")
        try:
            if model_name == "llama":
                response = ollama.chat(model=full_model_name, messages=[
                {
                    'role': 'user',
                    'content': question_to_ask,
                },
                ])
                translation = response['message']['content']
            else:
                inputs = selected_tokenizer(question_to_ask, return_tensors="pt", truncation=True, max_length=512)
                generated_ids = None
                generation_args = {"max_length": 150}
                if model_name == "nllb" and target_lang_code and not prompt_already_specifies_breton:
                    forced_token_id = selected_tokenizer.lang_code_to_id.get(target_lang_code)
                    if forced_token_id:
                        # print(f"Forcing NLLB target language to: {target_lang_code}")
                        generation_args["forced_bos_token_id"] = forced_token_id
                    else:
                        print(f"Warning: Language code '{target_lang_code}' not found. Using default NLLB generation.")
                # print(f"Generating translation with args: {generation_args}")
                generated_ids = selected_model.generate(**inputs, **generation_args)
                translation = selected_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                # print(f"Raw translation received: {translation}")
        except Exception as e:
            print(f"Error during generation with model {full_model_name}: {e}")
            translation = f"Error during generation: {e}"

        return question_to_ask, translation # use_prompt, use_rag, 