# src/app.py
import gradio as gr
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ... (Initial imports and setup remain the same) ...
try:
    from src import config
    config.setup_cache()
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import configuration from src.config. Check path. Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR during cache setup: {e}")
    sys.exit(1)
try:
    from src import utils
    print("--- Initializing Utils ---")
    if not utils.initialize_utils():
         print("WARNING: Utils initialization failed. RAG/Prompt features might be unavailable.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import utils from src. Check path. Error: {e}")
except Exception as e:
    print(f"CRITICAL ERROR during utils initialization: {e}")
try:
    from src.translator import BretonTraducteur
except ImportError as e:
     print(f"CRITICAL ERROR: Could not import BretonTraducteur. Error: {e}")
     sys.exit(1)
translator_global = None
try:
    print("--- Initializing Global Translator Instance ---")
    translator_global = BretonTraducteur()
    print("--- Global Translator Ready ---")
except Exception as e:
    print(f"CRITICAL ERROR DURING GLOBAL TRANSLATOR INITIALIZATION: {e}")
    sys.exit(1)
# ... (gradio_translate_interface function remains the same) ...
def gradio_translate_interface(text_input, selected_model_short_name, mode, k_value_str):
    """Wrapper function called by the Gradio interface."""
    if translator_global is None:
         return text_input or "", "Translator failed to initialize.", "Please check server logs."
    if (mode in ["RAG", "Prompt prédéfini"] and (utils.ZILLIZ_COLLECTION is None or utils.RAG_ENCODER is None)):
         gr.Warning(f"{mode} assistance is unavailable due to initialization issues. Check logs.")
    if not text_input:
        return "", "", "Please enter text to translate."
    try:
        k = int(k_value_str) if k_value_str is not None else 0
        k = max(0, k)
    except (ValueError, TypeError):
         print(f"Warning: Invalid k value '{k_value_str}'. Using k=0.")
         k = 0
    use_rag_param = 0
    use_prompt_param = 0
    if mode == "RAG" and k > 0:
        use_rag_param = k
    elif mode == "Few-shot learning" and k > 0:
        use_prompt_param = k
    print(f"--- Gradio Request ---")
    print(f"Input Text: '{text_input[:100]}...'")
    print(f"Selected Model: {selected_model_short_name}")
    print(f"Assistance Mode: {mode}, k: {k}")
    print(f"Params for translate(): use_rag={use_rag_param}, use_prompt={use_prompt_param}")
    try:
        question_sent, translation_result = translator_global.translate( # _prompt_ret, _rag_ret, 
            text=text_input,
            model_name=selected_model_short_name,
            use_rag=use_rag_param,
            use_prompt=use_prompt_param
        )
        print(f"Translation Result: '{translation_result[:100]}...'")
        return text_input, question_sent, translation_result
    except Exception as e:
        print(f"ERROR processing translation request in Gradio interface: {e}")
        return text_input, "Error before/during prompt generation", f"An error occurred: {e}"

# --- Gradio Interface Definition ---
desc_nllb = f"`nllb`: Multi-lingual ({config.MODEL_NAME_NLLB}). Targets **Breton**."
desc_helsinki = f"`helsinki`: French to English ({config.MODEL_NAME_HELSINKI}). Targets **English**."
desc_llama = f"`Llama`: Multi-lingual ({config.MODEL_NAME_LLAMA}). Targets **Breton**. "
desc_rag_model = f"RAG uses `{config.MODEL_NAME_RAG_ENCODER}` via Zilliz."

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(f"""
    # Interface de Traduction Multi-modèles

    Entrez du texte en Français et choisissez un modèle de traduction.
    - **{desc_nllb}**
    - **{desc_helsinki}**
    - **{desc_llama}**

    *Assistance Prompt (Optionnel):* {desc_rag_model}
    - **RAG**: Récupère **k** exemples **similaires** au texte d'entrée depuis Zilliz (collection: `{config.RAG_COLLECTION_NAME}`) pour enrichir le prompt (ajuster 'k' avec le curseur ci-dessous). Nécessite une configuration Zilliz correcte.
    - **Prompt prédéfini**: Récupère **k** exemples **aléatoires** depuis Zilliz (via une recherche sur vecteur aléatoire) pour fournir un contexte varié (ajuster 'k' avec le curseur ci-dessous). Nécessite une configuration Zilliz correcte.
    - **Défaut**: Envoie un prompt simple au modèle.
    """)
    # ... (Rest of gr.Blocks definition remains the same) ...
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(label="Texte à traduire (Français)", lines=4, placeholder="Entrez votre texte ici...")
            model_choice = gr.Radio(
                ["nllb", "helsinki", "llama"], label="Choisir le modèle de traduction", value="nllb"
            )
            mode_selection = gr.Radio(["Défaut", "Few-shot learning", "RAG"], label="Mode d'assistance Prompt", value="Défaut")
            k_slider = gr.Slider(minimum=0, maximum=30, value=5, step=1, label="Nombre d'exemples (k)", info="Utilisé si RAG ou Prompt prédéfini est sélectionné et k > 0")
            submit_button = gr.Button("Traduire", variant="primary")
        with gr.Column(scale=3):
            output_original = gr.Textbox(label="1. Texte Français Initial", interactive=False, lines=2)
            output_prompt = gr.Textbox(label="2. Prompt Final Envoyé au Modèle", interactive=False, lines=6)
            output_translation = gr.Textbox(label="3. Résultat de la Traduction", interactive=False, lines=4)
    submit_button.click(
        fn=gradio_translate_interface,
        inputs=[input_text, model_choice, mode_selection, k_slider],
        outputs=[output_original, output_prompt, output_translation]
    )

# --- Application Entry Point ---
# ... (remains the same) ...
# http://127.0.0.1:7860
if __name__ == "__main__":
    if translator_global is None:
         print("CRITICAL: Translator failed to initialize. Aborting launch.")
    elif utils.UTILS_INITIALIZED and (utils.ZILLIZ_COLLECTION is None or utils.RAG_ENCODER is None):
         print("WARNING: RAG/Prompt features will be unavailable due to Zilliz/Encoder issues.")
         print("--- Launching Gradio Interface (RAG/Prompt Disabled) ---")
         iface.launch(server_name="0.0.0.0")
    elif not utils.UTILS_INITIALIZED:
         print("WARNING: Utils failed to initialize properly. RAG/Prompt may not work.")
         print("--- Launching Gradio Interface (Utils Issues) ---")
         iface.launch(server_name="0.0.0.0")
    else:
         print("--- Launching Gradio Interface (All components initialized) ---")
         iface.launch(server_name="0.0.0.0")
    print("--- Gradio Interface Stopped ---")