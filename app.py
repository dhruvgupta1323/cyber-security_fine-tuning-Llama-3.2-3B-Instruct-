# ============================================================
# 🛡️ CYBERSHIELD
# Llama 3.2 3B + QLoRA + REAL-TIME QUESTION SUGGESTIONS
# ============================================================

# ============================================================
# 1. INSTALL
# ============================================================

!pip install -q sentence-transformers scikit-learn


# ============================================================
# 2. IMPORTS
# ============================================================

import os
import re
import json
import numpy as np
import torch
import gradio as gr

from unsloth import FastLanguageModel
from peft import PeftModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 3. GPU
# ============================================================

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available.")

print("GPU:", torch.cuda.get_device_name(0))

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2
    ),
    "GB"
)


# ============================================================
# 4. PATHS
# ============================================================

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

ADAPTER_PATH = (
    "/kaggle/input/datasets/"
    "dhruvgutpa/fine-tuning1001"
)

RECOMMENDER_PATH = (
    "/kaggle/input/models/"
    "dhruvgutpa/cyber-ques/"
    "transformers/default/1/"
    "CyberShield_Question_Recommender"
)

MAX_SEQ_LENGTH = 2048


# ============================================================
# 5. LOAD BASE MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING LLAMA 3.2 3B")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

print("✅ Base model loaded")


# ============================================================
# 6. LOAD LORA
# ============================================================

print("Loading CyberShield LoRA...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

print("✅ CyberShield LoRA loaded")


# ============================================================
# 7. INFERENCE MODE
# ============================================================

FastLanguageModel.for_inference(model)

print("✅ Inference mode enabled")


# ============================================================
# 8. FIX GENERATION CONFIG
# ============================================================
#
# Your model currently contains:
#
# max_length = 131072
#
# We remove it so Transformers doesn't complain when
# max_new_tokens is used.
# ============================================================

if hasattr(model, "generation_config"):

    model.generation_config.max_length = None

    model.generation_config.max_new_tokens = 256

    print(
        "✅ Generation config fixed"
    )

print()


# ============================================================
# 9. LOAD SENTENCE TRANSFORMER
# ============================================================

print("=" * 60)
print("LOADING QUESTION RECOMMENDER")
print("=" * 60)

sentence_model_path = os.path.join(
    RECOMMENDER_PATH,
    "sentence-transformer"
)

question_model = SentenceTransformer(
    sentence_model_path
)

print("✅ Sentence Transformer loaded")


# ============================================================
# 10. LOAD EMBEDDINGS
# ============================================================

question_embeddings = np.load(
    os.path.join(
        RECOMMENDER_PATH,
        "question_embeddings.npy"
    )
)

print(
    "Embeddings:",
    question_embeddings.shape
)


# ============================================================
# 11. LOAD QUESTIONS
# ============================================================

with open(
    os.path.join(
        RECOMMENDER_PATH,
        "questions.json"
    ),
    "r",
    encoding="utf-8"
) as f:

    cybersecurity_questions = json.load(f)


print(
    "Questions:",
    len(cybersecurity_questions)
)


# ============================================================
# 12. VERIFY
# ============================================================

assert len(cybersecurity_questions) == len(
    question_embeddings
)

print("✅ Questions and embeddings match")


# ============================================================
# 13. CLEAN RESPONSE
# ============================================================

def clean_response(text):

    if not text:
        return ""

    text = text.replace(
        "\\n",
        "\n"
    )

    text = text.replace(
        "\\t",
        "\t"
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# 14. GENERATE RESPONSE
# ============================================================

def generate_response(
    message,
    history
):

    messages = [

        {
            "role": "system",
            "content": (
                "You are CyberShield, a professional "
                "defensive cybersecurity assistant.\n\n"

                "Answer the user's question directly "
                "and clearly.\n"

                "Keep simple questions concise.\n"

                "For simple definitions, answer in "
                "2 to 5 sentences.\n"

                "For technical questions, use headings "
                "and bullet points.\n"

                "Focus on ethical and defensive "
                "cybersecurity guidance."
            )
        }

    ]


    # --------------------------------------------------------
    # OLD GRADIO HISTORY FORMAT
    # --------------------------------------------------------

    if history:

        for item in history[-6:]:

            if not isinstance(
                item,
                (list, tuple)
            ):
                continue

            if len(item) != 2:
                continue

            user_message = item[0]

            assistant_message = item[1]


            if user_message:

                messages.append({
                    "role": "user",
                    "content": str(
                        user_message
                    )
                })


            if assistant_message:

                messages.append({
                    "role": "assistant",
                    "content": str(
                        assistant_message
                    )
                })


    # --------------------------------------------------------
    # CURRENT QUESTION
    # --------------------------------------------------------

    messages.append({
        "role": "user",
        "content": message
    })


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer.apply_chat_template(

        messages,

        tokenize=True,

        add_generation_prompt=True,

        return_tensors="pt"

    ).to(model.device)


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(

            input_ids=inputs,

            max_new_tokens=256,

            temperature=0.2,

            top_p=0.85,

            repetition_penalty=1.1,

            do_sample=True,

            use_cache=True,

            # Explicitly override generation config
            max_length=None
        )


    # --------------------------------------------------------
    # RESPONSE ONLY
    # --------------------------------------------------------

    response_tokens = outputs[0][
        inputs.shape[-1]:
    ]


    response = tokenizer.decode(

        response_tokens,

        skip_special_tokens=True

    )


    return clean_response(
        response
    )


# ============================================================
# 15. REAL-TIME SUGGESTIONS
# ============================================================

def realtime_suggestions(
    current_question
):

    if not current_question:

        return gr.update(
            choices=[],
            value=None
        )


    current_question = (
        current_question.strip()
    )


    if len(current_question) < 5:

        return gr.update(
            choices=[],
            value=None
        )


    # --------------------------------------------------------
    # EMBED CURRENT TEXT
    # --------------------------------------------------------

    query_embedding = question_model.encode(

        [current_question],

        normalize_embeddings=True,

        show_progress_bar=False

    )


    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    scores = cosine_similarity(

        query_embedding,

        question_embeddings

    )[0]


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    top_indices = np.argsort(
        scores
    )[::-1]


    suggestions = []


    current_clean = (
        current_question
        .lower()
        .strip()
    )


    # --------------------------------------------------------
    # TOP 4
    # --------------------------------------------------------

    for idx in top_indices:

        question = cybersecurity_questions[idx]

        if (
            question
            .lower()
            .strip()
            ==
            current_clean
        ):
            continue


        if scores[idx] < 0.25:
            continue


        suggestions.append(
            question
        )


        if len(suggestions) >= 4:
            break


    return gr.update(

        choices=suggestions,

        value=None

    )


# ============================================================
# 16. SELECT SUGGESTION
# ============================================================

def select_suggestion(
    question
):

    if question is None:
        return ""

    return question


# ============================================================
# 17. CHAT RESPONSE
# ============================================================

def respond(
    message,
    history
):

    if not message:
        return "", history


    message = message.strip()


    if not message:
        return "", history


    try:

        response = generate_response(
            message,
            history
        )


    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()

        response = (
            "⚠️ GPU memory is full.\n\n"
            "Please ask a shorter question."
        )


    except Exception as e:

        response = (
            "⚠️ Generation error:\n\n"
            + str(e)
        )


    # ========================================================
    # OLD GRADIO FORMAT
    # ========================================================

    history = history + [
        [
            message,
            response
        ]
    ]


    return "", history


# ============================================================
# 18. CLEAR
# ============================================================

def clear_chat():

    return (
        [],
        gr.update(
            choices=[],
            value=None
        )
    )


# ============================================================
# 19. GRADIO UI
# ============================================================

with gr.Blocks(
    title="CyberShield",
    theme=gr.themes.Soft()
) as demo:


    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        """
        # 🛡️ CyberShield

        ### Defensive Cybersecurity AI Assistant

        **Llama 3.2 3B Instruct • QLoRA • 4-bit NF4 • Unsloth**

        Ask questions about cybersecurity, threat detection,
        vulnerabilities, incident response, secure coding,
        network security, and defensive security practices.
        """
    )


    # ========================================================
    # MAIN
    # ========================================================

    with gr.Row():


        # ----------------------------------------------------
        # CHAT
        # ----------------------------------------------------

        with gr.Column(
            scale=7
        ):

            # IMPORTANT:
            # NO type=
            # NO show_copy_button=

            chatbot = gr.Chatbot(
                height=600,
                label="CyberShield"
            )


            msg = gr.Textbox(
                placeholder=(
                    "Start typing a cybersecurity "
                    "question..."
                ),
                label="Your Question",
                lines=2
            )


            with gr.Row():

                submit = gr.Button(
                    "🛡️ Ask CyberShield",
                    variant="primary"
                )

                clear = gr.Button(
                    "🗑️ Clear"
                )


        # ----------------------------------------------------
        # SUGGESTIONS
        # ----------------------------------------------------

        with gr.Column(
            scale=3
        ):

            gr.Markdown(
                """
                ## 💡 Real-Time Suggestions

                Start typing your cybersecurity question
                to see related questions.
                """
            )


            suggestions_box = gr.Radio(
                choices=[],
                label="Suggested Questions",
                interactive=True
            )


            gr.Markdown(
                f"""
                ### 🧠 Recommendation Engine

                **Model:** `all-MiniLM-L6-v2`

                **Questions:** `{len(cybersecurity_questions):,}`

                **Embeddings:** `384`

                **Similarity:** Cosine Similarity
                """
            )


    # ========================================================
    # INFO
    # ========================================================

    gr.Markdown(
        """
        ---

        ## ⚙️ CyberShield Architecture

        | Component | Technology |
        |---|---|
        | Base Model | Llama 3.2 3B Instruct |
        | Fine-tuning | QLoRA |
        | Quantization | 4-bit NF4 |
        | Framework | Unsloth |
        | Dataset | 8,000 cybersecurity examples |
        | Training | 7,200 |
        | Validation | 800 |
        | Question Recommendation | Sentence Transformer |
        | Similarity | Cosine Similarity |
        | Runtime | Kaggle Tesla T4 |
        """
    )


    # ========================================================
    # BUTTON
    # ========================================================

    submit.click(

        respond,

        inputs=[
            msg,
            chatbot
        ],

        outputs=[
            msg,
            chatbot
        ]

    )


    # ========================================================
    # ENTER
    # ========================================================

    msg.submit(

        respond,

        inputs=[
            msg,
            chatbot
        ],

        outputs=[
            msg,
            chatbot
        ]

    )


    # ========================================================
    # REAL-TIME
    # ========================================================

    msg.input(

        realtime_suggestions,

        inputs=msg,

        outputs=suggestions_box

    )


    # ========================================================
    # SELECT SUGGESTION
    # ========================================================

    suggestions_box.change(

        select_suggestion,

        inputs=suggestions_box,

        outputs=msg

    )


    # ========================================================
    # CLEAR
    # ========================================================

    clear.click(

        clear_chat,

        outputs=[
            chatbot,
            suggestions_box
        ]

    )


# ============================================================
# 20. LAUNCH
# ============================================================

print()
print("=" * 60)
print("🚀 CYBERSHIELD READY")
print("=" * 60)
print(
    "Model: Llama 3.2 3B + CyberShield LoRA"
)
print(
    "Questions:",
    len(cybersecurity_questions)
)
print(
    "Embeddings:",
    question_embeddings.shape
)
print(
    "GPU:",
    torch.cuda.get_device_name(0)
)
print("=" * 60)


demo.launch(
    share=True,
    debug=False
)
