# 🛡️ CyberShield

### Fine-Tuned Cybersecurity LLM + Real-Time Next Question Recommendation

CyberShield is an AI-powered defensive cybersecurity assistant built by combining a **QLoRA fine-tuned Llama 3.2 3B Instruct model** with a **semantic next-question recommendation system**.

The system doesn't just answer cybersecurity questions — it also understands what the user is typing and recommends relevant cybersecurity questions in real time.

---

## 🚀 Project Overview

CyberShield consists of two independent AI components:

### 1️⃣ Cybersecurity LLM

A **Llama 3.2 3B Instruct** model fine-tuned on cybersecurity instruction data using:

- QLoRA
- 4-bit NF4 quantization
- Unsloth
- PEFT
- Transformers

### 2️⃣ Next Question Recommendation Model

A semantic recommendation system based on:

- Sentence Transformers
- `all-MiniLM-L6-v2`
- 7,999 cybersecurity questions
- 384-dimensional embeddings
- Cosine similarity

These two models work together inside a Gradio interface.

---

# 🧠 System Architecture

```text
                         USER
                           │
                           ▼
                 ┌──────────────────┐
                 │ CyberShield UI   │
                 │     Gradio       │
                 └────────┬─────────┘
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
    ┌──────────────────┐      ┌────────────────────┐
    │ Llama 3.2 3B     │      │ Question           │
    │ Instruct         │      │ Recommendation     │
    │                  │      │                    │
    │ QLoRA Fine-tuned │      │ Sentence           │
    │                  │      │ Transformer        │
    └────────┬─────────┘      └─────────┬──────────┘
             │                          │
             ▼                          ▼
    ┌──────────────────┐      ┌────────────────────┐
    │ Cybersecurity    │      │ Question Embedding │
    │ Answer           │      │ 7999 × 384         │
    └──────────────────┘      └─────────┬──────────┘
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ Cosine Similarity  │
                              └─────────┬──────────┘
                                        │
                                        ▼
                              💡 Next Questions
