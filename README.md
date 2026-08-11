# 🛡️ CyberShield

<p align="center">
  <img src="https://img.shields.io/badge/Model-Llama%203.2%203B-blue?style=for-the-badge" alt="Llama 3.2 3B">
  <img src="https://img.shields.io/badge/Fine--Tuning-QLoRA-purple?style=for-the-badge" alt="QLoRA">
  <img src="https://img.shields.io/badge/Quantization-4--bit%20NF4-orange?style=for-the-badge" alt="4-bit NF4">
  <img src="https://img.shields.io/badge/Framework-Unsloth-green?style=for-the-badge" alt="Unsloth">
  <img src="https://img.shields.io/badge/UI-Gradio-yellow?style=for-the-badge" alt="Gradio">
</p>

<p align="center">
  <b>A specialized defensive cybersecurity assistant fine-tuned with QLoRA on cybersecurity instruction data.</b>
</p>

---

## 🎥 Demo

<p align="center">
  <img src="demo.gif" alt="CyberShield demo" width="850">
</p>

> **Note:** Add your recorded demo GIF as `assets/demo.gif`. GitHub renders repository GIFs directly in a README using standard Markdown/HTML image syntax.

---

## ✨ Overview

**CyberShield** is a cybersecurity-focused conversational AI built by fine-tuning **Meta Llama 3.2 3B Instruct** with **QLoRA**.

The goal is to create a lightweight model that provides clear, structured and defensive cybersecurity guidance while being practical to run on limited GPU hardware.

### What CyberShield can help with

- 🔐 Network security
- 🎣 Phishing awareness
- 🦠 Malware concepts
- 🔥 Ransomware response
- 🌐 Web security
- 💉 SQL injection concepts
- 🛡️ IDS / IPS
- 🔑 Authentication and IAM
- 🚨 Incident response
- 🔎 Vulnerability concepts
- 💻 Secure coding
- 📚 Cybersecurity fundamentals

---

## 🧠 Model

| Component | Details |
|---|---|
| Base Model | Llama 3.2 3B Instruct |
| Fine-tuning | QLoRA |
| Quantization | 4-bit NF4 |
| Training Framework | Unsloth |
| Training Library | Transformers + TRL + PEFT |
| Training Dataset | Cybersecurity instruction dataset |
| Dataset Size | 8,000 examples |
| Training Split | 7,200 examples |
| Validation Split | 800 examples |
| Training GPU | Google Colab T4 16GB |
| Demo Runtime | Kaggle GPU |
| Interface | Gradio |

---

## 🏗️ Architecture

```text
Cybersecurity Instruction Dataset
              │
              ▼
       Data Cleaning
              │
              ▼
         8,000 Examples
          /          \
         ▼            ▼
     7,200 Train    800 Validation
         │
         ▼
  Llama 3.2 3B Instruct
         │
         ▼
      4-bit NF4
         │
         ▼
        QLoRA
         │
         ▼
       Unsloth
         │
         ▼
   CyberShield LoRA
         │
         ▼
       Gradio UI
         │
         ▼
  Defensive Cybersecurity Assistant
```

---

## ⚡ Why QLoRA?

Llama 3.2 3B contains billions of parameters, making full fine-tuning unnecessarily expensive for a 16 GB GPU.

QLoRA solves this by:

1. Loading the base model in 4-bit precision.
2. Keeping most base-model weights frozen.
3. Adding small trainable LoRA adapters.
4. Updating only the adapter parameters.

During training, approximately **24.3M parameters** were trainable out of roughly **3.24B total parameters**.

```text
Total parameters       ≈ 3.24B
Trainable parameters   ≈ 24.3M
Trainable percentage   ≈ 0.75%
```

---

## 📊 Dataset

The model was trained using a cybersecurity instruction-tuning dataset.

Each example follows a conversational structure:

```json
{
  "system": "You are a cybersecurity expert.",
  "user": "What is phishing?",
  "assistant": "Phishing is a social engineering technique..."
}
```

The final experiment used:

```text
8,000 total examples
│
├── 7,200 training
└──   800 validation
```

The training examples were converted using the Llama chat template before tokenization.

---

## 🛠️ Tech Stack

```text
Python
PyTorch
Hugging Face Transformers
Hugging Face TRL
PEFT
Unsloth
BitsAndBytes
Llama 3.2 3B Instruct
QLoRA
Gradio
Kaggle / Google Colab
```

---

## 🚀 Running the Model

### 1. Install dependencies

```bash
pip install unsloth peft transformers accelerate bitsandbytes gradio
```

### 2. Load the base model

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

### 3. Load the CyberShield LoRA adapter

```python
from peft import PeftModel

model = PeftModel.from_pretrained(
    model,
    "./CyberShield-Llama-3.2-3B-8K"
)
```

### 4. Enable inference

```python
FastLanguageModel.for_inference(model)
```

### 5. Launch the Gradio interface

```python
demo.launch(
    share=True
)
```

---

## 💬 Example

### User

```text
What is phishing?
```

### CyberShield

```text
Phishing is a social engineering attack where an attacker
attempts to trick a victim into revealing sensitive information
such as passwords, financial information, or authentication codes.

Common defenses include:

• Multi-factor authentication
• Security awareness training
• Email filtering
• Checking suspicious links
• Verifying unexpected requests
```

---

## 📈 Evaluation

CyberShield should be evaluated using both quantitative and qualitative methods.

### Quantitative

- Validation loss
- Perplexity

Perplexity can be calculated from validation loss:

```python
import math

perplexity = math.exp(eval_loss)
```

### Qualitative

Responses can be evaluated on:

| Metric | Score |
|---|---:|
| Correctness | /5 |
| Relevance | /5 |
| Completeness | /5 |
| Technical Quality | /5 |
| Defensive Safety | /5 |

A useful evaluation should compare the **base Llama model** against **CyberShield** on the same unseen cybersecurity questions.

---

## 📁 Project Structure

```text
CyberShield/
│
├── README.md
├── app.py
├── inference.py
├── requirements.txt
│
├── adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
│
└── assets/
    └── demo.gif
```

---

## ⚠️ Responsible Use

CyberShield is designed for **defensive cybersecurity education, awareness, analysis and secure development**.

It should not be treated as a replacement for professional security teams, incident-response procedures, or verified security documentation.

Always test security techniques only on systems and environments where you have explicit authorization.

---

## 🔮 Future Improvements

- [ ] Improve dataset quality and diversity
- [ ] Add a dedicated cybersecurity benchmark
- [ ] Compare base model vs fine-tuned model
- [ ] Add RAG for current security documentation
- [ ] Add MITRE ATT&CK knowledge integration
- [ ] Add CVE lookup and vulnerability analysis
- [ ] Add structured threat-analysis reports
- [ ] Add conversation export
- [ ] Deploy a lightweight local version
- [ ] Optimize inference for RTX 2050-class GPUs

---

## 👨‍💻 Project

**CyberShield — Defensive Cybersecurity AI**

Built with:

**Llama 3.2 3B + QLoRA + Unsloth + Gradio**

---

## ⭐ If you find this project useful

Give the repository a ⭐ and feel free to explore, improve and experiment with the model.
