
#  AI Symptom Checker & Triage Assistant

An intelligent, conversational healthcare triage assistant built with **Python**, **Streamlit**, and **OpenAI-compatible LLM backends** (OpenRouter & Groq).

The application evaluates user symptoms, flags potential medical emergencies in real-time, categorizes care urgency into 4 distinct tiers, and offers actionable next steps—all while enforcing strict clinical and privacy guardrails.

---

> ⚠️ **MEDICAL DISCLAIMER:**
> This application provides **general health information and triage guidance only**. It is **not** a certified medical device, does **not** diagnose conditions, and does **not** replace consultation with a qualified healthcare professional. It will **never** prescribe medications or recommend dosages. If you suspect a medical emergency, call your local emergency services (e.g., 911 / 112 / 999) immediately.

---

## 📌 Problem Statement & Solution

* **The Problem:** Patients frequently turn to search engines for symptom analysis, leading to undue panic (*cyberchondria*), dangerous delays during acute emergencies, and hospital emergency room overcrowding for non-urgent issues.
* **The Solution:** A dual-layer triage system combining **deterministic rule-based emergency detection** (zero-latency red-flag screening) with **contextual generative AI** to classify urgency into clear, actionable care tiers without issuing unverified diagnoses.

---

## ✨ Key Features

* 🚨 **Deterministic Emergency Interception:** Real-time client-side keyword matching flags life-threatening symptoms (chest pain, stroke signs, breathing failure, severe bleeding) and immediately triggers emergency protocols.
* 🏷️ **4-Tier Clinical Urgency Classification:**
  * 🔴 **Emergency:** Immediate emergency services / ER dispatch
  * 🟠 **Urgent:** Same-day clinical assessment (urgent care)
  * 🟡 **Soon:** Outpatient doctor visit within 2–3 days
  * 🟢 **Self-Care:** Home management, symptom tracking, and red-flag vigilance
* ❓ **Adaptive Clinical Follow-Ups:** Inquires about symptom duration, pain severity (1–10 scale), and general age bracket to refine guidance.
* 🔒 **Zero-Data-Retention (ZDR) Privacy:** No personal identifiable information (names, IDs, phone numbers) is collected. All data and API keys reside exclusively in temporary session memory.
* ⚡ **Multi-Provider Free LLM Support:** Seamlessly switch between **OpenRouter** (community free models) and **Groq** (ultra-fast LPU inference).
* 🔄 **Fault-Tolerant Retries:** Automatic exponential backoff handling rate limits and network latency.

---

## 🏗️ System Architecture & Workflow

```text
[ User Input (Natural Language) ]
               │
               ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Emergency Keyword Pre-Check        │
│  (Chest pain, stroke, breathing, bleeding)   │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼ [Red Flag Detected]           ▼ [Safe / Non-Emergency]
┌─────────────────────────────┐ ┌──────────────────────────────────────┐
│ 🚨 Immediate Alert:         │ │ Stage 2: Sliding Context Assembly    │
│ Directs to 911 / 112 / 999  │ │ (Retains last 3 turns)               │
└─────────────────────────────┘ └──────────────────┬───────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Stage 3: LLM Inference with Prompts  │
                                │ (OpenRouter / Groq API)              │
                                └──────────────────┬───────────────────┘
                                                   │
                                                   ▼
                                ┌──────────────────────────────────────┐
                                │ Stage 4: 4-Tier Triage Output        │
                                │ + Clinical Follow-up + Disclaimer    │
                                └──────────────────────────────────────┘
```

---

## 🔑 Supported Providers & Free API Setup

### Option A: OpenRouter (Recommended)

1. Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys).
2. **Important Privacy Setting:** Free models on OpenRouter require allowing non-ZDR endpoints.
   * Go to [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy).
   * Toggle **OFF** **"Zero Data Retention (ZDR)"** (or set Data Policy to allow free models).
3. **Available Free Models in App:**
   * `Nvidia Nemotron 3.5 Lightning (Free)` *(Default, 1M context)*
   * `Google Gemma 4 31B (Free)`
   * `Google Gemma 4 26B (Free)`
   * `Liquid LFM 2.5 2.6B (Free & Fast)`
   * `Meta Llama 3.3 70B (Free)`

### Option B: Groq (100% Free & Ultra-Fast)

1. Get a free API key instantly at [console.groq.com/keys](https://console.groq.com/keys) (no credit card required).
2. **Available Models in App:**
   * `Llama 3.3 70B Versatile`
   * `Llama 3.1 8B Instant`

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-symptom-checker.git
cd ai-symptom-checker
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, enter your API key in the sidebar, and begin chatting.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push your project to a GitHub repository (ensure `app.py` and `requirements.txt` are in the root directory).
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **New app**, select your repository, branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**.
5. Users can enter their own free API key directly in the sidebar on the deployed web page.

---

## 📁 Project Structure

```text
.
├── app.py              # Main Streamlit web application & triage logic
├── requirements.txt    # Python dependencies (streamlit, openai)
├── .streamlit/
│   └── config.toml     # UI theme styling & server configuration
└── README.md           # Project documentation
```

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.9+
* **Frontend / Framework:** [Streamlit](https://streamlit.io/)
* **API Client SDK:** [OpenAI Python SDK](https://github.com/openai/openai-python)
* **LLM Endpoints:** [OpenRouter API](https://openrouter.ai/) & [Groq Cloud](https://groq.com/)

---

## 🛡️ Safety, Guardrails & Ethical Compliance

* **No Diagnosis Policy:** System prompts explicitly prohibit naming definite diseases or claiming diagnostic authority.
* **No Prescription Policy:** Prohibits recommending specific drug brand names, dosages, or self-medication regimens.
* **Red-Flag Interception:** Catches acute distress phrases before querying the LLM to protect high-risk patients.
* **Session Ephemerality:** No conversation history or keys are stored in databases or server-side logs.

---

## 📜 References

1. **World Health Organization (WHO):** *Guidance on Ethics & Governance of Artificial Intelligence for Health*, 2021.
2. **Manchester Triage System (MTS):** Mackway-Jones, K. et al., *Emergency Triage*, BMJ Publishing Group, 3rd Ed.
3. **Clinical AI Benchmarks:** Singhal, K. et al., *"Large Language Models Encode Clinical Knowledge"*, *Nature*, 620, pp. 172–180, 2023.
4. **Symptom Checker Audits:** Semigran, H. L. et al., *"Evaluation of Symptom Checkers for Self-Diagnosis and Triage"*, *BMJ*, 351:h3480, 2015.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
