import time

import streamlit as st
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APIStatusError

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Symptom Checker & Triage Assistant",
    page_icon="🩺",
    layout="centered",
)

# ── System prompt ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI-powered symptom checker and triage support assistant.
Your role is to help users understand symptoms and provide general health information.

HARD RULES — never break these:
1. You are NOT a doctor. Never diagnose any condition.
2. Never prescribe medicines, recommend dosages, or advise starting or stopping any treatment.
3. Always recommend consulting a qualified healthcare professional for diagnosis and treatment.
4. Do NOT claim to access medical records, examine the user, or replace a doctor.
5. Do NOT ask for names, addresses, national identity numbers, or other sensitive personal data.
6. Keep all responses clear, cautious, and easy for a general audience to understand.

EMERGENCY RULE — apply this before anything else:
If the user reports ANY of the following (or anything similar), immediately advise them to call
local emergency services (e.g. 911 / 112 / 999) and do NOT continue as if it is a normal chat:
  - Severe difficulty breathing or choking
  - Chest pain or pressure (possible heart attack)
  - Signs of stroke (sudden facial drooping, arm weakness, slurred speech)
  - Severe or uncontrolled bleeding
  - Loss of consciousness or unresponsiveness
  - Severe allergic reaction (anaphylaxis)
  - Thoughts of self-harm or harming others

TRIAGE GUIDANCE — when appropriate, categorise using one of these urgency levels:
  🔴 EMERGENCY – call emergency services immediately
  🟠 URGENT – seek medical attention today (urgent care / ER)
  🟡 SOON – see a doctor within the next few days
  🟢 SELF-CARE – general self-care measures may help; monitor symptoms

FOLLOW-UP QUESTIONS — ask relevant follow-up questions when helpful, such as:
  - How long have you had this symptom?
  - How would you rate the severity (mild / moderate / severe)?
  - Broad age range (child, adult, elderly)?
  - Any associated symptoms?
Avoid asking for unnecessary personal information.

DISCLAIMER — always remind users:
  - This is general health information, not a diagnosis.
  - A qualified healthcare professional should evaluate their condition.
  - This tool does not access medical records or replace a doctor.
"""

# ── Supported Providers and Models ───────────────────────────────────────────────
PROVIDERS = {
    "OpenRouter (Free Models)": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_prefix": "sk-or-",
        "help_url": "https://openrouter.ai/keys",
        "models": {
            "Nvidia Nemotron 3.5 Lightning (Free)": "nvidia/nemotron-3.5-lightning:free",
            "Google Gemma 4 31B (Free)": "google/gemma-4-31b-it:free",
            "Google Gemma 4 26B (Free)": "google/gemma-4-26b-a4b-it:free",
            "Liquid LFM 2.5 2.6B (Free & Fast)": "liquid/lfm-2.5-2.6b:free",
            "Meta Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
        },
        "default_model": "Nvidia Nemotron 3.5 Lightning (Free)",
    },
    "Groq (100% Free & Ultra-Fast)": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_prefix": "gsk_",
        "help_url": "https://console.groq.com/keys",
        "models": {
            "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
            "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
        },
        "default_model": "Llama 3.3 70B Versatile",
    },
}

# ── Emergency keyword detection (fast client-side pre-check) ───────────────────
EMERGENCY_KEYWORDS = [
    "can't breathe", "cannot breathe", "trouble breathing", "difficulty breathing",
    "chest pain", "chest pressure", "heart attack",
    "stroke", "face drooping", "arm weakness", "slurred speech",
    "unconscious", "unresponsive", "passed out", "fainted",
    "severe bleeding", "uncontrolled bleeding",
    "anaphylaxis", "severe allergic",
    "suicidal", "self-harm", "kill myself",
]

def has_emergency_keywords(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in EMERGENCY_KEYWORDS)


def validate_api_key(api_key: str, base_url: str) -> tuple[bool, str | None]:
    """Validates API token by listing models without making a completion call."""
    try:
        headers = {}
        if "openrouter.ai" in base_url:
            headers = {
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Symptom Checker",
            }
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers,
        )
        # client.models.list() validates authentication without model name dependencies
        client.models.list()
        return True, None
    except APIStatusError as e:
        if e.status_code == 401:
            return False, "**Invalid API Key.** Please verify your key in the sidebar."
        if e.status_code == 403:
            return False, "**Access denied.** Your key may be restricted or lack permissions."
        return False, f"**API validation error ({e.status_code}):** {str(e)[:250]}"
    except APIConnectionError:
        return False, "**Connection error.** Could not connect to API provider."
    except Exception as e:
        return False, f"**Validation notice:** {str(e)[:250]}"


def format_api_error(e: Exception, model: str, provider_name: str) -> str:
    """Provides clear, actionable error guidance."""
    msg = str(e)
    if "ZDR violation" in msg or "data policy" in msg:
        return (
            "⚠️ **OpenRouter Privacy Setting (ZDR) Notice:**\n\n"
            "OpenRouter's free models require allowing endpoints that train or log prompts.\n\n"
            "**How to fix in 10 seconds:**\n"
            "1. Open [openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy)\n"
            "2. Turn **OFF** the 'Zero Data Retention' toggle (or set Data Policy to allow free models)\n"
            "3. Return here and send your question again!"
        )
    if "model_not_supported" in msg or "not supported by any provider" in msg:
        return (
            f"⚠️ The model `{model}` is currently busy or unavailable. "
            "Please select another model from the dropdown in the sidebar."
        )
    if isinstance(e, RateLimitError):
        return (
            "⚠️ Rate limit reached. Free tiers allow a limited number of requests per minute. "
            "Please wait a moment and try again."
        )
    if isinstance(e, APIStatusError):
        if e.status_code == 401:
            return "⚠️ Invalid API key. Please check your key in the sidebar."
        if e.status_code == 429:
            return "⚠️ Daily or minute rate limit reached. Please try another model or wait a few minutes."
    return f"API error ({getattr(e, 'status_code', 'unknown')}): {msg[:300]}"


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    provider_name = st.selectbox(
        "API Provider",
        options=list(PROVIDERS.keys()),
        index=0,
    )
    provider_config = PROVIDERS[provider_name]

    api_key = st.text_input(
        f"{provider_name.split()[0]} API Key",
        type="password",
        placeholder="Paste your API key…",
        help=f"Get your key at: {provider_config['help_url']}",
    )
    st.caption("Your key is only kept in session memory and is never logged.")

    # ── Model selector ──────────────────────────────────────────────────────────
    model_options = provider_config["models"]
    default_model_name = provider_config["default_model"]
    default_index = list(model_options.keys()).index(default_model_name)

    selected_model_label = st.selectbox(
        "Model",
        options=list(model_options.keys()),
        index=default_index,
        help="Choose a free model.",
    )
    active_model = model_options[selected_model_label]
    st.caption(f"Endpoint: `{active_model}`")

    # ── Validate key ────────────────────────────────────────────────────────────
    if api_key and api_key.strip():
        cache_id = f"{provider_name}_{hash(api_key.strip())}"
        if st.session_state.get("_key_cache_id") != cache_id:
            with st.spinner("Checking API key…"):
                key_valid, key_error = validate_api_key(api_key.strip(), provider_config["base_url"])
            st.session_state["_key_cache_id"] = cache_id
            st.session_state["_key_valid"] = key_valid
            st.session_state["_key_error"] = key_error
        else:
            key_valid = st.session_state.get("_key_valid", False)
            key_error = st.session_state.get("_key_error")

        if key_valid:
            st.success("✅ API Key is ready")
        elif key_error:
            st.warning(key_error)
            # Do not block execution if listing models had minor permissions issues
            st.session_state["_key_valid"] = True

    if "OpenRouter" in provider_name:
        with st.expander("ℹ️ Using OpenRouter Free Models?"):
            st.markdown(
                "If you see a *ZDR violation* error, go to "
                "[openrouter.ai/settings/privacy](https://openrouter.ai/settings/privacy) "
                "and disable Zero Data Retention so free community endpoints are allowed."
            )

    st.divider()
    st.subheader("🔒 Privacy Notice")
    st.caption(
        "Do **not** enter your full name, address, identity / insurance numbers, "
        "or any sensitive personal data. Share only general symptoms you are comfortable discussing."
    )

    st.divider()
    st.subheader("⚠️ Medical Disclaimer")
    st.caption(
        "This tool provides **general health information only**. "
        "It is **not** a medical device, does not diagnose conditions, "
        "and does **not** replace professional medical advice. Always consult a doctor."
    )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main header ─────────────────────────────────────────────────────────────────
st.title("🩺 AI Symptom Checker & Triage Assistant")
st.caption(
    "Describe your symptoms to receive general health information and triage guidance. "
    "This tool does **not** diagnose conditions or replace a doctor."
)

# ── Emergency banner ─────────────────────────────────────────────────────────────
st.error(
    "🚨 **If you or someone else is experiencing a life-threatening emergency "
    "(severe difficulty breathing, chest pain, signs of stroke, severe bleeding, "
    "loss of consciousness, or an anaphylactic reaction), call your local emergency "
    "services (911 / 112 / 999) immediately.**"
)

# ── Session state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome message & sample prompts ────────────────────────────────────────────
if not st.session_state.messages:
    st.info(
        "👋 **Welcome!** I can help you understand symptoms and provide general "
        "triage guidance. I am not a doctor — please consult a healthcare professional "
        "for any diagnosis or treatment decision.\n\n"
        "**Try one of these prompts:**\n"
        "- *I have had a headache and mild fever for two days — what should I do?*\n"
        "- *What warning signs mean I need urgent care?*\n"
        "- *What questions should I ask my doctor about chest tightness?*"
    )

# ── Render chat history ──────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ───────────────────────────────────────────────────────────────────
user_input = st.chat_input("Describe your symptom or ask a health question…")

if user_input:
    # Validate key presence
    if not api_key or not api_key.strip():
        st.warning(f"Please enter your {provider_name.split()[0]} API key in the sidebar to continue.")
        st.stop()

    user_input = user_input.strip()
    if not user_input:
        st.warning("Please type a message before sending.")
        st.stop()

    # Fast emergency pre-check
    if has_emergency_keywords(user_input):
        emergency_msg = (
            "🚨 **EMERGENCY — Please call your local emergency services immediately "
            "(e.g. 911 / 112 / 999).**\n\n"
            "The symptoms you described may be life-threatening. "
            "Do not wait — seek emergency medical help right now. "
            "This assistant cannot provide emergency care."
        )
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": emergency_msg})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(emergency_msg)
        st.stop()

    # Append user message and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build messages for multi-turn chat (keep last 6)
    recent_messages = st.session_state.messages[-7:-1]
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in recent_messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})
    api_messages.append({"role": "user", "content": user_input})

    # Call API with retry logic
    MAX_RETRIES = 2
    BACKOFF_BASE = 2

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            assistant_reply = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    headers = {}
                    if "openrouter.ai" in provider_config["base_url"]:
                        headers = {
                            "HTTP-Referer": "http://localhost:8501",
                            "X-Title": "Symptom Checker",
                        }

                    client = OpenAI(
                        base_url=provider_config["base_url"],
                        api_key=api_key.strip(),
                        default_headers=headers,
                    )

                    extra_kwargs = {}
                    if "openrouter.ai" in provider_config["base_url"]:
                        extra_kwargs["extra_body"] = {
                            "provider": {
                                "data_collection": "allow"
                            }
                        }

                    response = client.chat.completions.create(
                        model=active_model,
                        messages=api_messages,
                        temperature=0.3,
                        max_tokens=2048,
                        **extra_kwargs,
                    )

                    choice = response.choices[0] if response.choices else None
                    if choice and choice.message and choice.message.content:
                        assistant_reply = choice.message.content.strip()

                    finish = choice.finish_reason if choice else "no choices"

                    if assistant_reply and finish == "length":
                        assistant_reply += (
                            "\n\n---\n*Note: Response was cut off due to length. "
                            "Feel free to ask a follow-up question.*"
                        )

                    if not assistant_reply:
                        assistant_reply = (
                            "⚠️ The model did not return any text. "
                            f"(finish reason: `{finish}`). "
                            "Please try again or select a different free model from the sidebar."
                        )

                    break  # success

                except (APIConnectionError, APIStatusError) as e:
                    status = getattr(e, "status_code", None)
                    if status and status >= 500 and attempt < MAX_RETRIES:
                        time.sleep(BACKOFF_BASE ** attempt)
                        continue
                    assistant_reply = format_api_error(e, active_model, provider_name)
                    break
                except RateLimitError as e:
                    if attempt < MAX_RETRIES:
                        time.sleep(BACKOFF_BASE ** attempt)
                        continue
                    assistant_reply = format_api_error(e, active_model, provider_name)
                    break
                except Exception as e:
                    assistant_reply = f"⚠️ Unexpected error: {str(e)[:300]}"
                    break

        st.markdown(assistant_reply)
        st.caption(
            "ℹ️ General health information only — not medical diagnosis. "
            "Always consult a qualified healthcare professional."
        )

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
