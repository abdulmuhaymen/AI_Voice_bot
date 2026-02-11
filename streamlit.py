import streamlit as st
import os
import numpy as np
import tempfile
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from google.genai import Client
from google.genai.errors import ClientError
import time
import wave
from audiorecorder import audiorecorder

# =============================
# Page Configuration
# =============================
st.set_page_config(
    page_title="ਪੰਜਾਬੀ/اردو AI Voice Bot",
    page_icon="🎤",
    layout="centered"
)

# =============================
# Load Environment Variables
# =============================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GEMINI_API_KEY or not ELEVENLABS_API_KEY:
    st.error("❌ Missing API keys in .env file or Streamlit secrets")
    st.stop()

# =============================
# Initialize Clients
# =============================
@st.cache_resource
def initialize_clients():
    genai_client = Client(
        api_key=GEMINI_API_KEY,
        http_options={"api_version": "v1"}
    )
    eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return genai_client, eleven_client

genai_client, eleven_client = initialize_clients()

# =============================
# Detect Available Gemini Model
# =============================
@st.cache_data
def get_working_model():
    models = genai_client.models.list()
    for m in models:
        if "flash" in m.name.lower():
            return m.name.replace("models/", "")
    return models[0].name.replace("models/", "")

MODEL_NAME = get_working_model()

# =============================
# Speech to Text with Multiple Language Attempts
# =============================
def speech_to_text_multilang(audio_path):
    """
    Try transcribing with both Punjabi and Urdu to see which works better.
    Returns: (transcribed_text, detected_language_code)
    """
    results = []
    
    # Try Urdu first
    try:
        with open(audio_path, "rb") as audio_file:
            transcript_ur = eleven_client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                language_code="urd"  # Urdu
            )
            if transcript_ur.text.strip():
                results.append(("urd", transcript_ur.text.strip()))
    except Exception as e:
        pass
    
    # Try Punjabi
    try:
        with open(audio_path, "rb") as audio_file:
            transcript_pa = eleven_client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v1",
                language_code="pan"  # Punjabi
            )
            if transcript_pa.text.strip():
                results.append(("pan", transcript_pa.text.strip()))
    except Exception as e:
        pass
    
    # If we have results, let user choose or use AI to detect
    if results:
        # For now, return the first successful result
        # You can enhance this by comparing which one makes more sense
        return results[0][1], results[0][0]
    
    return "", "pan"

# =============================
# Intelligent Language Detection using Gemini
# =============================
def detect_language_with_ai(text):
    """
    Use Gemini to detect if the text is Punjabi or Urdu.
    This is more reliable than script detection.
    """
    if not text.strip():
        return "pan"
    
    detection_prompt = f"""You are a language detection expert. 
Analyze the following text and determine if it is:
1. Punjabi (ਪੰਜਾਬੀ) - can be written in Gurmukhi or Shahmukhi script
2. Urdu (اردو) - written in Perso-Arabic script
3. English

Text: "{text}"

Respond with ONLY ONE WORD: either "punjabi", "urdu", or "english"
Do not provide any explanation, just the language name in lowercase."""

    try:
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=detection_prompt,
        )
        
        detected = response.text.strip().lower()
        
        if "urdu" in detected:
            return "urd"
        elif "punjabi" in detected or "panjabi" in detected:
            return "pan"
        elif "english" in detected:
            return "eng"
        else:
            # Fallback to script detection
            return fallback_script_detection(text)
            
    except Exception as e:
        st.warning(f"AI detection failed: {e}")
        return fallback_script_detection(text)

# =============================
# Fallback Script Detection
# =============================
def fallback_script_detection(text):
    """Fallback method using character analysis"""
    if not text:
        return "pan"
    
    # Count different script characters
    gurmukhi = sum(1 for c in text if '\u0A00' <= c <= '\u0A7F')
    perso_arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\u0750' <= c <= '\u077F')
    latin = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    
    # Urdu-specific characters
    urdu_specific = sum(1 for c in text if c in 'ںےۓہھ')
    
    # Punjabi-specific (Shahmukhi)
    punjabi_specific = sum(1 for c in text if c in 'ੜ੍')
    
    if gurmukhi > 0:
        return "pan"
    elif urdu_specific > punjabi_specific and perso_arabic > 0:
        return "urd"
    elif latin > perso_arabic and latin > gurmukhi:
        return "eng"
    else:
        # When in doubt, ask user or default
        return "urd" if perso_arabic > 0 else "pan"

# =============================
# Gemini Response with Language Detection
# =============================
def generate_response(prompt, detected_language):
    """
    Generate response in the same language as the user's input.
    """
    if not prompt.strip():
        return ""
    
    # Language-specific instructions
    language_instructions = {
        "urd": """You are a helpful AI assistant. The user is speaking in Urdu (اردو). 
You MUST respond ONLY in Urdu using Perso-Arabic/Nastaliq script.
Important: This is URDU, not Punjabi. Use proper Urdu vocabulary and grammar.
Keep your responses natural, conversational, and friendly in Urdu.
Do not mix languages. Use only Urdu (اردو).""",
        
        "pan": """You are a helpful AI assistant. The user is speaking in Punjabi (ਪੰਜਾਬੀ). 
You MUST respond ONLY in Punjabi.
Important: This is PUNJABI, not Urdu. Use proper Punjabi vocabulary and grammar.
You can use either Gurmukhi (ਪੰਜਾਬੀ) or Shahmukhi (پنجابی) script.
Keep your responses natural, conversational, and friendly in Punjabi.
Do not mix languages. Use only Punjabi.""",
        
        "eng": """You are a helpful AI assistant. The user is speaking in English.
Respond in English with natural and conversational language."""
    }
    
    system_instruction = language_instructions.get(detected_language, language_instructions["pan"])
    
    full_prompt = f"{system_instruction}\n\nUser: {prompt}\n\nAssistant:"
    
    response_text = ""
    try:
        stream = genai_client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=full_prompt,
        )
        for chunk in stream:
            if chunk.text:
                response_text += chunk.text
        return response_text.strip()
    except ClientError as e:
        st.error(f"❌ Gemini API Error: {e}")
        return ""

# =============================
# Text to Speech
# =============================
def speak_text(text):
    if not text.strip():
        return
    
    try:
        audio_bytes = eleven_client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_turbo_v2_5",
            text=text,
        )
        
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(temp_audio.name, "wb") as f:
            for chunk in audio_bytes:
                f.write(chunk)
        
        # Use Streamlit's audio player instead of sounddevice for playback
        st.audio(temp_audio.name)
        
        os.remove(temp_audio.name)
    except Exception as e:
        st.error(f"❌ TTS Error: {e}")

# =============================
# Get Language Display Name
# =============================
def get_language_display(lang_code):
    """Return display name and flag for language code"""
    lang_map = {
        "urd": ("Urdu (اردو)", "🇵🇰"),
        "pan": ("Punjabi (ਪੰਜਾਬੀ/پنجابی)", "🇮🇳"),
        "eng": ("English", "🇬🇧")
    }
    return lang_map.get(lang_code, ("Unknown", "🌐"))

# =============================
# Process User Input (Text or Voice)
# =============================
def process_user_input(user_text, detected_lang=None):
    """Common function to process both text and voice input"""
    if not user_text.strip():
        return
    
    # Use manual override if enabled, otherwise use AI detection
    if st.session_state.get('manual_override', False):
        detected_lang = st.session_state.get('selected_lang', 'pan')
        st.info(f"🔧 Manual override: {get_language_display(detected_lang)[0]}")
    elif detected_lang is None:
        with st.spinner("🤖 Detecting language with AI..."):
            detected_lang = detect_language_with_ai(user_text)
    
    lang_name, lang_flag = get_language_display(detected_lang)
    
    st.success(f"**🧑 You {lang_flag}:** {user_text}")
    st.info(f"✅ Detected Language: **{lang_name}**")
    
    with st.spinner(f"🤖 Generating response in {lang_name}..."):
        # Generate AI response in the detected language
        response = generate_response(user_text, detected_lang)
        
        if response:
            st.info(f"**🤖 AI {lang_flag}:** {response}")
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                'user': user_text,
                'ai': response,
                'user_language': detected_lang,
                'ai_language': detected_lang
            })
            
            with st.spinner(f"🔊 Speaking..."):
                speak_text(response)
            
            st.success("✅ Response completed!")
        else:
            st.error("❌ Failed to generate response")

# =============================
# Streamlit UI
# =============================
st.title("🎤 ਪੰਜਾਬੀ/اردو AI Voice Bot")
st.markdown("### 🗣️ Speak in Any Language - AI Detects & Responds")
st.markdown("**Supported:** Punjabi (ਪੰਜਾਬੀ/پنجابی) | Urdu (اردو) | English")
st.markdown("---")

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Language selection override (optional manual selection)
st.sidebar.header("⚙️ Settings")
st.session_state.manual_override = st.sidebar.checkbox("🔧 Manual Language Override", value=False)
if st.session_state.manual_override:
    st.session_state.selected_lang = st.sidebar.selectbox(
        "Select Language",
        options=["urd", "pan", "eng"],
        format_func=lambda x: get_language_display(x)[0]
    )

# Display conversation history
if st.session_state.conversation_history:
    st.subheader("💬 Conversation History")
    for i, entry in enumerate(st.session_state.conversation_history):
        with st.container():
            user_lang_name, user_flag = get_language_display(entry.get('user_language', 'pan'))
            ai_lang_name, ai_flag = get_language_display(entry.get('ai_language', 'pan'))
            
            st.markdown(f"**🧑 You {user_flag} ({user_lang_name}):** {entry['user']}")
            st.markdown(f"**🤖 AI {ai_flag} ({ai_lang_name}):** {entry['ai']}")
            st.markdown("---")

# Input method selection
st.subheader("💬 Send Message")

# Text Input Tab
with st.expander("✍️ Text Input", expanded=False):
    text_input = st.text_area(
        "Type your message in Punjabi, Urdu, or English:",
        height=100,
        placeholder="ਪੰਜਾਬੀ / اردو / English"
    )
    
    if st.button("📤 Send Text", type="primary", use_container_width=True):
        if text_input.strip():
            process_user_input(text_input)
        else:
            st.warning("⚠️ Please enter some text")

# Voice Input (works on cloud!)
with st.expander("🎤 Voice Input", expanded=True):
    st.markdown("**Click the mic button below, speak, then click stop:**")
    
    audio = audiorecorder("🎤 Start Recording", "⏹️ Stop Recording")
    
    if len(audio) > 0:
        # Save the recorded audio to a temporary file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_audio.name, format="wav")
        
        with st.spinner("🔄 Transcribing your voice..."):
            # Convert speech to text
            user_text, initial_lang = speech_to_text_multilang(temp_audio.name)
            
            if not user_text:
                st.warning("⚠️ کوئی آواز نہیں ملی / ਕੋਈ ਆਵਾਜ਼ ਨਹੀਂ ਮਿਲੀ / No voice detected")
            else:
                process_user_input(user_text, initial_lang)
            
            os.remove(temp_audio.name)
# Footer
st.markdown("---")
st.markdown("**🤖 AI Model:** " + MODEL_NAME)
st.markdown("**🔍 Detection:** AI-powered language detection")

# Clear conversation button
if st.session_state.conversation_history:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Clear Conversation", type="secondary", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()