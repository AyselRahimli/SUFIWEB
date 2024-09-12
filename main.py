import streamlit as st
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a pre-trained conversational model (DialoGPT in this case)
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to recognize speech and convert to text
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Mikrofona danışın...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="az-AZ")
        st.write("Siz dediniz: ", text)
        return text
    except sr.UnknownValueError:
        st.write("Səs tanınmadı")
        return None

# Function to generate chatbot response using DialoGPT
def get_chatbot_response(user_input):
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Text-to-speech (TTS) function for Azerbaijani responses
def speak(text):
    tts = gTTS(text=text, lang='az')
    audio = BytesIO()
    tts.write_to_fp(audio)
    audio.seek(0)
    return audio

# Streamlit web interface
st.title("Azerbaijani Voice Chatbot")
st.write("Səsli chatbotla danışın")

if st.button("Söhbətə Başlayın"):
    user_input = recognize_speech()
    if user_input:
        response = get_chatbot_response(user_input)
        st.write("Chatbot cavabı: ", response)

        # Play the response in Azerbaijani
        audio = speak(response)
        st.audio(audio, format='audio/mp3')
