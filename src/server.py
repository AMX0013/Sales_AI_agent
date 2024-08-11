import os
import uuid
import pyaudio
import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import record_audio_chunk, transcribe_audio, play_text_to_speech, load_whisper
from graph import create_graphflow

def initialize_audio_stream():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    return audio, stream

def cleanup_audio_stream(audio, stream):
    stream.stop_stream()
    stream.close()
    audio.terminate()

def display_message(user_type, message):
    background_color = "#f0f0f0" if user_type == "user" else "#d0d0f0"
    st.markdown(f'<div style="background-color: {background_color}; padding: 10px; border-radius: 5px;">{user_type.capitalize()} 👤: {message}</div>', unsafe_allow_html=True)

def main():
    st.markdown('<h1 style="color: darkblue;">AI Voice Assistant️</h1>', unsafe_allow_html=True)
    
    model = load_whisper()
    graph = create_graphflow()
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }
    
    if st.button("Enquire Now!"):
        audio, stream = initialize_audio_stream()
        try:
            while True:
                record_audio_chunk(audio, stream, 'temp_audio_chunk.wav')
                text = transcribe_audio(model, 'temp_audio_chunk.wav')
                
                if text:
                    display_message("customer", text)
                    os.remove('temp_audio_chunk.wav')
                    
                    messages = graph.invoke({"messages": ("user", text)}, config)
                    response = messages['messages'][-1].content
                    
                    display_message("ai assistant", response)
                    play_text_to_speech(text=response)
                else:
                    break  # Exit the loop if transcription returns None or empty string
        finally:
            cleanup_audio_stream(audio, stream)
        print("End Conversation")

if __name__ == "__main__":
    main()
