import os

import pyaudio
import streamlit as st
from langchain.memory import ConversationBufferMemory

from utils import record_audio_chunk, transcribe_audio, play_text_to_speech, load_whisper
from graph import create_graphflow

chunk_file = 'temp_audio_chunk.wav'




model = load_whisper()
def main():
    st.markdown('<h1 style="color: darkblue;">AI Voice Assistant️</h1>', unsafe_allow_html=True)
    # configure the printing of the chat history ?
    
    graph = create_graphflow()
    
    import uuid 
    _printed = set()
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }
    

    if st.button("Enquire Now!"):
        
        while True:
            # Audio Stream Initialization
            audio = pyaudio.PyAudio()
            
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

            # Record and save audio chunk
            record_audio_chunk(audio, stream)

            text = transcribe_audio(model, chunk_file)
            print(text)
            if text is not None:
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">Customer 👤: {text}</div>',
                    unsafe_allow_html=True)

                os.remove(chunk_file)
                
                msg = {"messages": ("user", text)}
                messages = graph.invoke(msg,config)
                response_llm = messages['messages'][-1].content
                
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">AI Assistant 🤖: {response_llm}</div>',
                    unsafe_allow_html=True)

                play_text_to_speech(text=response_llm)
                
            else:
                stream.stop_stream()
                stream.close()
                audio.terminate()
                break  # Exit the while loop
        print("End Conversation")



if __name__ == "__main__":
    main()