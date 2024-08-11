import os
from dotenv import load_dotenv

import wave
import pyaudio
from scipy.io import wavfile
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
import whisper

from gtts import gTTS
import pygame




def is_silence(data, max_amplitude_threshold=300):
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    # print("Is silinet")
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=5):
    print("Recording...")
    frames = []
    # Calculate the number of chunks needed for the specified length of recording
    # 16000 Hertz -> sufficient for capturing the human voice
    # 1024 frames -> the higher, the higher the latency
    num_chunks = int(16000 / 1024 * chunk_length)

    # Record the audio data in chunks
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = './temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # Sample width
        wf.setframerate(16000)  # Sample rate
        wf.writeframes(b''.join(frames))  # Write audio frames

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            print("Silence detected")
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")





def load_whisper():
    model = whisper.load_model("base")
    return model


def transcribe_audio(model, file_path):
    print("Transcribing...")
    # Print all files in the current directory
    print("Current directory files:", os.listdir())
    if os.path.isfile(file_path):
        print("Current directory files:", os.listdir(), file_path, file_path in os.listdir())
        results = model.transcribe(file_path , fp16=False)
        return results['text']
    else:
        return None



def play_text_to_speech(text, language='en', slow=False, speed_factor=1.3):
    # Generate text-to-speech audio from the provided text
    tts = gTTS(text=text, lang=language, slow=slow)

    # Save the generated audio to a temporary file
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    # Load the audio file with pydub
    audio = AudioSegment.from_file(temp_audio_file)

    # Adjust the speed (playback speed)
    fast_audio = audio.speedup(playback_speed=speed_factor)

    # Play the adjusted audio
    play(fast_audio)

    # Clean up: Remove the temporary audio file
    os.remove(temp_audio_file)
    
    
def deepgram_tts(text):
    from deepgram import (
    DeepgramClient,
    SpeakOptions,
    )
    load_dotenv()
    deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
    # STEP 2: Configure the options (such as model choice, audio configuration, etc.)
    options = SpeakOptions(
        model="aura-asteria-en",
        encoding="linear16",
        container="wav"
    )
    SPEAK_OPTIONS = {"text": text}
    
    temp_audio_file = "temp_audio.mp3"
    response = deepgram.speak.v("1").save(temp_audio_file, SPEAK_OPTIONS, options)
    print(response.to_json(indent=4))
    
    # Initialize the pygame mixer for audio playback
    pygame.mixer.init()

    # Load the temporary audio file into the mixer
    pygame.mixer.music.load(temp_audio_file)

    # Start playing the audio
    pygame.mixer.music.play()

    # Wait until the audio playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(1)  # Control the playback speed

    # Stop the audio playback
    pygame.mixer.music.stop()

    # Clean up: Quit the pygame mixer and remove the temporary audio file
    pygame.mixer.quit()
    os.remove(temp_audio_file)
    
    