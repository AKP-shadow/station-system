
 
import webrtcvad
import pyaudio
import wave

vad = webrtcvad.Vad()
vad.set_mode(0)  # Aggressiveness level (0-3)

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

RECORD_SECONDS = 4  

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

frames = []
active = False

print("Listening for speech...")

while True:
    data = stream.read(CHUNK)
    frames.append(data)
    print(len(data))
    if vad.is_speech(data, RATE):
        if not active:
            print("Speech detected.")
            active = True
    elif active:
        print("Speech ended.")
        active = False
        break

print("Recording completed.")

stream.stop_stream()
stream.close()
audio.terminate()

# Save the recorded speech to a WAV file
with wave.open("recorded_speech.wav", "wb") as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
