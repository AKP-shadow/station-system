import os
import pytube
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
from scipy.io import wavfile
import noisereduce as nr

 

# video_path = os.path.join(f"downloads/{video_filename}" + '.mp4')
audio_path = os.path.join(f"../downloads/test2.wav")

# audio = AudioSegment.from_file(video_path, format='mp4')
# audio.export(audio_path, format='wav')
def noise_reduction(audio_path, output_path):
    audio = AudioSegment.from_mp3(audio_path)
    audio = audio.set_channels(1)  # Convert to mono (optional)
    audio = audio.set_frame_rate(44100)  # Set the frame rate (optional)

    audio_segments = split_on_silence(audio, min_silence_len=1000, silence_thresh=-200)

    reduced_audio = AudioSegment.empty()
    for segment in audio_segments:
        reduced_audio += segment

    reduced_audio.export(output_path, format='wav')

reduced_audio_path = os.path.join(f'downloads/test' + '_reduced.wav')
noise_reduction(audio_path,reduced_audio_path)




def extract_text_from_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source,offset=4,duration=15)
    try:
        recognizer.energy_threshold = 2000  
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

extracted_text = extract_text_from_audio(reduced_audio_path)
print("Extracted Text:")
print(extracted_text)

# Clean up temporary files
# os.remove(video_path)
# os.remove(audio_path)
# os.remove(reduced_audio_path)
