import assemblyai as aai

# replace with your API token
aai.settings.api_key = f"d2855d1b70b1426ca82150cfe8cc8cff"

# URL of the file to transcribe
FILE_URL = "./downloads/test2.wav"

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)

print(transcript.text)
