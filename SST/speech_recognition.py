#  !/usr/bin/env python3

import speech_recognition as sr
 
# obtain audio from the microphone
r = sr.Recognizer()
audio_file_path = './downloads/test2.wav'

 
# with sr.Microphone() as source:
#     print("Say something!")
#     audio = r.listen(source)

with sr.AudioFile(audio_file_path) as source:
    print("Processing audio file...")
    r.energy_threshold = 4000
    english_text=""
    while True:
        try:
            audio = r.record(source,duration=12,offset=-3)  
            try:
                english_text += r.recognize_google(audio, language="en-IN" ) + "/////"
            except sr.UnknownValueError:
                print("Speech recognition could not understand English audio")
                break
            except sr.RequestError as e:
                print("Could not request English results; {0}".format(e))
                break
        except:
            print("end of Audio")
            break
    
    print("English Transcription: " , english_text)
 
    # audio = r.record(source,offset=60)  
    # try:
        
    #     tamil_text = r.recognize_google(audio,language="ta-IN" )
    #     print("Tamil Transcription: " + tamil_text)
    # except sr.UnknownValueError:
    #     print("Speech recognition could not understand Tamil audio")
    # except sr.RequestError as e:
    #     print("Could not request Tamil results; {0}".format(e))