import pyttsx3

text = "Hi I want to buy meat"
text_speech = pyttsx3.init()
text_speech.say(text)
text_speech.runAndWait()
