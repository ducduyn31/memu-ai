from gtts import gTTS #pip install gtts
import os
from playsound import playsound  #pip install playsound==1.2.2


text = 'Hi I want to buy meat'
language = 'en'

obj = gTTS(text=text, lang=language, slow=False)

obj.save("./gTTS_mp3/test.mp3")

if os.name == "posix":
    os.system("mpg321 ./gTTS_mp3/test.mp3") 
else: 
    playsound("./gTTS_mp3/test.mp3")

