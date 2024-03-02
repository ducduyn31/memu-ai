import requests
from fastapi import Response
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fam.llm.fast_inference import TTS



tts = TTS()
app = FastAPI()

audio_id = {
    "11": "male1.mp3",
    "12": "male2.mp3",
    "21": "female1.mp3",
    "22": "female2.mp3"
}

class TTSRequest(BaseModel):
    key: str
    user_id: str

@app.post("/tts", response_class=Response)
async def text_to_speech(req: TTSRequest):
    userId = req.user_id
    audio_path = audio_id[userId]
    output_file = tts.synthesise(
        req.key,
        spk_ref_path= audio_path)
    return output_file
    