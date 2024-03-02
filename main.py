import json
import uuid

import requests
import uvicorn
from attr import dataclass
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Request
from fastapi.responses import Response
from pydantic import BaseModel
from fam.llm.fast_inference import TTS

tts = TTS()
app = FastAPI()

audio_id = {
    "11": "voices/male1.mp3",
    "12": "voices/male2.mp3",
    "21": "voices/female1.mp3",
    "22": "voices/female2.mp3"
}

# class ImageData(BaseModel):
#     file_image: dict

@app.post("/getUserId")
async def getUserId(file: UploadFile = File(...)):
    picpurify_url = 'https://www.picpurify.com/analyse/1.1'
    contents = await file.read()
    payload = {
        "API_KEY": "q2RrZaG7nhIboOiLXvtdjIQ3nw5gAM3p",
        "task": "face_gender_age_detection"
    }
    if not file.content_type.startswith("image"):
        return Response(
            content="Uploaded file is not an image.",
            status_code=400,
        )
    try:
        result_data = requests.post(picpurify_url, files={"file_image": contents},
                                    params=payload)
        result_data.raise_for_status()
        result_json = result_data.json()
        # tmp=json.dumps(result_json) #input: dict -> output: string
        # rs=json.loads(tmp) #input: string -> output: dict
        gender = result_json['face_detection']['results'][0]['gender']['decision']
        age = result_json['face_detection']['results'][0]['age_majority']['decision']
        user_id = 0
        match (gender, age):
            case ("male", "major"):
                user_id = 11
            case ("male", "minor"):
                user_id = 12
            case ("female", "major"):
                user_id = 21
            case ("female", "minor"):
                user_id = 22
        return {
            "user_id": user_id
        }
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error sending image to API: {str(e)}")

class TTSRequest(BaseModel):
    key: str
    user_id: str

@app.get("/tts")
async def text_to_speech(req: TTSRequest):
    userId = req.user_id
    audio_path = audio_id[userId]
    output_file = tts.synthesise(
        req.key,
        spk_ref_path= audio_path)
    return output_file
