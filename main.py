import json
import uuid

import requests
import uvicorn
from attr import dataclass
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi import Request
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

# class ImageData(BaseModel):
#     file_image: dict

@app.post("/getUserId")
async def getUserId(file: UploadFile = File(...)):
    # Face     Recognition     model -> Used     to     identify     which
    # voice     to     use
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
        return Response(
            content=str({"userId": user_id}),
            status_code=200,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error sending image to API: {str(e)}")

    # Generate     voice     embedding and load
    # to     GPU     first


@app.get("/tts")
async def text_to_speech(req: Request):
    audiodata = await req.body()
    payload = None
    wav_out_path = None

    # userId = req.headers.get("X-UserID")

    # try:
    #     headers = req.headers
    #     payload = headers["X-Payload"]
    #     payload = json.loads(payload)
    #     tts_req = TTSRequest(**payload)

    # Get the matched referecent path from the user id

    # voice_reference_path = get_voice_reference_path(user_id)

    #     wav_out_path = GlobalState.tts.synthesise(
    #         text=tts_req.text,
    #         # spk_ref_path=voice_reference_path,
    #         top_p=tts_req.top_p,
    #         guidance_scale=tts_req.guidance,
    #     )
    #
    #     with open(wav_out_path, "rb") as f:
    #         return Response(content=f.read(), media_type="audio/wav")
    # except Exception as e:
    #     # traceback_str = "".join(traceback.format_tb(e.__traceback__))
    #     logger.exception(f"Error processing request {payload}")
    #     return Response(
    #         content="Something went wrong. Please try again in a few mins or contact us on Discord",
    #         status_code=500,
    #     )
    # finally:
    #     if wav_out_path is not None:
    #         Path(wav_out_path).unlink(missing_ok=True
    # return Response(content=f.read(), media_type="audio/wav")
    return Response(
        content="Something went wrong. Please try again in a few mins or contact us on Discord",
        status_code=500,
    )
