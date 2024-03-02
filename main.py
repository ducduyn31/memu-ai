import uuid

from attr import dataclass
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import Response

app = FastAPI()


@dataclass(frozen=True)
class TTSRequest:
    text: str


@app.post("/getUserId")
def read_root():
    # Face     Recognition     model -> Used     to     identify     which
    # voice     to     use

    # Generate     voice     embedding and load
    # to     GPU     first

    # Return  user   id
    return {"userId": uuid.uuid4()}


@app.post("tts")
async def text_to_speech(req: Request):
    audiodata = await req.body()
    payload = None
    wav_out_path = None

    # req.headers.get("X-UserID")

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
