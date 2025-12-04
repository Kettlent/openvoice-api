import io
import os
import tempfile
from typing import List

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

# ---------------------
# Config
# ---------------------
CKPT_CONVERTER_DIR = "checkpoints_v2/converter"
BASE_SE_DIR = "checkpoints_v2/base_speakers/ses"
OUTPUT_DIR = "outputs_v2"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SUPPORTED_LANGUAGES = [
    "EN_NEWEST",
    "EN",
    "ES",
    "FR",
    "ZH",
    "JP",
    "KR",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="OpenVoice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from typing import Optional

tone_color_converter: Optional[ToneColorConverter] = None
# ---------------------
# Startup: load converter
# ---------------------
@app.on_event("startup")
def load_openvoice():
    global tone_color_converter

    converter_config = os.path.join(CKPT_CONVERTER_DIR, "config.json")
    converter_ckpt = os.path.join(CKPT_CONVERTER_DIR, "checkpoint.pth")

    if not os.path.exists(converter_config) or not os.path.exists(converter_ckpt):
        raise RuntimeError(
            f"Converter checkpoints not found in {CKPT_CONVERTER_DIR}. "
            f"Expected config.json and checkpoint.pth."
        )

    tone_color_converter = ToneColorConverter(converter_config, device=DEVICE)
    tone_color_converter.load_ckpt(converter_ckpt)
    print(f"[OpenVoice] ToneColorConverter loaded on {DEVICE}")


# ---------------------
# Utility
# ---------------------
def get_source_se_for_language(language: str, model: TTS):
    """
    Pick a default base speaker for the given language and load its SE (.pth).
    You can later extend this to accept a 'base_speaker' parameter from the client.
    """
    speaker_ids = model.hps.data.spk2id
    if not speaker_ids:
        raise RuntimeError(f"No speakers found for language {language}")

    # pick the first speaker as default
    speaker_key = list(speaker_ids.keys())[0]
    speaker_id = speaker_ids[speaker_key]

    # the SES file names use lowercased, dash-separated keys
    speaker_key_norm = speaker_key.lower().replace("_", "-")
    se_path = os.path.join(BASE_SE_DIR, f"{speaker_key_norm}.pth")

    if not os.path.exists(se_path):
        raise RuntimeError(f"SE file not found for speaker '{speaker_key}' at {se_path}")

    source_se = torch.load(se_path, map_location=DEVICE)
    return speaker_id, source_se, speaker_key


# ---------------------
# Endpoints
# ---------------------
@app.get("/")
def root():
    return {
        "status": "openvoice server running",
        "device": DEVICE,
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.get("/languages")
def list_languages():
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/clone_voice")
async def clone_voice(
    text: str = Form(...),
    language: str = Form("EN_NEWEST"),
    speed: float = Form(1.0),
    reference_audio: UploadFile = File(...),
):
    """
    Text + reference voice → cloned voice WAV.
    - text: text to speak
    - language: one of SUPPORTED_LANGUAGES
    - speed: TTS speed (1.0 = normal)
    - reference_audio: mp3/wav/etc. of the voice to clone
    """
    global tone_color_converter

    if tone_color_converter is None:
        raise HTTPException(status_code=500, detail="ToneColorConverter not initialized.")

    language = language.upper()
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}",
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Save reference audio to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(reference_audio.filename or "")[1] or ".wav") as ref_f:
            ref_path = ref_f.name
            ref_f.write(await reference_audio.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save reference audio: {e}")

    src_path = None
    out_path = None

    try:
        # 1) Extract target speaker embedding from reference audio
        target_se, _ = se_extractor.get_se(
            ref_path, tone_color_converter, vad=True
        )

        # 2) Create Melo TTS model
        tts_model = TTS(language=language, device=DEVICE)

        # 3) Pick default base speaker and load its SE
        speaker_id, source_se, speaker_key = get_source_se_for_language(language, tts_model)

        # 4) Generate base speech to a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as src_f:
            src_path = src_f.name

        tts_model.tts_to_file(
            text=text,
            speaker_id=speaker_id,
            output_path=src_path,
            speed=speed,
        )

        # 5) Run tone color converter to match the target speaker
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_f:
            out_path = out_f.name

        encode_message = "@MyShell"
        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            message=encode_message,
        )

        # 6) Read final WAV and return as response
        with open(out_path, "rb") as f:
            audio_bytes = f.read()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="openvoice_clone.wav"'},
        )

    except Exception as e:
        print("[ERROR] clone_voice:", e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp files
        for p in (ref_path, src_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass