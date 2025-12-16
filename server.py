#ssh -i /Users/scallercell_2/Desktop/cosyvoice root@69.30.85.218 -p 22169
#python3 server.py --port 8888 --model_dir ../../../pretrained_models/CosyVoice2-0.5B
#podid - https://hmwecuisc92c1a-8888.proxy.runpod.net

#ssh -i /Users/scallercell_2/Desktop/cosyvoice root@74.2.96.22 -p 15815 

#ssh -i /Users/scallercell_2/Desktop/cosyvoice root@69.30.85.167 -p 22049 
#ssh  -i /Users/scallercell_2/Desktop/cosyvoice root@38.147.83.16 -p 14576 

#ssh -i /Users/scallercell_2/Desktop/cosyvoice root@198.13.252.9 -p 11017 

#ssh -i /Users/scallercell_2/Desktop/cosyvoice root@69.30.85.191 -p 22129 

#ssh y38ge80rvfc9hr-64411d3c@ssh.runpod.io -i /Users/scallercell_2/Desktop/cosyvoice

#scp -p 18397 -i /Users/scallercell_2/Desktop/cosyvoice "/Users/scallercell_2/Downloads/silencevideo.mp4" root@198.13.252.9:/workspace

# cd /workspace
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# bash miniconda.sh -b -p /workspace/miniconda
#source /workspace/miniconda/etc/profile.d/conda.sh

#uvicorn upload_server:app --host 0.0.0.0 --port 8000


# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
# conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# conda create -y -n cosyvoice python=3.10
# conda activate cosyvoice

#echo "source /workspace/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc

# pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com


#chmod +x /workspace/start.sh
# bash /workspace/start.sh
# tail -50 /workspace/cosyvoice.log



#   curl -X POST "https://hmwecuisc92c1a-8888.proxy.runpod.net/tts_zero_shot" \
#   -F "tts_text=Hello from CosyVoice on Runpod" \
#   -F "prompt_text=希望你以后能够做的比我还好呦。" \
#   -F "prompt_wav=@/Users/scallercell_2/CosyVoice/asset/zero_shot_prompt.wav" \
#   --output result.wav

# curl -X POST "https://06pk065k5dkotm-8000.proxy.runpod.net/clone_voice" \
#   -F "text=Now let make" \
#   -F "language=EN" \
#   -F "reference_audio=@/Users/scallercell_2/Downloads/audiosample.wav" \
#   --output cloned_voice.wav

# curl -X POST "https://y38ge80rvfc9hr-8000.proxy.runpod.net/edit" \
#   -F "prompt=Move the camera to a top-down overhead view while keeping the scene consistent." \
#   -F "image_file=@/Users/scallercell_2/Downloads/input.png"




#  uvicorn server:app --host 0.0.0.0 --port 8000 --reload

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

import nltk   # <-- ADD THIS

# -------------------------------------------------------
# Auto-download required NLTK models if missing
# -------------------------------------------------------
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

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
    speaker_key = sorted(speaker_ids.keys())[0]
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
    - speed: TTS speed
    - reference_audio: sample voice file (wav/mp3/etc.)
    """

    global tone_color_converter

    if tone_color_converter is None:
        raise HTTPException(status_code=500, detail="ToneColorConverter not initialized.")

    # Normalize language
    language = language.upper()
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}",
        )

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # --- TEMP FILE PLACEHOLDERS ---
    ref_path = None
    src_path = None
    out_path = None

    try:
        # =======================================================
        # STEP 1 — Save uploaded reference audio
        # =======================================================
        suffix = os.path.splitext(reference_audio.filename or "")[1] or ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as ref_f:
            ref_path = ref_f.name
            ref_f.write(await reference_audio.read())

        # =======================================================
        # STEP 2 — Extract speaker embedding
        # =======================================================
        target_se, _ = se_extractor.get_se(
            ref_path,
            tone_color_converter,
            vad=True
        )

        # =======================================================
        # STEP 3 — Initialize Melo TTS for the selected language
        # =======================================================
        tts_model = TTS(language=language, device=DEVICE)

        # Base speaker mapping for this language
        speaker_id, source_se, speaker_key = get_source_se_for_language(language, tts_model)

        # =======================================================
        # STEP 4 — Generate intermediate base speaker audio
        # =======================================================
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as src_f:
            src_path = src_f.name

        tts_model.tts_to_file(
            text=text,
            speaker_id=speaker_id,
            output_path=src_path,
            speed=speed,
        )

        # =======================================================
        # STEP 5 — Convert tone into reference voice
        # =======================================================
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out_f:
            out_path = out_f.name

        tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=out_path,
            tau=1.0,
            message="",
        )

        # =======================================================
        # STEP 6 — Return WAV bytes
        # =======================================================
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
        # Cleanup
        for p in (ref_path, src_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass