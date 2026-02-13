from flask import Flask, request, send_file, jsonify
import os
import uuid
import re
from TTS.api import TTS
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

app = Flask(__name__)

# ---------------- FOLDERS ----------------
os.makedirs("temp", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("audio", exist_ok=True)

SPEAKER_MP3 = "audio/speaker.mp3"  # INPUT (you upload this)
SPEAKER_PROCESSED = "audio/speaker_clean.wav"  # XTTS uses this

MIN_SPEAKER_SECONDS = 600
MAX_CHARS_PER_CHUNK = 180


# ---------------- LOAD MODEL ----------------
print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("XTTS loaded")


# ---------------- AUDIO PREP ----------------
def prepare_speaker_wav():

    if not os.path.exists(SPEAKER_MP3):
        raise RuntimeError("speaker.mp3 not found inside /audio folder")

    # skip if already processed (Railway restart optimization)
    if os.path.exists(SPEAKER_PROCESSED):
        print("Speaker already processed")
        return

    print("Converting MP3 → WAV...")

    audio = AudioSegment.from_file(SPEAKER_MP3, format="mp3")

    # convert to mono 22050 (XTTS requirement)
    audio = audio.set_channels(1).set_frame_rate(22050)

    # remove long silence
    ns = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=-40)
    if ns:
        audio = audio[ns[0][0] : ns[-1][1]]

    # validate duration
    if len(audio) / 1000 < MIN_SPEAKER_SECONDS:
        raise RuntimeError("Speaker audio must be at least 10 minutes")

    # normalize volume
    gain = -18.0 - audio.dBFS
    audio = audio.apply_gain(gain)

    audio.export(SPEAKER_PROCESSED, format="wav")
    print("Speaker ready")


prepare_speaker_wav()


# ---------------- TEXT ----------------
def clean_hindi_text(text):
    text = re.sub(r"[,.]", " । ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith("।"):
        text += "।"
    return text


def split_text(text):
    sentences = re.split("।", text)
    chunks, current = [], ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(current) + len(s) < MAX_CHARS_PER_CHUNK:
            current += s + "। "
        else:
            chunks.append(current.strip())
            current = s + "। "

    if current:
        chunks.append(current.strip())

    return chunks


# ---------------- API ----------------
@app.route("/clone", methods=["POST"])
def clone():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "text missing"}), 400

    text = clean_hindi_text(data["text"])
    chunks = split_text(text)

    final_audio = AudioSegment.silent(duration=300)

    for i, chunk in enumerate(chunks):
        temp_path = f"temp/{uuid.uuid4()}.wav"

        tts.tts_to_file(
            text=chunk,
            speaker_wav=SPEAKER_PROCESSED,
            language="hi",
            speed=0.95,
            file_path=temp_path,
        )

        seg = AudioSegment.from_wav(temp_path)
        final_audio += seg + AudioSegment.silent(duration=150)

    out_name = f"{uuid.uuid4()}.wav"
    out_path = f"outputs/{out_name}"
    final_audio.export(out_path, format="wav")

    return send_file(out_path, mimetype="audio/wav")


# ---------------- HEALTH CHECK ----------------
@app.route("/")
def health():
    return "XTTS Server Running"
