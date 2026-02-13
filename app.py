from flask import Flask, request, send_file, jsonify
import os, uuid, re, threading, sys, traceback

app = Flask(__name__)


# ---------------- GLOBAL CRASH LOGGER ----------------
def log_exceptions(type, value, tb):
    traceback.print_exception(type, value, tb)


sys.excepthook = log_exceptions


# ---------------- PATHS ----------------
TEMP = "temp"
OUT = "outputs"
AUDIO = "audio"

os.makedirs(TEMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(AUDIO, exist_ok=True)

SPEAKER_MP3 = f"{AUDIO}/speaker.mp3"
SPEAKER_PROCESSED = f"{AUDIO}/speaker_clean.wav"

MIN_SPEAKER_SECONDS = 600
MAX_CHARS_PER_CHUNK = 180

# ---------------- GLOBAL MODEL ----------------
tts_model = None
model_lock = threading.Lock()


# ---------------- MODEL LOADER ----------------
def load_model():
    global tts_model
    if tts_model is None:
        from TTS.api import TTS

        print("Loading XTTS model (CPU mode)...")
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        print("XTTS ready")


# background preload (non-blocking)
def warmup():
    try:
        load_model()
    except Exception as e:
        print("Model warmup failed:", e)


# ---------------- AUDIO PREP ----------------
def prepare_speaker():
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    if not os.path.exists(SPEAKER_MP3):
        print("speaker.mp3 missing — waiting for upload")
        return

    if os.path.exists(SPEAKER_PROCESSED):
        return

    print("Preparing speaker audio...")
    audio = AudioSegment.from_file(SPEAKER_MP3, format="mp3")
    audio = audio.set_channels(1).set_frame_rate(22050)

    ns = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=-40)
    if ns:
        audio = audio[ns[0][0] : ns[-1][1]]

    if len(audio) / 1000 < MIN_SPEAKER_SECONDS:
        raise RuntimeError("Speaker audio must be 10 minutes")

    gain = -18.0 - audio.dBFS
    audio = audio.apply_gain(gain)

    audio.export(SPEAKER_PROCESSED, format="wav")
    print("Speaker prepared")


# ---------------- TEXT ----------------
def clean_text(text):
    text = re.sub(r"[,.]", " । ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith("।"):
        text += "।"
    return text


def split_text(text):
    sentences = re.split("।", text)
    chunks, cur = [], ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) < MAX_CHARS_PER_CHUNK:
            cur += s + "। "
        else:
            chunks.append(cur.strip())
            cur = s + "। "

    if cur:
        chunks.append(cur.strip())
    return chunks


# ---------------- ROUTES ----------------
@app.route("/")
def health():
    return "OK"


@app.route("/clone", methods=["POST"])
def clone():

    from pydub import AudioSegment

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "text missing"}), 400

    if not os.path.exists(SPEAKER_PROCESSED):
        prepare_speaker()
        if not os.path.exists(SPEAKER_PROCESSED):
            return jsonify({"error": "upload speaker.mp3 first"}), 400

    with model_lock:
        load_model()

    text = clean_text(data["text"])
    chunks = split_text(text)

    final_audio = AudioSegment.silent(duration=300)

    for chunk in chunks:
        temp = f"{TEMP}/{uuid.uuid4()}.wav"

        tts_model.tts_to_file(
            text=chunk,
            speaker_wav=SPEAKER_PROCESSED,
            language="hi",
            speed=0.95,
            file_path=temp,
        )

        seg = AudioSegment.from_wav(temp)
        final_audio += seg + AudioSegment.silent(duration=150)

    out = f"{OUT}/{uuid.uuid4()}.wav"
    final_audio.export(out, format="wav")

    return send_file(out, mimetype="audio/wav")


# start background model preload
threading.Thread(target=warmup, daemon=True).start()


# ---------------- START ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
