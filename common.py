"""Shared helpers for the workshop scripts.

You do not run this file directly. The other scripts import it.
"""

import os
import sys
from pathlib import Path

# Keep Hugging Face quiet and private. Must be set before transformers is imported.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Never crash on odd characters in an old Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                       # your own recordings
STARTER_DIR = ROOT / "starter_data"
STARTER_CLIPS = STARTER_DIR / "clips"          # synthetic starter clips
STARTER_ZIP = STARTER_DIR / "starter_clips.zip"
STARTER_FEATURES = STARTER_DIR / "starter_features.npz"
FEATURES_FILE = ROOT / "features.npz"
MODELS_DIR = ROOT / "models"
WHISPER_DIR = MODELS_DIR / "whisper-tiny"

STARTER_URL = ("https://github.com/cvmlarun/pda-turn-taking-workshop/"
               "releases/download/starter-v1/starter_clips.zip")
STARTER_MIN_CLIPS = 400

# ---------------------------------------------------------------- task
LABELS = ["respond", "keep_going", "stop"]
LABEL_HELP = {
    "respond": "Say a FULL sentence or question, then stay quiet.",
    "keep_going": "Stop in the MIDDLE of a sentence, as if you are thinking.",
    "stop": "Say a stop word: stop / wait / ruko / bas / thamba / nillisu.",
}
LANGS = {"kn": "Kannada", "hi": "Hindi", "mr": "Marathi", "en": "English"}

SR = 16000            # sample rate the model expects
CLIP_SECONDS = 3.0    # length of every recording
FEATURE_VERSION = "whisper-tiny-mean+last1s-v1"


# ---------------------------------------------------------------- audio
def load_wav(path):
    """Load any WAV file as 16 kHz mono float32."""
    import librosa
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio.astype(np.float32)


def save_wav(path, audio):
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), SR, subtype="PCM_16")


def record(seconds=CLIP_SECONDS):
    """Record from the default microphone. Returns 16 kHz mono float32."""
    import sounddevice as sd
    try:
        audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        return audio[:, 0].copy()
    except Exception:
        # Some microphones refuse 16 kHz. Record at their own rate and convert.
        import librosa
        dev_sr = int(sd.query_devices(kind="input")["default_samplerate"])
        audio = sd.rec(int(seconds * dev_sr), samplerate=dev_sr, channels=1, dtype="float32")
        sd.wait()
        return librosa.resample(audio[:, 0], orig_sr=dev_sr, target_sr=SR).astype(np.float32)


def play(audio):
    import sounddevice as sd
    try:
        sd.play(audio, SR)
        sd.wait()
    except Exception:
        import librosa
        dev_sr = int(sd.query_devices(kind="output")["default_samplerate"])
        sd.play(librosa.resample(audio, orig_sr=SR, target_sr=dev_sr), dev_sr)
        sd.wait()


def peak_level(audio):
    return float(np.max(np.abs(audio))) if len(audio) else 0.0


def has_speech(audio):
    """True if the clip has a clear loud part (speech) above the background.

    Compares the loud 20 ms frames with the quiet ones. Background noise is
    flat (ratio about 2). Speech is not (ratio 50+). Works for quiet and loud mics.
    """
    a = np.asarray(audio, dtype=np.float32)
    frames = a[: len(a) // 320 * 320].reshape(-1, 320)
    if len(frames) < 10 or peak_level(a) < 0.005:
        return False
    rms = np.sqrt((frames ** 2).mean(1)) + 1e-6
    return np.percentile(rms, 95) / np.percentile(rms, 10) > 4.0


# ---------------------------------------------------------------- whisper
def download_whisper():
    """Download whisper-tiny (about 150 MB) into models/whisper-tiny."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        "openai/whisper-tiny",
        local_dir=str(WHISPER_DIR),
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    )


def whisper_is_downloaded():
    return all((WHISPER_DIR / f).exists()
               for f in ["config.json", "preprocessor_config.json", "model.safetensors"])


class Embedder:
    """Frozen whisper-tiny encoder. Turns audio into a 768-number vector.

    The vector = [average over the whole clip, average over the last second].
    The last second matters most for turn-taking, so we give it its own slot.
    """

    def __init__(self):
        if not whisper_is_downloaded():
            raise SystemExit("whisper-tiny is not downloaded yet. Run: python setup_check.py")
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.torch = torch
        self.fe = WhisperFeatureExtractor.from_pretrained(str(WHISPER_DIR))
        model = WhisperModel.from_pretrained(str(WHISPER_DIR), use_safetensors=True)
        self.encoder = model.get_encoder().eval()
        del model
        for p in self.encoder.parameters():
            p.requires_grad = False   # frozen: we never train whisper

    def __call__(self, audios):
        """audios: list of 1-D float arrays at 16 kHz. Returns (n, 768) array."""
        torch = self.torch
        audios = [np.asarray(a, dtype=np.float32)[: SR * 30] for a in audios]
        inputs = self.fe(audios, sampling_rate=SR, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.encoder(inputs.input_features).last_hidden_state  # (n, 1500, 384)
        out = []
        for h, a in zip(hidden, audios):
            n = max(1, min(h.shape[0], int(np.ceil(len(a) / 320))))  # 1 frame = 20 ms
            whole = h[:n].mean(0)
            last = h[max(0, n - 50):n].mean(0)                       # last 1 s
            out.append(torch.cat([whole, last]).numpy())
        return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------- datasets
def list_clips():
    """All WAV files in starter_data/clips and data, as (path, label, lang)."""
    items = []
    for base in (STARTER_CLIPS, DATA_DIR):
        for label in LABELS:
            folder = base / label
            if folder.is_dir():
                for wav in sorted(folder.glob("*.wav")):
                    lang = wav.stem.split("_")[0].lower()
                    items.append((wav, label, lang if lang in LANGS else "other"))
    return items


def rel_key(path):
    """Same key on Windows and Linux, e.g. 'data/stop/kn_asha_1.wav'."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


def load_features_file(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def build_features(batch_size=8, verbose=True):
    """Embed every clip and save features.npz. Reuses cached results."""
    from tqdm import tqdm

    clips = list_clips()
    if not clips:
        raise SystemExit("No WAV files found. Run setup_check.py or record.py first.")

    # Reuse vectors we already computed (precomputed starter set + last run).
    cache = {}
    for f in (STARTER_FEATURES, FEATURES_FILE):
        if f.exists():
            try:
                d = load_features_file(f)
                if str(d.get("version", "")) != FEATURE_VERSION:
                    continue
                for key, size, vec in zip(d["files"], d["sizes"], d["X"]):
                    cache[(str(key), int(size))] = vec
            except Exception:
                pass

    keys = [rel_key(p) for p, _, _ in clips]
    sizes = [p.stat().st_size for p, _, _ in clips]
    vectors = [cache.get((k, s)) for k, s in zip(keys, sizes)]
    todo = [i for i, v in enumerate(vectors) if v is None]
    if verbose:
        print(f"Found {len(clips)} clips. {len(clips) - len(todo)} already done. "
              f"{len(todo)} to process.")

    if todo:
        if verbose:
            print("Loading whisper-tiny encoder ...")
        embed = Embedder()
        bar = tqdm(total=len(todo), desc="Extracting", unit="clip", ascii=True,
                   disable=not verbose)
        for start in range(0, len(todo), batch_size):
            idx = todo[start:start + batch_size]
            vecs = embed([load_wav(clips[i][0]) for i in idx])
            for i, v in zip(idx, vecs):
                vectors[i] = v
            bar.update(len(idx))
        bar.close()

    data = dict(
        X=np.stack(vectors).astype(np.float32),
        y=np.array([LABELS.index(l) for _, l, _ in clips], dtype=np.int64),
        labels=np.array(LABELS),
        langs=np.array([g for _, _, g in clips]),
        files=np.array(keys),
        sizes=np.array(sizes, dtype=np.int64),
        version=np.array(FEATURE_VERSION),
    )
    np.savez_compressed(FEATURES_FILE, **data)
    return data


def load_training_features():
    """features.npz if it exists, else the precomputed starter features."""
    for f in (FEATURES_FILE, STARTER_FEATURES):
        if f.exists():
            return load_features_file(f), f
    raise SystemExit("No features found. Run: python extract_features.py")


# ---------------------------------------------------------------- models
def make_mlp(in_dim, n_classes=len(LABELS), hidden=128):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden, n_classes),
    )


class Predictor:
    """Loads the model saved by train.py and predicts from raw audio."""

    def __init__(self):
        import json
        info_file = MODELS_DIR / "model_info.json"
        if not info_file.exists():
            raise SystemExit("No trained model yet. Run: python train.py")
        self.info = json.loads(info_file.read_text(encoding="utf-8"))
        self.kind = self.info["best"]
        self.embed = Embedder()
        if self.kind == "mlp":
            import torch
            ck = torch.load(MODELS_DIR / "mlp.pt", map_location="cpu", weights_only=True)
            self.mean, self.scale = ck["mean"].numpy(), ck["scale"].numpy()
            self.net = make_mlp(ck["in_dim"], hidden=ck["hidden"])
            self.net.load_state_dict(ck["state_dict"])
            self.net.eval()
        else:
            import joblib
            self.clf = joblib.load(MODELS_DIR / "logreg.joblib")

    def predict(self, audio):
        """Returns (label, confidence, {label: probability})."""
        x = self.embed([audio])
        if self.kind == "mlp":
            import torch
            with torch.inference_mode():
                logits = self.net(torch.from_numpy((x - self.mean) / self.scale))
                probs = torch.softmax(logits, -1)[0].numpy()
        else:
            probs = self.clf.predict_proba(x)[0]
        i = int(np.argmax(probs))
        return LABELS[i], float(probs[i]), {l: float(p) for l, p in zip(LABELS, probs)}
