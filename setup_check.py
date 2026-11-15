"""Check that your laptop is ready for the workshop.

Run:  python setup_check.py

It checks Python, the packages, downloads the model and the starter data,
and tests your microphone and speakers. Needs internet the first time.
After it passes once, everything works offline.
"""

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "setup_check.log"
results = []


def log_error():
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(traceback.format_exc() + "\n")


def report(name, ok, msg=""):
    results.append(ok)
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"{tag} {name}" + (f" - {msg}" if msg else ""))


def step(title):
    print()
    print(f"--- {title} ---")


LOG.write_text("setup_check.py log\n", encoding="utf-8")
print("Workshop setup check. This can take a few minutes the first time.")

# 1. Python version ------------------------------------------------------
step("1. Python version")
v = sys.version_info
if (v.major, v.minor) == (3, 11):
    report("Python version", True, f"{v.major}.{v.minor}.{v.micro}")
elif (v.major, v.minor) in [(3, 10), (3, 12)]:
    report("Python version", True, f"{v.major}.{v.minor} works, but 3.11 is recommended")
else:
    report("Python version", False,
           f"you have {v.major}.{v.minor}. Install Python 3.11 from python.org.")
if sys.prefix == sys.base_prefix:
    print("       Note: the virtual environment is NOT active. Run .venv\\Scripts\\activate first.")

# 2. Packages ------------------------------------------------------------
step("2. Packages")
PACKAGES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"), ("librosa", "librosa"), ("soundfile", "soundfile"),
    ("sounddevice", "sounddevice"), ("torch", "torch"), ("torchaudio", "torchaudio"),
    ("transformers", "transformers"), ("tqdm", "tqdm"), ("jupyter", "notebook"),
]
packages_ok = True
for pip_name, mod in PACKAGES:
    try:
        m = __import__(mod)
        report(f"import {pip_name}", True, getattr(m, "__version__", ""))
    except Exception as e:
        log_error()
        packages_ok = False
        hint = "Run: python -m pip install -r requirements.txt"
        if pip_name in ("torch", "torchaudio"):
            hint = ("Run: python -m pip install torch==2.5.1 torchaudio==2.5.1 "
                    "--index-url https://download.pytorch.org/whl/cpu")
        if pip_name == "sounddevice" and isinstance(e, OSError):
            hint = "The audio library is missing. Reinstall: python -m pip install --force-reinstall sounddevice"
        report(f"import {pip_name}", False, f"not installed or broken. {hint}")

try:
    import torch
    if torch.cuda.is_available() or "+cu" in torch.__version__:
        print("       Note: you have the GPU version of torch. It works, but it is much bigger.")
except Exception:
    pass

if not packages_ok:
    print()
    print("Some packages are missing. Fix them and run this check again.")
    print(f"STOPPED: {results.count(False)} check(s) failed.")
    sys.exit(1)

import numpy as np  # noqa: E402

import common  # noqa: E402

# 3. Model ---------------------------------------------------------------
step("3. whisper-tiny model (about 150 MB)")
try:
    if common.whisper_is_downloaded():
        print("Already downloaded.")
    else:
        print("Downloading ... please wait.")
        common.download_whisper()
    emb = common.Embedder()
    vec = emb([np.zeros(common.SR, dtype=np.float32)])
    assert vec.shape == (1, 768)
    report("whisper-tiny", True, f"saved in {common.WHISPER_DIR}")
except Exception as e:
    log_error()
    msg = "download failed. Check your internet (college Wi-Fi may block it; try a phone hotspot)."
    if "No space" in str(e):
        msg = "your disk is full. Free at least 3 GB."
    report("whisper-tiny", False, msg)

# 4. Starter data --------------------------------------------------------
step("4. Starter dataset")
try:
    def count():
        return len(list(common.STARTER_CLIPS.rglob("*.wav")))

    if count() >= common.STARTER_MIN_CLIPS:
        print("Already downloaded.")
    else:
        import urllib.request
        import zipfile
        if not common.STARTER_ZIP.exists():
            print("Downloading starter_clips.zip (about 35 MB) ...")
            tmp = common.STARTER_ZIP.with_suffix(".part")
            urllib.request.urlretrieve(common.STARTER_URL, tmp)
            tmp.replace(common.STARTER_ZIP)
        with zipfile.ZipFile(common.STARTER_ZIP) as z:
            z.extractall(common.STARTER_DIR)
    n = count()
    if n < common.STARTER_MIN_CLIPS:
        raise RuntimeError(f"only {n} clips")
    report("starter dataset", True, f"{n} clips in {common.STARTER_CLIPS}")
    report("precomputed features", common.STARTER_FEATURES.exists(),
           "" if common.STARTER_FEATURES.exists()
           else "starter_data/starter_features.npz is missing. Run: git pull")
except Exception:
    log_error()
    common.STARTER_ZIP.unlink(missing_ok=True)
    report("starter dataset", False,
           "download failed. Check internet, or copy starter_clips.zip from the "
           "coordinator into the starter_data folder and run this again.")

# 5. Microphone and speakers --------------------------------------------
step("5. Microphone and speakers")
try:
    import sounddevice as sd
    try:
        mic = sd.query_devices(kind="input")["name"]
    except Exception:
        mic = None
    if not mic:
        report("microphone", False,
               "no microphone found. Plug one in, or allow mic access in Windows "
               "Settings > Privacy & security > Microphone.")
    else:
        print(f"Microphone: {mic}")
        input("Press Enter, then say 'hello, testing one two three' ...")
        print("Recording for 3 seconds ... speak now!")
        audio = common.record(3.0)
        level = common.peak_level(audio)
        print(f"Recording done. Loudness: {level:.2f}")
        if not common.has_speech(audio):
            report("microphone", False,
                   "no voice heard - the recording is silent or only noise. Speak louder. Allow mic access in Windows Settings > "
                   "Privacy & security > Microphone (turn on 'Let desktop apps access "
                   "your microphone'). Check the mic is not muted.")
        else:
            report("microphone", True, f"loudness {level:.2f}")
            common.save_wav(ROOT / "mic_test.wav", audio)
            print("Playing it back now ...")
            common.play(audio)
            heard = input("Did you hear your voice? Type y or n, then Enter: ").strip().lower()
            if heard.startswith("y"):
                report("speakers", True)
            else:
                report("speakers", False,
                       "turn up the volume or plug in earphones. "
                       "Your recording is saved as mic_test.wav if you want to check it.")
except EOFError:
    report("microphone", False, "run this check in a normal Command Prompt window.")
except Exception:
    log_error()
    report("microphone", False,
           "could not open the microphone. Close Zoom/Teams/Meet, allow mic access "
           "in Windows privacy settings, and try again.")

# Summary ----------------------------------------------------------------
print()
print("=" * 50)
failed = results.count(False)
if failed == 0:
    print("ALL CHECKS PASSED")
    print("Take a screenshot of this window and send it to the coordinator.")
else:
    print(f"{failed} CHECK(S) FAILED. Read the [FAIL] lines above.")
    print("See the Troubleshooting section in README.md.")
    print(f"If you are stuck, send setup_check.log to the coordinator.")
print("=" * 50)
sys.exit(0 if failed == 0 else 1)
