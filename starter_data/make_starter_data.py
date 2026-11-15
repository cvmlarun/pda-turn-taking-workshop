"""Make the synthetic starter dataset (for the workshop coordinator).

Students do NOT need to run this. setup_check.py downloads the result.

It uses Microsoft Edge text-to-speech (needs internet and one extra package):
    python -m pip install edge-tts
    python starter_data/make_starter_data.py

How the three labels are made from TTS audio:
  respond    - a full sentence, then a short silence (the speaker is done).
  keep_going - the SAME sentence cut at a word in the middle, then silence
               (the speaker paused mid-thought and will continue).
  stop       - a short stop command ("stop", "ruko", "thamba", "nillisu" ...).
Every clip gets the same random pause length (0.3-0.8 s), length (2.5-3.0 s)
and light background noise, so the model cannot cheat on those.

Output:
  starter_data/clips/<label>/<lang>_tts<voice>_<n>.wav
  starter_data/starter_clips.zip      (upload to the GitHub release)
  starter_data/starter_features.npz   (precomputed whisper-tiny features)
"""

import asyncio
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import common  # noqa: E402

try:
    import edge_tts
except ImportError:
    raise SystemExit("Please install edge-tts first:  python -m pip install edge-tts")

VOICES = {
    "kn": ["kn-IN-GaganNeural", "kn-IN-SapnaNeural"],
    "hi": ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"],
    "mr": ["mr-IN-ManoharNeural", "mr-IN-AarohiNeural"],
    "en": ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"],
}
RATES = ["-10%", "+5%"]

SENTENCES = {
    "en": [
        "Can you tell me the weather in Bangalore today?",
        "I want to book a train ticket to Mumbai for tomorrow morning.",
        "What is the best way to learn programming quickly?",
        "My laptop is running very slowly since yesterday evening.",
        "Please remind me to call my mother at six o'clock.",
        "How many students are coming to the workshop this afternoon?",
        "I was thinking that we could meet after the lecture.",
        "Could you explain how a neural network actually learns?",
        "The canteen food was really good today, especially the dosa.",
        "Which bus goes from the college to the railway station?",
    ],
    "hi": [
        "क्या आप मुझे आज बेंगलुरु का मौसम बता सकते हैं?",
        "मुझे कल सुबह मुंबई के लिए ट्रेन का टिकट बुक करना है।",
        "प्रोग्रामिंग जल्दी सीखने का सबसे अच्छा तरीका क्या है?",
        "मेरा लैपटॉप कल शाम से बहुत धीरे चल रहा है।",
        "कृपया मुझे छह बजे माँ को फोन करने की याद दिलाना।",
        "आज दोपहर की कार्यशाला में कितने छात्र आ रहे हैं?",
        "मैं सोच रहा था कि हम लेक्चर के बाद मिल सकते हैं।",
        "क्या आप समझा सकते हैं कि न्यूरल नेटवर्क कैसे सीखता है?",
        "आज कैंटीन का खाना बहुत अच्छा था, खासकर डोसा।",
        "कॉलेज से रेलवे स्टेशन तक कौन सी बस जाती है?",
    ],
    "mr": [
        "तुम्ही मला आज बेंगळुरूचे हवामान सांगू शकता का?",
        "मला उद्या सकाळी मुंबईसाठी रेल्वेचे तिकीट बुक करायचे आहे.",
        "प्रोग्रामिंग लवकर शिकण्याचा सर्वात चांगला मार्ग कोणता आहे?",
        "माझा लॅपटॉप कालपासून खूप हळू चालत आहे.",
        "कृपया मला सहा वाजता आईला फोन करण्याची आठवण करून द्या.",
        "आज दुपारच्या कार्यशाळेला किती विद्यार्थी येणार आहेत?",
        "मला वाटत होतं की आपण लेक्चरनंतर भेटू शकतो.",
        "न्यूरल नेटवर्क नेमकं कसं शिकतं ते तुम्ही समजावून सांगाल का?",
        "आज कॅन्टीनचं जेवण खूप छान होतं, विशेषतः डोसा.",
        "कॉलेजपासून रेल्वे स्टेशनपर्यंत कोणती बस जाते?",
    ],
    "kn": [
        "ಇವತ್ತು ಬೆಂಗಳೂರಿನ ಹವಾಮಾನ ಹೇಗಿದೆ ಎಂದು ಹೇಳುತ್ತೀರಾ?",
        "ನಾಳೆ ಬೆಳಿಗ್ಗೆ ಮುಂಬೈಗೆ ರೈಲು ಟಿಕೆಟ್ ಬುಕ್ ಮಾಡಬೇಕು.",
        "ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಬೇಗ ಕಲಿಯಲು ಉತ್ತಮ ದಾರಿ ಯಾವುದು?",
        "ನನ್ನ ಲ್ಯಾಪ್‌ಟಾಪ್ ನಿನ್ನೆ ಸಂಜೆಯಿಂದ ತುಂಬಾ ನಿಧಾನವಾಗಿ ಓಡುತ್ತಿದೆ.",
        "ದಯವಿಟ್ಟು ಆರು ಗಂಟೆಗೆ ಅಮ್ಮನಿಗೆ ಕರೆ ಮಾಡಲು ನೆನಪಿಸಿ.",
        "ಇವತ್ತು ಮಧ್ಯಾಹ್ನದ ಕಾರ್ಯಾಗಾರಕ್ಕೆ ಎಷ್ಟು ವಿದ್ಯಾರ್ಥಿಗಳು ಬರುತ್ತಿದ್ದಾರೆ?",
        "ಲೆಕ್ಚರ್ ಆದ ಮೇಲೆ ನಾವು ಭೇಟಿಯಾಗಬಹುದು ಎಂದು ಯೋಚಿಸುತ್ತಿದ್ದೆ.",
        "ನ್ಯೂರಲ್ ನೆಟ್‌ವರ್ಕ್ ಹೇಗೆ ಕಲಿಯುತ್ತದೆ ಎಂದು ವಿವರಿಸುತ್ತೀರಾ?",
        "ಇವತ್ತು ಕ್ಯಾಂಟೀನ್ ಊಟ ತುಂಬಾ ಚೆನ್ನಾಗಿತ್ತು, ಅದರಲ್ಲೂ ದೋಸೆ.",
        "ಕಾಲೇಜಿನಿಂದ ರೈಲು ನಿಲ್ದಾಣಕ್ಕೆ ಯಾವ ಬಸ್ ಹೋಗುತ್ತದೆ?",
    ],
}

STOP_PHRASES = {
    "en": ["Stop.", "Wait.", "Hold on.", "Stop talking.", "One second.",
           "Never mind.", "Okay, stop.", "Cancel.", "Be quiet.", "No no, stop."],
    "hi": ["रुको।", "बस करो।", "रुक जाओ।", "एक मिनट।", "चुप हो जाओ।",
           "ठहरो।", "बंद करो।", "रहने दो।", "नहीं, रुको।", "बस।"],
    "mr": ["थांबा.", "थांब.", "बस्स.", "पुरे.", "एक मिनिट.",
           "थांबव.", "बंद कर.", "नको, थांबा.", "राहू दे.", "गप्प बस."],
    "kn": ["ನಿಲ್ಲಿಸು.", "ನಿಲ್ಲಿ.", "ಸಾಕು.", "ಒಂದು ನಿಮಿಷ.", "ಸುಮ್ಮನಿರು.",
           "ಬೇಡ.", "ಸಾಕು, ನಿಲ್ಲಿಸು.", "ತಡಿ.", "ಬೇಡ, ನಿಲ್ಲಿಸಿ.", "ಸ್ವಲ್ಪ ತಡೆಯಿರಿ."],
}

SR = common.SR
rng = np.random.default_rng(1234)


async def tts(text, voice, rate):
    """Returns (audio at 16 kHz, list of (start_s, end_s) for each word)."""
    import librosa
    import soundfile as sf
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    mp3, words = b"", []
    for attempt in range(4):
        try:
            mp3, words = b"", []
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3 += chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    s = chunk["offset"] / 1e7
                    words.append((s, s + chunk["duration"] / 1e7))
            break
        except Exception as e:  # network hiccup: try again
            print("  retry:", e)
            await asyncio.sleep(2 + attempt * 3)
            comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    audio, sr = sf.read(io.BytesIO(mp3), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32), words


def make_clip(audio, speech_end_s, speech_start_s=0.0):
    """Clip that ends `pause` seconds after speech_end, total 2.5-3.0 s."""
    length = rng.uniform(2.5, 3.0)
    pause = rng.uniform(0.3, 0.8)
    end = int(speech_end_s * SR)
    start = max(int(speech_start_s * SR), end - int((length - pause) * SR))
    speech = audio[start:end]
    # small fade so the cut does not click
    fade = min(len(speech), int(0.02 * SR))
    if fade:
        speech[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    total = int(length * SR)
    clip = np.zeros(total, np.float32)
    lead = total - int(pause * SR) - len(speech)
    clip[max(0, lead):max(0, lead) + len(speech)] = speech[max(0, -lead):]
    # random loudness + light background noise (SNR about 30-45 dB)
    clip *= rng.uniform(0.3, 0.9) / (np.max(np.abs(clip)) + 1e-9)
    rms = np.sqrt(np.mean(clip ** 2)) + 1e-9
    clip += rng.normal(0, rms / (10 ** (rng.uniform(30, 45) / 20)), total).astype(np.float32)
    return np.clip(clip, -1, 1)


async def main():
    counts = {l: 0 for l in common.LABELS}
    for lang in VOICES:
        for vi, voice in enumerate(VOICES[lang]):
            vname = "tts" + voice.split("-")[2].replace("Neural", "").lower()
            n = {l: 0 for l in common.LABELS}

            def save(label, clip):
                n[label] += 1
                counts[label] += 1
                common.save_wav(common.STARTER_CLIPS / label / f"{lang}_{vname}_{n[label]}.wav", clip)

            print(f"{lang} {voice}")
            for text in SENTENCES[lang]:
                for rate in RATES:
                    audio, words = await tts(text, voice, rate)
                    if len(words) < 4:
                        print("  skipped (too few words):", rate)
                        continue
                    first, last = words[0][0], words[-1][1]
                    save("respond", make_clip(audio.copy(), last + 0.05, first - 0.05))
                    # cut after a word between 40% and 75% of the sentence
                    lo = max(1, int(len(words) * 0.4))
                    hi = max(lo, min(len(words) - 2, int(len(words) * 0.75)))
                    k = int(rng.integers(lo, hi + 1))
                    save("keep_going", make_clip(audio.copy(), words[k][1] + 0.03, first - 0.05))
            for text in STOP_PHRASES[lang]:
                for rate in RATES:
                    audio, words = await tts(text, voice, rate)
                    if not words:
                        continue
                    save("stop", make_clip(audio.copy(), words[-1][1] + 0.05, words[0][0] - 0.05))
    print("clips per label:", counts)


if __name__ == "__main__":
    asyncio.run(main())

    print("Zipping ...")
    common.STARTER_ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(common.STARTER_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for wav in sorted(common.STARTER_CLIPS.rglob("*.wav")):
            z.write(wav, wav.relative_to(common.STARTER_DIR).as_posix())
    print("Wrote", common.STARTER_ZIP)

    print("Computing precomputed features ...")
    # Only starter clips: move features.npz aside if a student set exists.
    common.STARTER_FEATURES.unlink(missing_ok=True)
    saved_data_dir = common.DATA_DIR
    common.DATA_DIR = common.ROOT / "__no_data__"
    backup = common.FEATURES_FILE.with_suffix(".npz.bak")
    if common.FEATURES_FILE.exists():
        common.FEATURES_FILE.replace(backup)
    try:
        common.build_features()
        common.FEATURES_FILE.replace(common.STARTER_FEATURES)
    finally:
        common.DATA_DIR = saved_data_dir
        if backup.exists():
            backup.replace(common.FEATURES_FILE)
    print("Wrote", common.STARTER_FEATURES)
