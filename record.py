"""Record your own training clips.

Run:  python record.py

Each clip is 3 seconds, 16 kHz, mono. Clips are saved as
data/<label>/<lang>_<name>_<n>.wav
"""

import re

import common


def choose(title, options):
    print()
    print(title)
    for i, (key, text) in enumerate(options, 1):
        print(f"  {i}) {key:<11} {text}")
    while True:
        ans = input("Type a number and press Enter: ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return options[int(ans) - 1][0]
        print("Please type one of the numbers shown.")


def next_number(folder, prefix):
    nums = [int(m.group(1)) for f in folder.glob(prefix + "*.wav")
            if (m := re.fullmatch(re.escape(prefix) + r"(\d+)", f.stem))]
    return max(nums, default=0) + 1


def main():
    print("=== Record training clips ===")
    print("Tip: sit in a quiet place. Keep the mic about 20 cm from your mouth.")

    name = ""
    while not name:
        name = re.sub(r"[^a-z0-9]", "", input("\nYour first name (English letters): ").lower())
    lang = choose("Which language will you speak?", list(common.LANGS.items()))
    label = choose("What kind of clip?", list(common.LABEL_HELP.items()))
    last = None

    while True:
        folder = common.DATA_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{lang}_{name}_"
        n = next_number(folder, prefix)
        print()
        print(f"Label: {label}  |  Language: {common.LANGS[lang]}  |  Next clip: #{n}")
        print(f"  {common.LABEL_HELP[label]}")
        cmd = input("Enter = record   p = play last   d = delete last   "
                    "l = change label   q = quit : ").strip().lower()
        if cmd == "q":
            break
        if cmd == "l":
            label = choose("What kind of clip?", list(common.LABEL_HELP.items()))
            continue
        if cmd == "p":
            if last and last.exists():
                common.play(common.load_wav(last))
            else:
                print("Nothing recorded yet.")
            continue
        if cmd == "d":
            if last and last.exists():
                last.unlink()
                print(f"Deleted {last.name}")
                last = None
            else:
                print("Nothing to delete.")
            continue

        print(">>> RECORDING (3 seconds) ... speak now <<<")
        audio = common.record(common.CLIP_SECONDS)
        level = common.peak_level(audio)
        if not common.has_speech(audio):
            print("I did not hear your voice. Check your mic and try again. (Not saved.)")
            continue
        last = folder / f"{prefix}{n}.wav"
        common.save_wav(last, audio)
        warn = "  (very loud - move the mic back a little)" if level > 0.99 else ""
        print(f"Saved {last.relative_to(common.ROOT)}  loudness {level:.2f}{warn}")

    total = {l: len(list((common.DATA_DIR / l).glob("*.wav"))) for l in common.LABELS}
    print()
    print("Your clips so far:", total)
    print("Next step: python extract_features.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
