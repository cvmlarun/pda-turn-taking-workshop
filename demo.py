"""Live demo: speak into the mic and see what the model thinks.

Run:  python demo.py

Each round records 3 seconds, then prints:
  respond    - you finished, the assistant should answer now
  keep_going - you paused mid-sentence, the assistant should wait
  stop       - you told the assistant to stop talking
"""

import common


def bar(p, width=20):
    n = int(round(p * width))
    return "#" * n + "." * (width - n)


def main():
    print("Loading model ...")
    model = common.Predictor()
    print(f"Using the {model.kind} model "
          f"(test accuracy {model.info[model.kind + '_accuracy'] * 100:.0f}%).")
    print()
    print("Try: a full question, a half sentence with a pause, or 'stop'.")
    while True:
        cmd = input("\nPress Enter and speak for 3 seconds (q = quit): ").strip().lower()
        if cmd == "q":
            break
        print(">>> LISTENING ... <<<")
        audio = common.record(common.CLIP_SECONDS)
        if not common.has_speech(audio):
            print("I heard only silence. Speak louder or check your mic.")
            continue
        label, conf, probs = model.predict(audio)
        print(f"Prediction: {label.upper()}  ({conf * 100:.0f}% sure)")
        for l in common.LABELS:
            print(f"  {l:<11} {bar(probs[l])} {probs[l] * 100:5.1f}%")
    print("Bye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
