"""Turn every WAV clip into a whisper-tiny embedding.

Run:  python extract_features.py

Reads   starter_data/clips/<label>/*.wav  and  data/<label>/*.wav
Writes  features.npz

Clips that were already processed are skipped, so running it again is fast.
The starter clips come with precomputed features, so only your own
recordings need the model.
"""

import time

import numpy as np

import common


def main():
    t0 = time.time()
    data = common.build_features()
    X, y, langs = data["X"], data["y"], data["langs"]
    print()
    print(f"Saved {common.FEATURES_FILE.name}: {X.shape[0]} clips x {X.shape[1]} numbers "
          f"({time.time() - t0:.0f} s)")
    print("Clips per label:")
    for i, label in enumerate(common.LABELS):
        per_lang = ", ".join(f"{g}={int(np.sum((y == i) & (langs == g)))}"
                             for g in sorted(set(langs)))
        print(f"  {label:<11} {int(np.sum(y == i)):4d}   ({per_lang})")
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
