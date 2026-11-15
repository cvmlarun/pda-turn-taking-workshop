"""Train two small classifiers on the whisper-tiny features.

Run:  python train.py

1. Logistic regression (scikit-learn) - the simple baseline.
2. A tiny neural network (PyTorch MLP) - one hidden layer.

Both are tested on 20% of the clips that they never saw during training.
The better one is saved to models/ and used by demo.py.
"""

import json
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import common  # noqa: E402

LABELS = common.LABELS
SEED = 42


def show_results(name, y_true, y_pred, langs):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{name}: test accuracy = {acc * 100:.1f}%")
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    table = pd.DataFrame(cm, index=[f"true {l}" for l in LABELS],
                         columns=[f"pred {l}" for l in LABELS])
    print("Confusion matrix (rows = truth, columns = prediction):")
    print(table.to_string())
    per_lang = {g: accuracy_score(y_true[langs == g], y_pred[langs == g])
                for g in sorted(set(langs))}
    print("Accuracy per language: " +
          ", ".join(f"{g} {a * 100:.0f}%" for g, a in per_lang.items()))
    return acc, cm


def train_mlp(X_tr, y_tr, X_te, y_te, epochs=300, hidden=128):
    torch.manual_seed(SEED)
    mean, scale = X_tr.mean(0), X_tr.std(0) + 1e-6
    xt = torch.from_numpy((X_tr - mean) / scale)
    yt = torch.from_numpy(y_tr)
    xv = torch.from_numpy((X_te - mean) / scale)
    net = common.make_mlp(X_tr.shape[1], hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 64):          # mini-batches of 64
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        losses.append(float(loss))
        if epoch % 50 == 0:
            net.eval()
            with torch.no_grad():
                acc = (net(xv).argmax(1).numpy() == y_te).mean()
            print(f"  epoch {epoch:3d}  loss {losses[-1]:.3f}  test acc {acc * 100:.1f}%")
    net.eval()
    with torch.no_grad():
        pred = net(xv).argmax(1).numpy()
    return net, mean, scale, pred, losses


def save_confusion_plot(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    t0 = time.time()
    data, source = common.load_training_features()
    X, y, langs = data["X"], data["y"], data["langs"]
    print(f"Loaded {len(y)} clips from {source.name}")
    if source == common.STARTER_FEATURES:
        print("(Using the precomputed starter features. Run extract_features.py "
              "to add your own clips.)")
    print(pd.crosstab(pd.Series([LABELS[i] for i in y], name="label"),
                      pd.Series(langs, name="lang")).to_string())

    counts = np.bincount(y, minlength=len(LABELS))
    if counts.min() < 5:
        raise SystemExit("Each label needs at least 5 clips. Record more with record.py.")

    X_tr, X_te, y_tr, y_te, l_tr, l_te = train_test_split(
        X, y, langs, test_size=0.2, stratify=y, random_state=SEED)
    print(f"\nTraining on {len(y_tr)} clips, testing on {len(y_te)} clips.")

    # 1. Logistic regression
    print("\n=== 1. Logistic regression ===")
    logreg = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    logreg.fit(X_tr, y_tr)
    lr_acc, lr_cm = show_results("Logistic regression", y_te, logreg.predict(X_te), l_te)

    # 2. Tiny MLP
    print("\n=== 2. Tiny neural network (MLP) ===")
    net, mean, scale, mlp_pred, losses = train_mlp(X_tr, y_tr, X_te, y_te)
    mlp_acc, mlp_cm = show_results("MLP", y_te, mlp_pred, l_te)

    # Save
    common.MODELS_DIR.mkdir(exist_ok=True)
    import joblib
    joblib.dump(logreg, common.MODELS_DIR / "logreg.joblib")
    torch.save({"state_dict": net.state_dict(), "mean": torch.from_numpy(mean),
                "scale": torch.from_numpy(scale), "in_dim": int(X.shape[1]),
                "hidden": 128}, common.MODELS_DIR / "mlp.pt")
    best = "mlp" if mlp_acc >= lr_acc else "logreg"
    info = {"best": best, "logreg_accuracy": lr_acc, "mlp_accuracy": mlp_acc,
            "labels": LABELS, "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "feature_version": common.FEATURE_VERSION}
    (common.MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2),
                                                       encoding="utf-8")
    save_confusion_plot(lr_cm, f"LogReg {lr_acc * 100:.0f}%",
                        common.MODELS_DIR / "confusion_logreg.png")
    save_confusion_plot(mlp_cm, f"MLP {mlp_acc * 100:.0f}%",
                        common.MODELS_DIR / "confusion_mlp.png")

    print("\n" + "=" * 50)
    print(f"Logistic regression: {lr_acc * 100:.1f}%    MLP: {mlp_acc * 100:.1f}%")
    print(f"Saved both models in the models folder. demo.py will use: {best}")
    print(f"Done in {time.time() - t0:.0f} s. Next step: python demo.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
