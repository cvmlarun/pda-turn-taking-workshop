# Turn-Taking Workshop: Teach a Voice Assistant When to Talk

A 1-hour hands-on workshop. You train a small model that listens to 3 seconds
of speech and decides:

| label | what the speaker did | what the assistant should do |
|---|---|---|
| `respond` | finished a sentence or question | answer now |
| `keep_going` | paused in the middle of a sentence | wait and keep listening |
| `stop` | said "stop", "ruko", "thamba", "nillisu" | stop talking at once |

Languages: Kannada (`kn`), Hindi (`hi`), Marathi (`mr`), English (`en`).

It runs on a normal Windows laptop. No GPU needed. 4 GB RAM is enough.

**How it works:** a frozen [whisper-tiny](https://huggingface.co/openai/whisper-tiny)
speech encoder turns each clip into 768 numbers. A small classifier learns from
those numbers. First logistic regression, then a tiny neural network (MLP).
Training takes seconds.

---

## Before the workshop: setup (about 30 minutes)

Do this at home, with good internet. You need about 3 GB of free disk space.
Please finish it before the workshop day.

### 1. Install Python 3.11

1. Go to <https://www.python.org/downloads/release/python-3119/>.
2. Scroll down. Click **Windows installer (64-bit)**.
3. Run the installer.
4. **Tick the box "Add python.exe to PATH"** at the bottom of the first screen. This is important.
5. Click **Install Now**.

### 2. Install Git

1. Go to <https://git-scm.com/download/win>.
2. Download and run the 64-bit installer.
3. Click **Next** on every screen. The defaults are fine.

### 3. Download the workshop

1. Open the Start menu. Type `cmd`. Open **Command Prompt**.
   (Use Command Prompt, not PowerShell. It avoids a common problem.)
2. Type these lines. Press Enter after each one.

```bat
cd %USERPROFILE%\Documents
git clone https://github.com/cvmlarun/pda-turn-taking-workshop.git
cd pda-turn-taking-workshop
```

### 4. Make a virtual environment

A virtual environment keeps the workshop packages in one folder.

```bat
python -m venv .venv
.venv\Scripts\activate
```

You should now see `(.venv)` at the start of the line.
**Every time** you open a new Command Prompt, run these two lines again:

```bat
cd %USERPROFILE%\Documents\pda-turn-taking-workshop
.venv\Scripts\activate
```

### 5. Install the packages

First PyTorch (the CPU version, about 200 MB). Then the rest.

```bat
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

This can take 5 to 15 minutes. Wait until you see the prompt again.

### 6. Run the setup check

```bat
python setup_check.py
```

It will:

- check Python and every package,
- download the whisper-tiny model (about 150 MB),
- download the starter dataset (about 45 MB),
- record 3 seconds from your microphone and play it back.

Say something when it asks. Type `y` if you heard your voice.

At the end you must see:

```
ALL CHECKS PASSED
```

**Take a screenshot of that window and send it to the workshop coordinator.**

If you see `[FAIL]`, read the message next to it. Then look at
[Troubleshooting](#troubleshooting) below.

After this check passes, everything works **offline**.

---

## During the workshop

Open Command Prompt, go to the folder, and activate the environment (step 4).
Then run the scripts in order. None of them need extra arguments.

| step | command | what it does | time |
|---|---|---|---|
| 1 | `python train.py` | trains on the starter data right away | < 1 min |
| 2 | `python demo.py` | live test with your mic | 5 min |
| 3 | `python record.py` | record your own clips | 15 min |
| 4 | `python extract_features.py` | turn clips into whisper-tiny features | 1-3 min |
| 5 | `python train.py` | train again with your clips | < 1 min |
| 6 | `python demo.py` | is it better now? | 5 min |

Prefer a notebook? It does the same steps, with explanations:

```bat
jupyter notebook notebooks\workshop.ipynb
```

A browser tab opens. Run each cell with **Shift + Enter**.

### Recording tips (record.py)

- Record at least **10 clips per label**, in your own language.
- `respond`: say a full sentence or question. Then stay quiet.
  Example: "Naale class ide alva?"
- `keep_going`: stop in the **middle** of a sentence, as if you are thinking.
  Example: "Main soch raha tha ki..."
- `stop`: say a stop word. Example: "Thamba!", "Ruko!", "Nillisu!", "Stop!"
- Each clip is 3 seconds. Start speaking right after you press Enter.
- Clips are saved as `data\<label>\<lang>_<name>_<n>.wav`.

### What is in this folder

```
setup_check.py        check your laptop, download model and data
record.py             record your own clips
extract_features.py   clips -> whisper-tiny features (features.npz)
train.py              train logistic regression + MLP, save to models\
demo.py               live mic test
common.py             shared helper code (do not run directly)
notebooks\workshop.ipynb    the same steps as a notebook
starter_data\make_starter_data.py   makes the synthetic starter set (coordinator only)
starter_data\starter_features.npz   precomputed features for the starter set
```

Weak laptop? The starter set comes with precomputed features.
`train.py` works right away, even without running `extract_features.py`.
`extract_features.py` only processes clips it has not seen before.

---

## Troubleshooting

**"'python' is not recognized as an internal or external command"**

- Python is not on PATH. Run the Python installer again.
  Choose **Modify**, then **Next**, and tick **Add Python to environment variables**.
- Or use `py -3.11` instead of `python` when you make the venv:
  `py -3.11 -m venv .venv`. After you activate the venv, `python` works.
- Close and reopen Command Prompt after installing.
- If typing `python` opens the Microsoft Store: open **Settings > Apps >
  Advanced app settings > App execution aliases**. Turn off both
  **python.exe** and **python3.exe**.

**PowerShell says "running scripts is disabled on this system"**

- Easiest fix: use **Command Prompt** (`cmd`) instead of PowerShell.
- Or, in PowerShell, run this once:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
  Type `Y` and press Enter. Then activate with `.venv\Scripts\Activate.ps1`.

**The mic test says the recording is silent**

- Open **Settings > Privacy & security > Microphone** (Windows 11) or
  **Settings > Privacy > Microphone** (Windows 10).
- Turn on **Microphone access**.
- Turn on **Let desktop apps access your microphone**.
- Check your mic is not muted. Some laptops have a mic mute key (F4 or similar).
- Close Zoom, Teams, or Google Meet. They can hold the mic.
- Pick the right mic: right-click the speaker icon in the taskbar >
  **Sound settings** > **Input**. Speak and watch the level bar move.
- Run `python setup_check.py` again.

**I could not hear the playback**

- Turn up the volume. Try earphones.
- Your recording is saved as `mic_test.wav`. Double-click it to play it.

**pip is very slow or stops with a timeout**

- Use a better network. A phone hotspot often works better than college Wi-Fi.
- Add a longer timeout: `python -m pip install --timeout 120 -r requirements.txt`
- If it stopped half way, just run the same command again. It continues.
- Not enough disk space? Run `python -m pip cache purge` to free space.

**Antivirus blocks or deletes files, or install fails with "Access is denied"**

- Some antivirus tools (and Windows Defender) block new `.exe` files in `.venv\Scripts`.
- Add the folder `Documents\pda-turn-taking-workshop` as an exclusion in your antivirus.
  In Windows Security: **Virus & threat protection > Manage settings >
  Exclusions > Add an exclusion > Folder**.
- Then delete the `.venv` folder and start again from step 4.
- If your Documents folder is in OneDrive, pause OneDrive syncing during setup.

**The model or starter data download fails**

- College networks sometimes block downloads. Try a phone hotspot.
- Ask the coordinator for `starter_clips.zip`. Put it in the `starter_data` folder.
  Run `python setup_check.py` again.

**Something else is wrong**

- Send the file `setup_check.log` and a screenshot to the coordinator.

---

## For the coordinator

- The starter set: 480 synthetic clips (160 per label, 40 per label per language),
  made with Microsoft Edge text-to-speech, 2 voices per language.
  `keep_going` clips are the same sentences as `respond`, cut at a word in the
  middle. All labels get the same random pause length, clip length and
  background noise, so the model cannot cheat on those.
- To rebuild it: `python -m pip install edge-tts`, then
  `python starter_data\make_starter_data.py`. Upload `starter_data\starter_clips.zip`
  to the `starter-v1` GitHub release and commit `starter_data\starter_features.npz`.
- Suggested 60-minute plan: 10 min idea + demo, 10 min notebook steps 1-4,
  20 min recording in pairs, 10 min retrain + live test, 10 min discussion.
- Disk use: about 1.5 GB for the venv, 150 MB for the model, 50 MB for data.
