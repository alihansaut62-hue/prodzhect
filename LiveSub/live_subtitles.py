"""Offline English microphone -> Russian desktop subtitles (Windows, Python 3.11).

GPU (CUDA) is used for both Whisper and the Argos translator when available.
Install: pip install faster-whisper argostranslate sounddevice numpy nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
"""
import argparse
import ctypes
import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import sounddevice as sd

RATE = 16000
BLOCK = 1600                 # 100 ms audio blocks
MIN_RMS = 0.004              # absolute floor for the speech threshold
PRE_ROLL_BLOCKS = 3          # keep 300 ms before speech starts, so first words are not cut
END_SILENCE_SECONDS = 0.6    # pause that ends a phrase
MAX_PHRASE_SECONDS = 8.0     # force a final result for long monologues
INTERIM_EVERY_SECONDS = 0.8  # live preview while the phrase is still being spoken
HALLUCINATIONS = {"", "you", "thank you.", "thank you", "thanks for watching!", "bye.", "."}

LOCAL_CUDA = Path(__file__).resolve().parent / "cuda_libs"

stop_event = threading.Event()
pause_event = threading.Event()
text_queue = queue.Queue()
dll_handles = []


def status(message):
    print(message, flush=True)
    text_queue.put((message, ""))


def cuda_dirs():
    """Folders with cuBLAS/cuDNN: local cuda_libs plus pip wheels nvidia-*-cu12."""
    dirs = [LOCAL_CUDA]
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            dirs += [p for p in nvidia.glob("*/bin") if p.is_dir()]
    return [d for d in dirs if d.is_dir()]


def gpu_ready():
    """Register CUDA DLL folders and check that CTranslate2 can actually use the GPU."""
    for folder in cuda_dirs():
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
        dll_handles.append(os.add_dll_directory(str(folder)))
    try:
        ctypes.WinDLL("cublas64_12.dll")
        ctypes.WinDLL("cudnn64_9.dll")
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except (OSError, ImportError):
        return False


def get_translator():
    import argostranslate.package
    import argostranslate.translate

    def installed():
        langs = argostranslate.translate.get_installed_languages()
        en = next((lang for lang in langs if lang.code == "en"), None)
        ru = next((lang for lang in langs if lang.code == "ru"), None)
        if en is None or ru is None:
            return None
        # A direct pair is necessary, to avoid a missing-language pivot.
        return en.get_translation(ru)

    translator = installed()
    if translator is not None:
        return translator
    status("Скачиваю переводчик EN → RU (один раз)…")
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    package = next((p for p in packages if p.from_code == "en" and p.to_code == "ru"), None)
    if package is None:
        raise RuntimeError("Пакет Argos EN → RU не найден")
    argostranslate.package.install_from_path(package.download())
    translator = installed()
    if translator is None:
        raise RuntimeError("Переводчик EN → RU не установился")
    return translator


def worker(device_index, model_override, force_cpu):
    try:
        use_gpu = not force_cpu and gpu_ready()
        # Argos reads this when it loads its CTranslate2 model.
        os.environ["ARGOS_DEVICE_TYPE"] = "cuda" if use_gpu else "cpu"
        from faster_whisper import WhisperModel

        model_name = model_override or ("large-v3-turbo" if use_gpu else "base.en")
        device_name = "CUDA" if use_gpu else "CPU"
        status(f"Загружаю Whisper {model_name} ({device_name})… первый раз скачивается модель")
        model = WhisperModel(model_name,
                             device="cuda" if use_gpu else "cpu",
                             compute_type="float16" if use_gpu else "int8")
        translator = get_translator()

        # Warm-up: the first CUDA call is slow (kernel init), do it before the user speaks.
        list(model.transcribe(np.zeros(RATE, dtype=np.float32), language="en")[0])
        translator.translate("Hello")
        status(f"Готово ({device_name}). Говорите на английском")

        def transcribe(audio, final):
            segments, _ = model.transcribe(
                audio, language="en",
                beam_size=5 if final and use_gpu else 1,
                temperature=0.0,                 # no slow fallback re-decoding
                condition_on_previous_text=False,
                without_timestamps=True,
                vad_filter=False,                # phrases are already cut by our endpointing
            )
            parts = [s.text.strip() for s in segments
                     if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
            text = " ".join(parts).strip()
            return "" if text.lower() in HALLUCINATIONS else text

        def show(audio, final):
            started = time.perf_counter()
            original = transcribe(audio, final)
            if not original:
                return
            translated = translator.translate(original).strip()
            if not translated:
                return
            text_queue.put((translated, original))
            if final:
                took = (time.perf_counter() - started) * 1000
                print(f"EN: {original}\nRU: {translated}   [{took:.0f} мс]", flush=True)

        capture_queue = queue.Queue(maxsize=50)

        def on_audio(indata, frames, time_info, callback_status):
            if callback_status:
                print("Аудио:", callback_status, flush=True)
            if stop_event.is_set() or pause_event.is_set():
                return
            try:
                capture_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass  # recognition is lagging: drop audio to stay real time

        blocks_end = int(END_SILENCE_SECONDS * RATE / BLOCK)
        blocks_max = int(MAX_PHRASE_SECONDS * RATE / BLOCK)
        blocks_interim = int(INTERIM_EVERY_SECONDS * RATE / BLOCK)

        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                            blocksize=BLOCK, callback=on_audio, device=device_index):
            noise = MIN_RMS
            pre_roll = []
            phrase = []
            silent = 0
            since_interim = 0
            while not stop_event.is_set():
                if pause_event.is_set():
                    pre_roll, phrase, silent = [], [], 0
                    time.sleep(0.1)
                    continue
                try:
                    block = capture_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                rms = float(np.sqrt(np.mean(block * block)))
                threshold = max(MIN_RMS, noise * 3.0)
                loud = rms > threshold

                if not phrase:
                    # Waiting for speech; track background noise level.
                    if loud:
                        phrase = pre_roll + [block]
                        silent = 0
                        since_interim = 0
                    else:
                        noise = 0.95 * noise + 0.05 * rms
                        pre_roll = (pre_roll + [block])[-PRE_ROLL_BLOCKS:]
                    continue

                phrase.append(block)
                silent = 0 if loud else silent + 1
                since_interim += 1

                if silent >= blocks_end or len(phrase) >= blocks_max:
                    # Trim trailing silence, keep a little tail for the last word.
                    keep = len(phrase) - max(0, silent - 2)
                    show(np.concatenate(phrase[:keep]), final=True)
                    pre_roll, phrase, silent = [], [], 0
                    # Drain backlog accumulated during decoding, if we fell behind.
                    while capture_queue.qsize() > blocks_end:
                        capture_queue.get_nowait()
                elif since_interim >= blocks_interim and capture_queue.qsize() < 3:
                    since_interim = 0
                    show(np.concatenate(phrase), final=False)
    except Exception as exc:
        status(f"Ошибка: {exc}")
        print("Если ошибка связана с CUDA — запустите с --cpu или переустановите "
              "nvidia-cublas-cu12 и nvidia-cudnn-cu12.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Whisper EN → RU overlay")
    parser.add_argument("--list-devices", action="store_true", help="Показать микрофоны")
    parser.add_argument("--device", type=int, default=None, help="Номер микрофона")
    parser.add_argument("--model", default=None,
                        help="Модель Whisper (по умолчанию large-v3-turbo на GPU, base.en на CPU)")
    parser.add_argument("--cpu", action="store_true", help="Не использовать видеокарту")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    root = tk.Tk()
    root.title("Whisper Live EN → RU")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    magic = "#010203"
    root.configure(bg=magic)
    root.wm_attributes("-transparentcolor", magic)
    width, height = min(1100, root.winfo_screenwidth() - 40), 200
    x = (root.winfo_screenwidth() - width) // 2
    y = root.winfo_screenheight() - height - 70
    root.geometry(f"{width}x{height}+{x}+{y}")

    bar = tk.Frame(root, bg="#202530")
    bar.pack(anchor="center", pady=(0, 8))
    state = tk.StringVar(value="Пауза")

    def toggle():
        if pause_event.is_set():
            pause_event.clear()
            state.set("Пауза")
        else:
            pause_event.set()
            state.set("Продолжить")

    def close():
        stop_event.set()
        root.destroy()

    tk.Button(bar, textvariable=state, command=toggle, bg="#202530", fg="white",
              activebackground="#405070", activeforeground="white", relief="flat").pack(side="left", padx=8)
    tk.Label(bar, text="Перетащить", bg="#202530", fg="white", padx=8).pack(side="left")
    tk.Button(bar, text="✕", command=close, bg="#202530", fg="white",
              activebackground="#a04040", relief="flat").pack(side="left", padx=8)

    label = tk.Label(root, text="Загрузка моделей…", font=("Segoe UI", 23, "bold"),
                     bg=magic, fg="white", wraplength=width - 20, justify="center")
    label.pack(fill="both", expand=True)
    original_label = tk.Label(root, text="", font=("Segoe UI", 12),
                              bg=magic, fg="#c8d0e0", wraplength=width - 20, justify="center")
    original_label.pack(fill="x")

    drag = {"x": 0, "y": 0}
    def start_drag(event):
        drag["x"], drag["y"] = event.x_root - root.winfo_x(), event.y_root - root.winfo_y()
    def move_drag(event):
        root.geometry(f"+{event.x_root - drag['x']}+{event.y_root - drag['y']}")
    for widget in (bar, label, original_label):
        widget.bind("<ButtonPress-1>", start_drag)
        widget.bind("<B1-Motion>", move_drag)

    def update():
        latest = None
        try:
            while True:
                latest = text_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            label.configure(text=latest[0])
            original_label.configure(text=latest[1])
        if not stop_event.is_set():
            root.after(50, update)

    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<Escape>", lambda event: close())
    threading.Thread(target=worker, args=(args.device, args.model, args.cpu), daemon=True).start()
    update()
    root.mainloop()
    stop_event.set()


if __name__ == "__main__":
    main()
