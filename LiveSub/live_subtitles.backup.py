"""Offline English microphone -> Russian desktop subtitles (Windows, Python 3.11)."""
import argparse
import ctypes
import os
from pathlib import Path
import queue
import threading
import tkinter as tk

import numpy as np
import sounddevice as sd

# Recommended install: pip install faster-whisper argostranslate sounddevice numpy

RATE = 16000
CHUNK_SECONDS = 2.0
MIN_RMS = 0.004
CUDA_FOLDER = Path(__file__).resolve().parent / "cuda_libs"

stop_event = threading.Event()
pause_event = threading.Event()
text_queue = queue.Queue()
dll_handles = []


def status(message):
    print(message, flush=True)
    text_queue.put(message)


def gpu_ready():
    """Avoid attempting CUDA when runtime DLLs are missing."""
    paths = [CUDA_FOLDER] + [Path(p) for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    for dll in ("cublas64_12.dll", "cudnn64_9.dll"):
        if not any((folder / dll).is_file() for folder in paths):
            return False
    if CUDA_FOLDER.is_dir():
        os.environ["PATH"] = str(CUDA_FOLDER) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            dll_handles.append(os.add_dll_directory(str(CUDA_FOLDER)))
    try:
        ctypes.WinDLL("cublas64_12.dll")
        ctypes.WinDLL("cudnn64_9.dll")
        return True
    except OSError:
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


def worker(device_index):
    try:
        from faster_whisper import WhisperModel
        use_gpu = gpu_ready()
        model_name = "small.en" if use_gpu else "base.en"
        status(f"Загружаю Whisper {model_name} ({'RTX 3060 / CUDA' if use_gpu else 'CPU'})…")
        model = WhisperModel(model_name,
                             device="cuda" if use_gpu else "cpu",
                             compute_type="float16" if use_gpu else "int8")
        translator = get_translator()
        status("Готово. Говорите на английском")

        def on_audio(indata, frames, time_info, callback_status):
            if callback_status:
                print("Аудио:", callback_status, flush=True)
            if stop_event.is_set() or pause_event.is_set():
                return
            samples = indata[:, 0].copy()
            try:
                capture_queue.put_nowait(samples)
            except queue.Full:
                # Drop oldest samples to stay near real time under overload.
                try:
                    capture_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    capture_queue.put_nowait(samples)
                except queue.Full:
                    pass

        capture_queue = queue.Queue(maxsize=30)
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                            blocksize=1600, callback=on_audio, device=device_index):
            pending = np.empty(0, dtype=np.float32)
            limit = int(CHUNK_SECONDS * RATE)
            while not stop_event.is_set():
                if pause_event.is_set():
                    pending = np.empty(0, dtype=np.float32)
                    try:
                        capture_queue.get(timeout=0.1)
                    except queue.Empty:
                        pass
                    continue
                try:
                    next_block = capture_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                pending = np.concatenate((pending, next_block))
                if len(pending) < limit:
                    continue
                chunk = pending[:limit].copy()
                pending = pending[limit:]
                if float(np.sqrt(np.mean(chunk * chunk))) < MIN_RMS:
                    continue
                segments, _ = model.transcribe(
                    chunk, language="en", beam_size=1, best_of=1,
                    condition_on_previous_text=False, vad_filter=False,
                )
                original = " ".join(s.text.strip() for s in segments).strip()
                if not original:
                    continue
                translated = translator.translate(original).strip()
                if translated:
                    print("EN:", original, "\nRU:", translated, flush=True)
                    text_queue.put(translated)
    except Exception as exc:
        status(f"Ошибка: {exc}")
        print("Если ошибка связана с CUDA, проверьте CUDA 12/cuDNN 9 или временно используйте CPU.")


def main():
    parser = argparse.ArgumentParser(description="Whisper EN → RU overlay")
    parser.add_argument("--list-devices", action="store_true", help="Показать микрофоны")
    parser.add_argument("--device", type=int, default=None, help="Номер микрофона")
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
    width, height = min(1100, root.winfo_screenwidth() - 40), 160
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

    drag = {"x": 0, "y": 0}
    def start_drag(event):
        drag["x"], drag["y"] = event.x_root - root.winfo_x(), event.y_root - root.winfo_y()
    def move_drag(event):
        root.geometry(f"+{event.x_root - drag['x']}+{event.y_root - drag['y']}")
    for widget in (bar, label):
        widget.bind("<ButtonPress-1>", start_drag)
        widget.bind("<B1-Motion>", move_drag)

    def update():
        try:
            while True:
                label.configure(text=text_queue.get_nowait())
        except queue.Empty:
            pass
        if not stop_event.is_set():
            root.after(80, update)

    root.protocol("WM_DELETE_WINDOW", close)
    root.bind("<Escape>", lambda event: close())
    threading.Thread(target=worker, args=(args.device,), daemon=True).start()
    update()
    root.mainloop()
    stop_event.set()


if __name__ == "__main__":
    main()
