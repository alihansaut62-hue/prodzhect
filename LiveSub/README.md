# LiveSub — живые субтитры EN → RU

Офлайн-программа: слушает микрофон (английская речь) и показывает поверх экрана русские субтитры.
Распознавание — Whisper (faster-whisper), перевод — Argos Translate. Оба работают на видеокарте NVIDIA через CUDA,
без видеокарты — на процессоре (медленнее).

## Требования

| Что | Зачем |
|---|---|
| Windows 10/11 | окно субтитров сделано под Windows |
| Python **3.11** (64-bit) — https://www.python.org/downloads/ | при установке отметить «Add python.exe to PATH» |
| Видеокарта NVIDIA (желательно, напр. RTX 3060) + свежий драйвер — https://www.nvidia.com/Download/index.aspx | ускорение на GPU |
| Микрофон | |
| ~5 ГБ на диске, интернет при первом запуске | скачиваются модели |

CUDA Toolkit отдельно ставить **не нужно** — нужные DLL ставятся через pip.

## Библиотеки

| Пакет | Для чего |
|---|---|
| `faster-whisper` | распознавание речи (Whisper на движке CTranslate2) |
| `argostranslate` | офлайн-перевод EN → RU |
| `sounddevice` | запись с микрофона |
| `numpy` | работа со звуком как с массивом |
| `nvidia-cublas-cu12` | CUDA-библиотека матричных вычислений (GPU) |
| `nvidia-cudnn-cu12` (версия 9) | CUDA-библиотека для нейросетей (GPU) |

`tkinter` (окно) уже входит в Python.

## Установка (один раз)

Открыть PowerShell и выполнить по очереди:

```powershell
cd C:\LiveSub
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Если PowerShell ругается на `activate` («выполнение сценариев отключено»):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Без видеокарты NVIDIA последние два пакета можно не ставить:

```powershell
pip install faster-whisper argostranslate sounddevice numpy
```

## Запуск

```powershell
cd C:\LiveSub
.\.venv\Scripts\activate
python live_subtitles.py
```

Или одной командой, без активации:

```powershell
C:\LiveSub\.venv\Scripts\python.exe C:\LiveSub\live_subtitles.py
```

Первый запуск долгий: скачиваются Whisper `large-v3-turbo` (~1.6 ГБ) и пакет перевода EN → RU.
Дальше всё работает без интернета. Когда в окне появится «Готово (CUDA)» — можно говорить.

## Параметры

```powershell
python live_subtitles.py --list-devices      # список микрофонов с номерами
python live_subtitles.py --device 3          # выбрать микрофон по номеру
python live_subtitles.py --model small.en    # модель полегче (быстрее, менее точно)
python live_subtitles.py --cpu               # не использовать видеокарту
```

Модели по скорости → точности: `base.en`, `small.en`, `medium.en`, `large-v3-turbo` (по умолчанию на GPU), `large-v3`.

## Управление

- **Пауза / Продолжить** — кнопка сверху окна
- **Перетащить** — зажать мышь на панели или тексте
- **✕** или **Esc** — выход

## Если что-то не так

| Проблема | Решение |
|---|---|
| Пишет «(CPU)», хотя есть видеокарта | `pip install --force-reinstall nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"`, обновить драйвер NVIDIA |
| Ошибка про `cublas64_12.dll` / `cudnn64_9.dll` | то же самое, либо запустить с `--cpu` |
| Нет субтитров, хотя говорю | `--list-devices`, затем `--device N` с нужным микрофоном |
| Ловит фоновый шум | поднять `MIN_RMS` в начале `live_subtitles.py` (напр. 0.008) |
| Субтитры отстают | `--model small.en` |
| `WARNING: GPU requested, but is not available!` | не ошибка — это вспомогательная библиотека stanza, перевод всё равно идёт на GPU |
| Предупреждение про symlinks | не ошибка, можно игнорировать |
