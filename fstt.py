#!/usr/bin/python3
"""
Плавающее окно для записи и распознавания речи с микрофона.

Архитектура приложения специально осознанно выбрана как God-file
"""

import sys
import wave
import time
import numpy as np
import sounddevice as sd
import onnx_asr
import threading
import signal
import json
import os
import subprocess
import shutil
import shlex
import httpx
import gi
from enum import Enum
from typing import Callable, Optional, Protocol, Dict, Set

gi.require_version('Gtk', '3.0')
gi.require_version('GtkLayerShell', '0.1')
gi.require_version('Gdk', '3.0')
from gi.repository import Gtk, GtkLayerShell, GLib, Gdk


def log(message):
    """Вывод отладочной информации в stderr"""
    print(message, file=sys.stderr)


def load_prompt_from_file(file_path: str, default_prompt: str) -> str:
    """Загружает текст промпта из файла"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            log(f"⚠️  Файл с промптом не найден: {file_path}")
    except Exception as e:
        log(f"❌ Ошибка загрузки промпта из файла: {e}")
    return default_prompt


# ============================================================================
# STATE MACHINE
# ============================================================================

class AppState(Enum):
    """Состояния приложения"""
    IDLE = "idle"           # Готов к записи
    RECORDING = "recording" # Идёт запись
    PROCESSING = "processing" # Обработка записи
    POST_PROCESSING = "post_processing" # Пост-обработка текста
    RESTARTING = "restarting" # Перезапуск записи


class UIStateMachine:
    """Управление состоянием UI через конечный автомат"""

    # Карта разрешённых переходов между состояниями
    VALID_TRANSITIONS: Dict[AppState, Set[AppState]] = {
        AppState.IDLE: {AppState.RECORDING},
        AppState.RECORDING: {AppState.PROCESSING, AppState.RESTARTING, AppState.IDLE},
        AppState.PROCESSING: {AppState.POST_PROCESSING, AppState.IDLE},
        AppState.POST_PROCESSING: {AppState.IDLE},
        AppState.RESTARTING: {AppState.RECORDING, AppState.IDLE}
    }

    def __init__(self):
        self.state = AppState.IDLE
        self.observers = []

    def add_observer(self, observer: Callable[[AppState, AppState], None]):
        """Добавляет наблюдателя за изменениями состояния"""
        self.observers.append(observer)

    def transition_to(self, new_state: AppState):
        """Выполняет переход в новое состояние и уведомляет наблюдателей"""
        if new_state == self.state:
            log(f"⚠️  Попытка перейти в текущее состояние: {new_state.value}")
            return

        # Валидация перехода
        if not self._is_valid_transition(new_state):
            log(f"❌ Невалидный переход: {self.state.value} → {new_state.value}")
            raise ValueError(
                f"Невалидный переход состояния: {self.state.value} → {new_state.value}"
            )

        old_state = self.state
        self.state = new_state
        log(f"🔄 Переход состояния: {old_state.value} → {new_state.value}")

        # Уведомляем всех наблюдателей
        for observer in self.observers:
            try:
                observer(old_state, new_state)
            except Exception as e:
                log(f"❌ Ошибка в observer: {e}")

    def _is_valid_transition(self, new_state: AppState) -> bool:
        """Проверяет, является ли переход в новое состояние валидным"""
        allowed_states = self.VALID_TRANSITIONS.get(self.state, set())
        return new_state in allowed_states

    def is_state(self, state: AppState) -> bool:
        """Проверяет, находится ли приложение в указанном состоянии"""
        return self.state == state



# ============================================================================
# ПРОТОКОЛЫ (АБСТРАКЦИИ)
# ============================================================================

class ClipboardProtocol(Protocol):
    """Протокол для сервиса работы с буфером обмена"""

    def copy_standard(self, text: str) -> bool:
        """Копирует текст в стандартный буфер обмена (Ctrl+V)"""
        ...

    def copy_primary(self, text: str) -> bool:
        """Копирует текст в primary selection (средняя кнопка мыши)"""
        ...


class PasteProtocol(Protocol):
    """Протокол для сервиса вставки текста"""

    def paste(self) -> bool:
        """Эмулирует вставку текста в зависимости от настройки"""
        ...


class SpeechProtocol(Protocol):
    """Протокол для сервиса записи и распознавания речи"""

    @property
    def is_recording(self) -> bool:
        """Возвращает True, если идёт запись"""
        ...

    def start(self) -> bool:
        """Начинает запись аудио"""
        ...

    def stop(self) -> None:
        """Останавливает запись БЕЗ распознавания"""
        ...

    def stop_and_recognize(self) -> Optional[str]:
        """Останавливает запись и распознаёт речь"""
        ...


class PostProcessingProtocol(Protocol):
    """Протокол для сервиса пост-обработки текста"""

    def process(self, text: str) -> str:
        """Обрабатывает текст с помощью LLM"""
        ...


# ============================================================================
# КОНФИГУРАЦИЯ И КОНСТАНТЫ
# ============================================================================

class AudioConfig:
    """Настройки для аудио записи и распознавания"""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = 'int16'
    SAMPLE_WIDTH = 2
    MODEL_NAME = "gigaam-v3-e2e-rnnt"
    WAV_FILE = "recording.wav"


class UIConfig:
    """Настройки для пользовательского интерфейса"""
    DEFAULT_WINDOW_X = 20
    DEFAULT_WINDOW_Y = 20
    ICON_RECORD = "●"
    ICON_STOP = "■"
    ICON_PROCESSING = "⋯"
    ICON_CLOSE = "✕"
    ICON_RESTART = "↻"
    ICON_PP_ON = "☑"   # Квадрат с галочкой
    ICON_PP_OFF = "☐"  # Пустой квадрат
    BOX_SPACING = 5
    BOX_MARGIN = 10
    MOUSE_BUTTON_LEFT = 1

    CSS_STYLES = b"""
window {
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 10px;
}
button {
    background-color: rgba(0, 0, 0, 0.3);
    color: rgba(255, 255, 255, 0.5);
    border-radius: 5px;
    border: none;
    font-size: 20px;
    padding: 5px 10px;
}
button:hover {
    background-color: rgba(60, 60, 60, 0.3);
}
button:disabled {
    background-color: rgba(0, 0, 0, 0.3);
    color: rgba(120, 120, 120, 0.5);
}
.record-button label {
    margin-top: -2px;
    margin-bottom: 2px;
}
.restart-button label {
    margin-top: 1px;
    margin-bottom: -1px;
}
.close-button label {
    margin-top: 0px;
    margin-bottom: 0px;
}
.autopaste-button label {
    margin-top: 0px;
    margin-bottom: 0px;
}
"""


class AppSettings:
    """Настройки поведения приложения"""
    APP_ID = 'com.example.voice_recognition'
    COPY_METHOD = "clipboard"  # "primary", "clipboard"
    AUTO_PASTE = True
    LLM_ENABLED = True
    LLM_PROMPT_FILE = "prompt.md"
    LLM_TEMPERATURE = 1.0
    LLM_MAX_RETRIES = 2
    LLM_TIMEOUT_SEC = 60
    SMART_TEXT_PROCESSING = True  # Включает умную обработку текста (короткие/длинные фразы)
    SMART_TEXT_SHORT_PHRASE = 3  # Максимальное количество слов для постобработки обработки коротких фраз

    # OpenAI settings from environment
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
#    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
#    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gemini-2.5-flash")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Таймауты и задержки
    PASTE_DELAY_MS = 200
    RESTART_DELAY_SEC = 0.1


class WindowPositionPersistence:
    """Управление сохранением и загрузкой позиции окна"""

    CONFIG_FILE = os.path.expanduser("~/.config/voice-recognition-window.json")

    @classmethod
    def load(cls) -> tuple[int, int]:
        """Загружает сохранённую позицию окна из конфига"""
        try:
            if os.path.exists(cls.CONFIG_FILE):
                with open(cls.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    x = config.get('x', UIConfig.DEFAULT_WINDOW_X)
                    y = config.get('y', UIConfig.DEFAULT_WINDOW_Y)
                    log(f"📂 Загружена позиция окна: x={x}, y={y}")
                    return x, y
        except Exception as e:
            log(f"⚠️  Ошибка загрузки конфига: {e}")

        return UIConfig.DEFAULT_WINDOW_X, UIConfig.DEFAULT_WINDOW_Y

    @classmethod
    def save(cls, x: int, y: int) -> None:
        """Сохраняет позицию окна в конфиг"""
        try:
            config_dir = os.path.dirname(cls.CONFIG_FILE)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)

            config = {'x': x, 'y': y}
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            log(f"💾 Сохранена позиция окна: x={x}, y={y}")
        except Exception as e:
            log(f"⚠️  Ошибка сохранения конфига: {e}")


class AppConfig:
    """
    Объединённая конфигурация приложения.

    Предоставляет единую точку доступа ко всем настройкам через ссылки на под-конфиги.

    Использование:
        config = AppConfig()
        config.audio.SAMPLE_RATE  # Аудио настройки
        config.ui.ICON_RECORD     # UI настройки
        config.settings.AUTO_PASTE # Настройки поведения
        config.window.load()      # Работа с позицией окна
    """

    # Ссылки на под-конфиги
    audio = AudioConfig
    ui = UIConfig
    settings = AppSettings
    window = WindowPositionPersistence






# ============================================================================
# ФАБРИКА СЕРВИСОВ
# ============================================================================

class ServiceFactory:
    """
    Фабрика для создания сервисов с их зависимостями.

    Поддерживает Dependency Injection через конструктор для легкой замены реализаций.
    """

    def __init__(
        self,
        clipboard_class: type = None,
        paste_class: type = None,
        speech_class: type = None,
        post_processing_class: type = None
    ):
        """
        Инициализирует фабрику с возможностью внедрения зависимостей.

        Args:
            clipboard_class: Класс для создания сервиса буфера обмена (по умолчанию ClipboardService)
            paste_class: Класс для создания сервиса вставки (по умолчанию PasteService)
            speech_class: Класс для создания сервиса распознавания речи (по умолчанию SpeechService)
            post_processing_class: Класс для создания сервиса пост-обработки (по умолчанию PostProcessingService)
        """
        # Используем отложенную инициализацию дефолтных классов, чтобы избежать circular dependencies
        self._clipboard_class = clipboard_class
        self._paste_class = paste_class
        self._speech_class = speech_class
        self._post_processing_class = post_processing_class

    @property
    def clipboard_class(self):
        """Возвращает класс сервиса буфера обмена (ленивая инициализация)"""
        if self._clipboard_class is None:
            return ClipboardService
        return self._clipboard_class

    @property
    def paste_class(self):
        """Возвращает класс сервиса вставки (ленивая инициализация)"""
        if self._paste_class is None:
            return PasteService
        return self._paste_class

    @property
    def speech_class(self):
        """Возвращает класс сервиса распознавания речи (ленивая инициализация)"""
        if self._speech_class is None:
            return SpeechService
        return self._speech_class

    @property
    def post_processing_class(self):
        """Возвращает класс сервиса пост-обработки (ленивая инициализация)"""
        if self._post_processing_class is None:
            return PostProcessingService
        return self._post_processing_class

    def create_clipboard(self) -> ClipboardProtocol:
        """Создаёт сервис буфера обмена"""
        return self.clipboard_class()

    def create_paste(self, copy_method: str) -> PasteProtocol:
        """Создаёт сервис вставки текста"""
        return self.paste_class(copy_method)

    def create_speech(self, config: 'AppConfig') -> SpeechProtocol:
        """Создаёт сервис распознавания речи"""
        return self.speech_class(config)

    def create_post_processing(self, config: 'AppConfig') -> PostProcessingProtocol:
        """Создаёт сервис пост-обработки"""
        return self.post_processing_class(config)

    def create_all_services(self, config: 'AppConfig') -> tuple[SpeechProtocol, ClipboardProtocol, PasteProtocol, PostProcessingProtocol]:
        """Создаёт все необходимые сервисы"""
        speech = self.create_speech(config)
        clipboard = self.create_clipboard()
        paste = self.create_paste(config.settings.COPY_METHOD)
        post_processing = self.create_post_processing(config)
        return speech, clipboard, paste, post_processing


# ============================================================================
# СЕРВИСЫ
# ============================================================================

class ClipboardService:
    """Сервис для работы с буфером обмена (clipboard и primary selection)"""

    def copy_standard(self, text):
        """Копирует текст в стандартный буфер обмена (Ctrl+V)"""
        try:
            import pyclip
            pyclip.copy(text)
            log("📋 Скопировано в буфер обмена")
            return True
        except ImportError:
            log("⚠️  pyclip не установлен, используйте: pip install pyclip")
            log("⚠️  Или установите wl-clipboard для Wayland: sudo pacman -S wl-clipboard")
            return False
        except Exception as e:
            log(f"❌ Ошибка копирования в буфер обмена: {e}")
            return False

    def copy_primary(self, text):
        """Копирует текст в primary selection (средняя кнопка мыши)"""
        # Пробуем wl-copy для Wayland
        if shutil.which('wl-copy'):
            return self._copy_primary_wl(text)

        # Пробуем xsel для X11
        if shutil.which('xsel'):
            return self._copy_primary_xsel(text)

        # Пробуем xclip для X11
        if shutil.which('xclip'):
            return self._copy_primary_xclip(text)

        # Fallback на GTK API
        log("⚠️  Системные команды не найдены, пробую GTK Clipboard API...")
        log("💡 Установите wl-clipboard для Wayland: sudo pacman -S wl-clipboard")
        log("💡 Или установите xsel для X11: sudo pacman -S xsel")
        return self._copy_primary_gtk(text)

    def _copy_primary_wl(self, text):
        """Копирует через wl-copy (Wayland)"""
        try:
            escaped_text = shlex.quote(text)
            subprocess.Popen(
                f'printf %s {escaped_text} | wl-copy --primary &',
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            log("🖱️  Скопировано в primary selection через wl-copy")
            return True
        except Exception as e:
            log(f"❌ Ошибка при использовании wl-copy: {e}")
            return False

    def _copy_primary_xsel(self, text):
        """Копирует через xsel (X11)"""
        try:
            process = subprocess.Popen(
                ['xsel', '--primary', '--input'],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=text.encode('utf-8'))

            if process.returncode == 0:
                log("🖱️  Скопировано в primary selection через xsel")
                return True
            else:
                log(f"⚠️  xsel вернул код {process.returncode}: {stderr.decode('utf-8', errors='ignore')}")
                return False
        except Exception as e:
            log(f"❌ Ошибка при использовании xsel: {e}")
            return False

    def _copy_primary_xclip(self, text):
        """Копирует через xclip (X11)"""
        try:
            process = subprocess.Popen(
                ['xclip', '-selection', 'primary'],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=text.encode('utf-8'))

            if process.returncode == 0:
                log("🖱️  Скопировано в primary selection через xclip")
                return True
            else:
                log(f"⚠️  xclip вернул код {process.returncode}: {stderr.decode('utf-8', errors='ignore')}")
                return False
        except Exception as e:
            log(f"❌ Ошибка при использовании xclip: {e}")
            return False

    def _copy_primary_gtk(self, text):
        """Копирует через GTK Clipboard API"""
        try:
            clipboard = Gtk.Clipboard.get(Gdk.SELECTION_PRIMARY)
            clipboard.set_text(text, -1)
            clipboard.store()
            log("🖱️  Скопировано в primary selection через GTK")
            return True
        except Exception as e:
            log(f"❌ Ошибка копирования в primary selection через GTK: {e}")
            return False


class PasteService:
    """Сервис для вставки текста через эмуляцию клавиатуры (wtype)"""

    def __init__(self, copy_method: str):
        """
        Инициализация сервиса вставки

        Args:
            copy_method: Метод копирования ("clipboard", "primary")
        """
        self.copy_method = copy_method

    def paste(self):
        """Эмулирует вставку текста в зависимости от настройки copy_method"""
        if self.copy_method == "primary":
            return self._paste_primary()
        elif self.copy_method == "clipboard":
            return self._paste_clipboard()
        else:
            log(f"⚠️  Неизвестный метод копирования: {self.copy_method}")
            return self._paste_clipboard()

    def _paste_clipboard(self):
        """Эмулирует нажатие Ctrl+V для вставки из стандартного буфера обмена"""
        if not shutil.which('wtype'):
            log("⚠️  wtype не найден. Установите wtype: sudo pacman -S wtype")
            return False

        try:
            # wtype -M ctrl -k v -m ctrl
            subprocess.run(['wtype', '-M', 'ctrl', '-k', 'v', '-m', 'ctrl'], check=True)
            log("⌨️  Выполнена вставка из clipboard (Ctrl+V) через wtype")
            return True
        except Exception as e:
            log(f"❌ Ошибка при выполнении wtype: {e}")
            return False

    def _paste_primary(self):
        """Эмулирует нажатие Shift+Insert для вставки из primary selection"""
        if not shutil.which('wtype'):
            log("⚠️  wtype не найден. Установите wtype: sudo pacman -S wtype")
            return False

        try:
            # wtype -M shift -k Insert -m shift
            subprocess.run(['wtype', '-M', 'shift', '-k', 'Insert', '-m', 'shift'], check=True)
            log("⌨️  Выполнена вставка из primary selection (Shift+Insert) через wtype")
            return True
        except Exception as e:
            log(f"❌ Ошибка при выполнении wtype: {e}")
            return False


class SpeechService:
    """Сервис для записи и распознавания речи"""

    def __init__(self, config):
        self.config = config
        self.recording = []
        self.is_recording = False
        self.stream = None
        self.model = None
        self._stream_lock = threading.Lock()

        # Запускаем поток сразу при инициализации
        self._init_stream()

    def _init_stream(self):
        """Инициализирует и запускает постоянно работающий поток"""
        def callback(indata, frames, time, status):
            if status:
                log(f"⚠️  Статус: {status}")

            # Пишем ВСЕГДА, независимо от состояния
            # Но добавляем только если запись активна
            if self.is_recording:
                with self._stream_lock:
                    self.recording.append(indata.copy())

        # Создаём и запускаем поток
        self.stream = sd.InputStream(
            samplerate=self.config.audio.SAMPLE_RATE,
            channels=self.config.audio.CHANNELS,
            dtype=self.config.audio.DTYPE,
            callback=callback
        )
        self.stream.start()
        log("🎤 Аудио-поток инициализирован и прогрет")

    def start(self):
        """Начинает запись аудио"""
        if self.is_recording:
            return False

        log("🎤 Начинаю запись...")

        # Атомарная операция: сначала останавливаем запись, очищаем буфер, затем запускаем
        # Это гарантирует, что callback не добавит старые данные в новый буфер
        with self._stream_lock:
            # Сначала сбрасываем флаг (на всякий случай)
            self.is_recording = False
            # Очищаем буфер от любых остатков
            self.recording = []
            # Только теперь включаем запись - буфер чист
            self.is_recording = True

        log("✅ Запись началась (поток уже был готов)")
        return True

    def stop(self):
        """Останавливает запись БЕЗ распознавания (для перезапуска)"""
        if not self.is_recording:
            return

        log("⏹️  Запись остановлена (без распознавания)")

        # НЕ закрываем поток! Он работает постоянно
        # Атомарно останавливаем запись и очищаем буфер
        with self._stream_lock:
            self.is_recording = False
            self.recording = []

    def stop_and_recognize(self):
        """Останавливает запись и распознаёт речь"""
        if not self.is_recording:
            return None

        log("⏹️  Запись остановлена")

        # НЕ закрываем поток! Он работает постоянно
        # Атомарно останавливаем запись и копируем буфер
        with self._stream_lock:
            self.is_recording = False
            recording_copy = self.recording.copy()

        if not recording_copy:
            log("❌ Ничего не записано")
            return None

        # Объединяем все буферы
        audio_data = np.concatenate(recording_copy, axis=0)
        duration = len(audio_data) / self.config.audio.SAMPLE_RATE
        log(f"✅ Записано {len(audio_data)} сэмплов ({duration:.2f} сек)")

        # Сохраняем в WAV файл
        self._save_wav(audio_data)

        # Распознаём речь
        return self._recognize()

    def _save_wav(self, audio_data):
        """Сохраняет аудио данные в WAV файл"""
        log(f"💾 Сохраняю в {self.config.audio.WAV_FILE}...")

        with wave.open(self.config.audio.WAV_FILE, 'wb') as wf:
            wf.setnchannels(self.config.audio.CHANNELS)
            wf.setsampwidth(self.config.audio.SAMPLE_WIDTH)
            wf.setframerate(self.config.audio.SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        log(f"✅ Файл сохранён")

    def _recognize(self):
        """Распознаёт речь из WAV файла"""
        log(f"🧠 Загружаю модель {self.config.audio.MODEL_NAME}...")

        try:
            if not self.model:
                self.model = onnx_asr.load_model(self.config.audio.MODEL_NAME)
        except Exception as e:
            log(f"❌ Ошибка загрузки модели: {e}")
            log(f"💡 Модель загрузится автоматически при первом запуске")
            return None

        log("🔍 Распознаю речь...")

        try:
            text = self.model.recognize(self.config.audio.WAV_FILE)
            return text
        except Exception as e:
            log(f"❌ Ошибка распознавания: {e}")
            return None


class PostProcessingService:
    """Сервис для пост-обработки текста с помощью LLM"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.prompt = load_prompt_from_file(config.settings.LLM_PROMPT_FILE, "You are a helpful assistant.")

    def process(self, text: str) -> str:
        """Отправляет текст в LLM и возвращает обработанный результат"""
        if not self.config.settings.OPENAI_API_KEY:
            log("⚠️  OPENAI_API_KEY не найден. Пост-обработка отключена.")
            return text

        log(f"🧠 Отправка текста в LLM (модель: {self.config.settings.OPENAI_MODEL})...")

        for attempt in range(self.config.settings.LLM_MAX_RETRIES):
            try:
                with httpx.Client(timeout=self.config.settings.LLM_TIMEOUT_SEC) as client:
                    response = client.post(
                        f"{self.config.settings.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.config.settings.OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.config.settings.OPENAI_MODEL,
                            "messages": [
                                {"role": "system", "content": self.prompt},
                                {"role": "user", "content": text},
                            ],
                            "temperature": self.config.settings.LLM_TEMPERATURE,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()

                    processed_text = result["choices"][0]["message"]["content"].strip()
                    log(f"✅ LLM вернул обработанный текст: {processed_text}")
                    return processed_text

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                log(f"❌ Ошибка при обращении к LLM (попытка {attempt + 1}): {e}")
                if attempt < self.config.settings.LLM_MAX_RETRIES - 1:
                    time.sleep(1)  # Пауза перед повторной попыткой
                continue
            except (KeyError, IndexError) as e:
                log(f"❌ Неожиданный формат ответа от LLM: {e}")
                break  # Не повторяем при ошибках парсинга
            except Exception as e:
                log(f"❌ Неизвестная ошибка при пост-обработке: {e}")
                break  # Не повторяем при других ошибках

        # Fallback - возвращаем исходный текст
        log("⚠️  Не удалось получить ответ от LLM после нескольких попыток.")
        return text


# ============================================================================
# APPLICATION CONTROLLER (БИЗНЕС-ЛОГИКА)
# ============================================================================

class AsyncTaskRunner:
    """Управляет выполнением асинхронных задач в фоновых потоках"""

    # Режим работы: True для синхронного выполнения (для тестов), False для асинхронного (продакшн)
    _sync_mode = False

    @classmethod
    def set_sync_mode(cls, enabled: bool) -> None:
        """
        Включает/выключает синхронный режим (полезно для тестирования)

        Args:
            enabled: True для синхронного выполнения, False для асинхронного
        """
        cls._sync_mode = enabled

    @classmethod
    def run_async(cls, target: Callable, callback: Callable[[any], None]) -> None:
        """
        Запускает задачу в отдельном потоке и возвращает результат в UI-поток

        Args:
            target: Функция для выполнения в фоновом потоке
            callback: Функция для обработки результата в UI-потоке
        """
        if cls._sync_mode:
            # Синхронный режим для тестов - выполняем всё сразу
            result = target()
            callback(result)
        else:
            # Асинхронный режим для продакшна
            def task():
                result = target()
                GLib.idle_add(callback, result)

            thread = threading.Thread(target=task)
            thread.daemon = True
            thread.start()


class ApplicationController:
    """
    Контроллер приложения - управляет бизнес-логикой без привязки к UI

    Отвечает за:
    - Управление записью и распознаванием речи
    - Копирование текста в буфер обмена
    - Автоматическую вставку текста
    - Координацию работы сервисов
    """

    def __init__(
        self,
        config: AppConfig,
        speech: SpeechProtocol,
        clipboard: ClipboardProtocol,
        paste: PasteProtocol,
        post_processing: PostProcessingProtocol,
        state_machine: UIStateMachine
    ):
        """
        Инициализирует контроллер с зависимостями

        Args:
            config: Конфигурация приложения
            speech: Сервис распознавания речи
            clipboard: Сервис буфера обмена
            paste: Сервис вставки текста
            post_processing: Сервис пост-обработки
            state_machine: Машина состояний UI
        """
        self.config = config
        self.speech = speech
        self.clipboard = clipboard
        self.paste_service = paste
        self.post_processing = post_processing
        self.state_machine = state_machine

    def start_recording(self) -> bool:
        """
        Начинает запись речи

        Returns:
            True если запись успешно начата
        """
        if not self.state_machine.is_state(AppState.IDLE):
            log("⚠️  Невозможно начать запись - неправильное состояние")
            return False

        if self.speech.start():
            self.state_machine.transition_to(AppState.RECORDING)
            return True

        return False

    def stop_recording_and_recognize(self, on_complete: Callable[[Optional[str]], None]) -> None:
        """
        Останавливает запись и запускает распознавание речи

        Args:
            on_complete: Callback для обработки результата распознавания
        """
        if not self.state_machine.is_state(AppState.RECORDING):
            log("⚠️  Невозможно остановить запись - запись не идёт")
            return

        self.state_machine.transition_to(AppState.PROCESSING)

        # Запускаем распознавание в фоновом потоке
        AsyncTaskRunner.run_async(
            target=self.speech.stop_and_recognize,
            callback=lambda text: self._on_recognition_complete(text, on_complete)
        )

    def restart_recording(self, on_complete: Callable[[bool], None]) -> None:
        """
        Перезапускает запись (сбрасывает текущую и начинает новую)

        Args:
            on_complete: Callback для обработки результата перезапуска
        """
        if not self.state_machine.is_state(AppState.RECORDING):
            log("⚠️  Невозможно перезапустить - запись не идёт")
            return

        log("🔄 Сброс записи и перезапуск...")
        self.state_machine.transition_to(AppState.RESTARTING)

        # Запускаем перезапуск в фоновом потоке
        AsyncTaskRunner.run_async(
            target=self._restart_recording_task,
            callback=lambda success: self._on_restart_complete(success, on_complete)
        )

    def _restart_recording_task(self) -> bool:
        """Задача перезапуска записи (выполняется в фоновом потоке)"""
        # Останавливаем текущую запись БЕЗ распознавания
        self.speech.stop()

        # Небольшая пауза для освобождения ресурсов
        time.sleep(AppSettings.RESTART_DELAY_SEC)

        # Запускаем новую запись
        return self.speech.start()

    def _on_restart_complete(self, success: bool, callback: Callable[[bool], None]) -> None:
        """Обработчик завершения перезапуска"""
        if success:
            log("✅ Запись успешно перезапущена")
            self.state_machine.transition_to(AppState.RECORDING)
        else:
            log("❌ Не удалось перезапустить запись")
            self.state_machine.transition_to(AppState.IDLE)

        callback(success)

    def _on_recognition_complete(self, text: Optional[str], callback: Callable[[Optional[str]], None]) -> None:
        """Обработчик завершения распознавания"""
        if not text:
            self.state_machine.transition_to(AppState.IDLE)
            callback(None)
            return

        log(f"🎤 Распознанный текст: {text}")

        if self.config.settings.LLM_ENABLED:
            self.state_machine.transition_to(AppState.POST_PROCESSING)
            AsyncTaskRunner.run_async(
                target=lambda: self.post_processing.process(text),
                callback=lambda processed_text: self._on_post_processing_complete(processed_text + " \n", callback)
            )

        self._on_post_processing_complete(text, callback)

    def _process_short_text(self, text: str) -> str:
        """Обрабатывает короткий текст (1-2 слова) без LM"""
        if not text:
            return text

        # Первая буква маленькая
        processed = text.lower()

        # Удаляем точку в конце, если есть
        processed = processed.rstrip('.')

        log(f"🔧 Обработка короткой фразы")
        return processed

    def _process_long_text(self, text: str) -> str:
        """Обрабатывает короткий текст (1-2 слова) без LM"""
        if not text:
            return text

        log(f"🔧 Обработка длинной фразы")
        return text + ' \n'

    def _on_post_processing_complete(self, text: str, callback: Callable[[str], None]) -> None:
        """Обработчик завершения пост-обработки"""

        if self.config.settings.SMART_TEXT_PROCESSING:
            # Проверяем количество слов
            word_count = len(text.split())
            log(f"📊 Количество слов: {word_count}")

            if word_count <= self.config.settings.SMART_TEXT_SHORT_PHRASE:
                text = self._process_short_text(text)
            else:
                text = self._process_long_text(text)

        self._copy_paste_text(text)
        self.state_machine.transition_to(AppState.IDLE)
        callback(text)

    def _copy_paste_text(self, text: str) -> None:
        """Копирует текст в буфер обмена и вставляет его при необходимости"""
        if self.config.settings.COPY_METHOD == "clipboard":
            self.clipboard.copy_standard(text)
        elif self.config.settings.COPY_METHOD == "primary":
            self.clipboard.copy_primary(text)

        if self.config.settings.AUTO_PASTE:
            GLib.timeout_add(
                AppSettings.PASTE_DELAY_MS,
                lambda: (self.paste_service.paste(), False)[1]
            )


# ============================================================================
# UI
# ============================================================================

class RecognitionWindow:
    """
    Плавающее окно для записи и распознавания речи

    Отвечает только за:
    - Создание и настройку UI элементов
    - Обработку событий UI (клики, drag-and-drop)
    - Синхронизацию состояния UI с состоянием приложения

    Вся бизнес-логика делегируется в ApplicationController
    """

    def __init__(
        self,
        config: AppConfig,
        controller: ApplicationController,
        state_machine: UIStateMachine
    ):
        """
        Инициализирует окно с внедрёнными зависимостями

        Args:
            config: Конфигурация приложения
            controller: Контроллер бизнес-логики
            state_machine: Машина состояний UI
        """
        self.config = config
        self.controller = controller
        self.state_machine = state_machine

        self.window = None
        self.button = None
        self.restart_button = None
        self.pp_button = None
        self.app = None

        # Для drag-and-drop
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.is_dragging = False
        self.was_moved = False
        self.window_x = self.config.ui.DEFAULT_WINDOW_X
        self.window_y = self.config.ui.DEFAULT_WINDOW_Y

        # Подписываемся на изменения состояния
        self.state_machine.add_observer(self._on_state_changed)

    @classmethod
    def create_with_defaults(cls, factory: ServiceFactory = None) -> 'RecognitionWindow':
        """
        Фабричный метод для создания окна с дефолтными зависимостями

        Args:
            factory: Фабрика сервисов для DI (по умолчанию создается с дефолтными реализациями)

        Returns:
            Настроенный экземпляр RecognitionWindow
        """
        config = AppConfig()

        # Создаём фабрику с возможностью инъекции зависимостей
        if factory is None:
            factory = ServiceFactory()

        speech, clipboard, paste, post_processing = factory.create_all_services(config)
        state_machine = UIStateMachine()
        controller = ApplicationController(config, speech, clipboard, paste, post_processing, state_machine)
        return cls(config, controller, state_machine)

    def _update_record_button(self, label: str, is_sensitive: bool = True):
        """
        Обновляет состояние кнопки записи (лейбл и чувствительность)

        Args:
            label: Текст лейбла кнопки
            is_sensitive: True если кнопка активна, False если отключена
        """
        if not self.button:
            return

        self.button.set_label(label)
        self.button.set_sensitive(is_sensitive)

    def _update_restart_button(self, label: str, is_restart: bool, is_sensitive: bool = True):
        """
        Полностью обновляет состояние кнопки рестарта (лейбл, класс и чувствительность)

        Args:
            label: Текст лейбла кнопки
            is_restart: True для класса restart-button (перезапуск), False для close-button (закрытие)
            is_sensitive: True если кнопка активна, False если отключена
        """
        if not self.restart_button:
            return

        self.restart_button.set_label(label)
        self.restart_button.set_sensitive(is_sensitive)

        # Переключаем CSS класс
        style_context = self.restart_button.get_style_context()
        if is_restart:
            style_context.remove_class("close-button")
            style_context.add_class("restart-button")
        else:
            style_context.remove_class("restart-button")
            style_context.add_class("close-button")


    def _on_state_changed(self, old_state: AppState, new_state: AppState):
        """Обработчик изменения состояния - обновляет UI"""
        if new_state == AppState.IDLE:
            self._update_ui_for_idle_state()
        elif new_state == AppState.RECORDING:
            self._update_ui_for_recording_state()
        elif new_state == AppState.PROCESSING:
            self._update_ui_for_processing_state()
        elif new_state == AppState.POST_PROCESSING:
            self._update_ui_for_processing_state() # Same as processing
        elif new_state == AppState.RESTARTING:
            self._update_ui_for_restarting_state()

    def _update_ui_for_idle_state(self):
        """Обновляет UI для состояния IDLE (готов к записи)"""
        self._update_record_button(self.config.ui.ICON_RECORD, is_sensitive=True)
        self._update_restart_button(self.config.ui.ICON_CLOSE, is_restart=False, is_sensitive=True)

    def _update_ui_for_recording_state(self):
        """Обновляет UI для состояния RECORDING (идёт запись)"""
        self._update_record_button(self.config.ui.ICON_STOP, is_sensitive=True)
        self._update_restart_button(self.config.ui.ICON_RESTART, is_restart=True, is_sensitive=True)

    def _update_ui_for_processing_state(self):
        """Обновляет UI для состояния PROCESSING (обработка)"""
        self._update_record_button(self.config.ui.ICON_PROCESSING, is_sensitive=False)
        self._update_restart_button(self.config.ui.ICON_CLOSE, is_restart=False, is_sensitive=False)

    def _update_ui_for_restarting_state(self):
        """Обновляет UI для состояния RESTARTING (перезапуск записи)"""
        self._update_record_button(self.config.ui.ICON_PROCESSING, is_sensitive=False)
        self._update_restart_button(self.config.ui.ICON_RESTART, is_restart=True, is_sensitive=False)

    def on_button_press(self, widget, event):
        """Обработчик начала перетаскивания"""
        if event.button == self.config.ui.MOUSE_BUTTON_LEFT:
            self.is_dragging = True
            self.was_moved = False  # Флаг фактического перемещения
            self.drag_start_x = event.x_root
            self.drag_start_y = event.y_root

    def on_button_release(self, widget, event):
        """Обработчик окончания перетаскивания"""
        if event.button == self.config.ui.MOUSE_BUTTON_LEFT:
            self.is_dragging = False
            # Сохраняем позицию только если окно действительно перемещалось
            if self.was_moved:
                self.config.window.save(self.window_x, self.window_y)
            self.was_moved = False

    def on_motion_notify(self, widget, event):
        """Обработчик перемещения мыши при перетаскивании"""
        if self.is_dragging:
            # Вычисляем смещение
            dx = event.x_root - self.drag_start_x
            dy = event.y_root - self.drag_start_y

            # Обновляем позицию
            # Инвертируем dx, так как окно привязано к правому краю
            self.window_x -= dx
            self.window_y += dy

            # Устанавливаем флаг, что окно было перемещено
            self.was_moved = True

            # Обновляем позицию окна через margins
            GtkLayerShell.set_margin(self.window, GtkLayerShell.Edge.TOP, int(self.window_y))
            GtkLayerShell.set_margin(self.window, GtkLayerShell.Edge.RIGHT, int(self.window_x))

            # Обновляем начальную позицию для следующего движения
            self.drag_start_x = event.x_root
            self.drag_start_y = event.y_root


    def on_restart_clicked(self, button):
        """Обработчик нажатия кнопки перезапуска/закрытия"""
        if self.state_machine.is_state(AppState.RECORDING):
            # Если идёт запись - делегируем перезапуск контроллеру
            self.controller.restart_recording(on_complete=lambda success: None)
        else:
            # Если не идёт запись - закрываем приложение
            log("🛑 Закрытие приложения...")
            if self.app:
                self.app.quit()

    def on_pp_clicked(self, button):
        """Обработчик нажатия кнопки пост-обработки"""
        # Переключаем состояние пост-обработки
        self.config.settings.LLM_ENABLED = not self.config.settings.LLM_ENABLED

        # Обновляем иконку кнопки
        if self.config.settings.LLM_ENABLED:
            self.pp_button.set_label(self.config.ui.ICON_PP_ON)
            log("✅ Пост-обработка включена")
        else:
            self.pp_button.set_label(self.config.ui.ICON_PP_OFF)
            log("⬜ Пост-обработка выключена")

    def _setup_css_styles(self, screen):
        """Настраивает CSS стили для окна"""
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(self.config.ui.CSS_STYLES)
        Gtk.StyleContext.add_provider_for_screen(
            screen,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _setup_transparency(self):
        """Настраивает прозрачность окна"""
        screen = self.window.get_screen()
        visual = screen.get_rgba_visual()
        if visual:
            self.window.set_visual(visual)
        # НЕ устанавливаем set_app_paintable(True) - позволяем GTK рисовать фон с CSS стилями
        return screen

    def _setup_wayland_layer(self):
        """Настраивает Wayland Layer Shell"""
        GtkLayerShell.init_for_window(self.window)

        # Привязываем к верхнему правому углу
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.TOP, True)
        GtkLayerShell.set_anchor(self.window, GtkLayerShell.Edge.RIGHT, True)

        # Устанавливаем отступы из сохранённой позиции
        GtkLayerShell.set_margin(self.window, GtkLayerShell.Edge.TOP, int(self.window_y))
        GtkLayerShell.set_margin(self.window, GtkLayerShell.Edge.RIGHT, int(self.window_x))

        # Устанавливаем слой поверх всего
        GtkLayerShell.set_layer(self.window, GtkLayerShell.Layer.OVERLAY)

    def _setup_drag_and_drop(self):
        """Настраивает обработчики для drag-and-drop"""
        self.window.add_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                              Gdk.EventMask.BUTTON_RELEASE_MASK |
                              Gdk.EventMask.POINTER_MOTION_MASK)
        self.window.connect("button-press-event", self.on_button_press)
        self.window.connect("button-release-event", self.on_button_release)
        self.window.connect("motion-notify-event", self.on_motion_notify)

    def _create_ui_elements(self, app):
        """Создаёт UI элементы (кнопки)"""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=self.config.ui.BOX_SPACING)
        box.set_margin_top(self.config.ui.BOX_MARGIN)
        box.set_margin_bottom(self.config.ui.BOX_MARGIN)
        box.set_margin_start(self.config.ui.BOX_MARGIN)
        box.set_margin_end(self.config.ui.BOX_MARGIN)

        # Кнопка перезапуска записи (изначально показываем закрытие)
        self.restart_button = Gtk.Button(label=self.config.ui.ICON_CLOSE)
        self.restart_button.get_style_context().add_class("close-button")
        self.restart_button.connect("clicked", self.on_restart_clicked)

        # Кнопка записи
        self.button = Gtk.Button(label=self.config.ui.ICON_RECORD)
        self.button.get_style_context().add_class("record-button")
        self.button.connect("clicked", self.on_button_clicked)

        # Кнопка пост-обработки
        initial_pp_icon = (self.config.ui.ICON_PP_ON
                                 if self.config.settings.LLM_ENABLED
                                 else self.config.ui.ICON_PP_OFF)
        self.pp_button = Gtk.Button(label=initial_pp_icon)
        self.pp_button.get_style_context().add_class("autopaste-button") # Keep old class for styles
        self.pp_button.connect("clicked", self.on_pp_clicked)

        # Сохраняем ссылку на app для возможности закрытия приложения
        self.app = app

        box.add(self.restart_button)
        box.add(self.button)
        box.add(self.pp_button)

        return box



    def on_button_clicked(self, button):
        """Обработчик нажатия кнопки"""
        if self.state_machine.is_state(AppState.IDLE):
            # Начинаем запись через контроллер
            self.controller.start_recording()
        elif self.state_machine.is_state(AppState.RECORDING):
            # Останавливаем запись и распознаём через контроллер
            self.controller.stop_recording_and_recognize(
                on_complete=lambda text: None  # UI обновляется через observer
            )

    def on_activate(self, app):
        """Создает и настраивает окно"""
        # Загружаем сохранённую позицию окна
        self.window_x, self.window_y = self.config.window.load()

        # Создаем окно
        self.window = Gtk.ApplicationWindow(application=app)

        # Настройка прозрачности
        screen = self._setup_transparency()

        # Настройка Wayland Layer
        self._setup_wayland_layer()

        # CSS стили
        self._setup_css_styles(screen)

        # Drag-and-drop
        self._setup_drag_and_drop()

        # Создаем UI элементы
        box = self._create_ui_elements(app)

        self.window.add(box)
        self.window.show_all()


def main():
    """Основная функция"""
    log("=" * 50)
    log("🎙️  Распознавание речи с микрофона")
    log("=" * 50)

    recognition_window = RecognitionWindow.create_with_defaults()
    app = Gtk.Application(application_id=AppConfig.settings.APP_ID)
    app.connect('activate', recognition_window.on_activate)

    # Обработчик Ctrl+C для корректного завершения
    def signal_handler(sig, frame):
        log("\n⚠️  Получен сигнал прерывания (Ctrl+C)")
        log("🛑 Останавливаю приложение...")

        # Останавливаем запись если она идёт
        if recognition_window.controller.speech.is_recording:
            recognition_window.controller.speech.stop_and_recognize()

        # Завершаем приложение
        app.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    log("💡 Нажмите Ctrl+C для выхода")

    app.run(None)



if __name__ == "__main__":
    main()
