import importlib.util
import json
import queue
import threading
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

if importlib.util.find_spec("tkinterdnd2"):
    from tkinterdnd2 import DND_FILES, TkinterDnD
else:
    TkinterDnD = None
    DND_FILES = None

if importlib.util.find_spec("torch"):
    import torch
else:
    torch = None

if importlib.util.find_spec("whisper"):
    import whisper
    from whisper.tokenizer import LANGUAGES
else:
    whisper = None
    LANGUAGES = {}


APP_TITLE = "Whisper Transcription Studio"
ACCENT = "#007AFF"
LIGHT_BG = "#F5F5F7"
PANEL_BG = "#FFFFFF"
TEXT_BG = "#FFFFFF"
TEXT_FG = "#1D1D1F"
SUBTLE_FG = "#6E6E73"
CONFIG_PATH = Path.home() / ".whispergui.json"

MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "turbo",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
]


@dataclass
class TranscriptionResult:
    text: str
    segments: List[dict]
    language: str


class Tooltip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, _event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#FFFFE0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("SF Pro Display", 10),
        )
        label.pack(ipadx=6, ipady=4)

    def hide_tip(self, _event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Widget, **kwargs):
        super().__init__(master, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, background=LIGHT_BG)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _resize(event):
            canvas.itemconfig(window, width=event.width)

        canvas.bind("<Configure>", _resize)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class WhisperApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.configure(bg=LIGHT_BG)
        self.root.geometry("1220x760")
        self.root.minsize(1100, 700)

        self.model_cache: Dict[str, object] = {}
        self.transcribe_thread: Optional[threading.Thread] = None
        self.queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.start_time: Optional[float] = None
        self.file_path: Optional[Path] = None

        self.device = self.detect_device()
        self.config = self.load_config()

        self.setup_style()
        self.build_layout()
        self.load_config_into_ui()
        self.update_status("Ready")
        self.poll_queue()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TFrame",
            background=LIGHT_BG,
        )
        style.configure(
            "Panel.TFrame",
            background=PANEL_BG,
        )
        style.configure(
            "TLabel",
            background=PANEL_BG,
            foreground=TEXT_FG,
            font=("SF Pro Display", 11),
        )
        style.configure(
            "Title.TLabel",
            background=LIGHT_BG,
            foreground=TEXT_FG,
            font=("SF Pro Display", 18, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=LIGHT_BG,
            foreground=SUBTLE_FG,
            font=("SF Pro Display", 11),
        )
        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="white",
            font=("SF Pro Display", 11, "bold"),
            padding=(18, 8),
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#0060D1")],
        )
        style.configure(
            "Pill.TButton",
            background="#EFEFF4",
            foreground=TEXT_FG,
            font=("SF Pro Display", 10, "bold"),
            padding=(14, 6),
            borderwidth=0,
        )
        style.map(
            "Pill.TButton",
            background=[("active", "#E5E5EA")],
        )
        style.configure(
            "TCombobox",
            padding=6,
            relief="flat",
        )
        style.configure(
            "TCheckbutton",
            background=PANEL_BG,
            font=("SF Pro Display", 10),
        )

    def build_layout(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=28, pady=(20, 6))
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Professional transcription powered by Whisper", style="Subtitle.TLabel").pack(anchor="w")

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=20, pady=10)

        self.settings_panel = ScrollableFrame(main)
        self.settings_panel.pack(side="left", fill="y", padx=(0, 16))

        self.content_panel = ttk.Frame(main, style="Panel.TFrame")
        self.content_panel.pack(side="left", fill="both", expand=True)

        self.build_settings_panel(self.settings_panel.scrollable_frame)
        self.build_content_panel(self.content_panel)
        self.build_status_bar()

    def build_settings_panel(self, panel: ttk.Frame):
        section = ttk.Label(panel, text="Settings", font=("SF Pro Display", 13, "bold"))
        section.pack(anchor="w", pady=(8, 4))

        self.model_var = tk.StringVar()
        self.language_var = tk.StringVar()
        self.task_var = tk.StringVar(value="transcribe")
        self.temperature_var = tk.DoubleVar(value=0.0)
        self.beam_size_var = tk.IntVar(value=5)
        self.best_of_var = tk.IntVar(value=5)
        self.patience_var = tk.DoubleVar(value=1.0)
        self.fp16_var = tk.BooleanVar(value=True)
        self.word_ts_var = tk.BooleanVar(value=False)
        self.condition_prev_var = tk.BooleanVar(value=True)
        self.max_line_width_var = tk.IntVar(value=42)
        self.max_line_count_var = tk.IntVar(value=2)

        self.add_labeled_combobox(panel, "Model", self.model_var, MODEL_OPTIONS, "Whisper model size & speed tradeoff")
        languages = self.get_language_options()
        self.add_labeled_combobox(panel, "Language", self.language_var, languages, "Auto-detect or choose a language")

        task_frame = ttk.Frame(panel, style="Panel.TFrame")
        task_frame.pack(fill="x", pady=6)
        ttk.Label(task_frame, text="Mode", font=("SF Pro Display", 11, "bold")).pack(anchor="w")
        mode_container = ttk.Frame(task_frame, style="Panel.TFrame")
        mode_container.pack(anchor="w", pady=4)
        ttk.Radiobutton(mode_container, text="Transcribe", value="transcribe", variable=self.task_var).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(mode_container, text="Translate", value="translate", variable=self.task_var).pack(side="left")
        Tooltip(task_frame, "Translate converts speech to English")

        self.add_slider(panel, "Temperature", self.temperature_var, 0.0, 1.0, 0.1, "Creativity vs. accuracy trade-off")
        self.add_spinbox(panel, "Beam Size", self.beam_size_var, 1, 10, "Beam search width for decoding")
        self.add_spinbox(panel, "Best Of", self.best_of_var, 1, 10, "Number of candidates to sample")
        self.add_slider(panel, "Patience", self.patience_var, 0.1, 2.0, 0.1, "Beam search patience level")

        self.add_checkbox(panel, "FP16 (GPU acceleration)", self.fp16_var, "Use half precision on compatible GPUs")
        self.add_checkbox(panel, "Word-level timestamps", self.word_ts_var, "More detailed timestamps per word")
        self.add_checkbox(panel, "Condition on previous text", self.condition_prev_var, "Improves consistency across segments")

        self.add_spinbox(panel, "Max Line Width", self.max_line_width_var, 20, 80, "Max characters per line in output")
        self.add_spinbox(panel, "Max Line Count", self.max_line_count_var, 1, 5, "Max lines per subtitle block")

    def build_content_panel(self, panel: ttk.Frame):
        panel.configure(padding=16)

        top_row = ttk.Frame(panel, style="Panel.TFrame")
        top_row.pack(fill="x")
        self.file_label = ttk.Label(top_row, text="Drop a file or click Browse", font=("SF Pro Display", 12))
        self.file_label.pack(side="left")

        browse_btn = ttk.Button(top_row, text="Browse", style="Pill.TButton", command=self.browse_file)
        browse_btn.pack(side="right")
        Tooltip(browse_btn, "Select an audio or video file")

        self.drop_area = tk.Label(
            panel,
            text="Drag & drop audio/video here",
            bg="#F0F0F3",
            fg=SUBTLE_FG,
            height=4,
            relief="ridge",
            bd=0,
            font=("SF Pro Display", 11),
        )
        self.drop_area.pack(fill="x", pady=12)

        if TkinterDnD and DND_FILES:
            self.drop_area.drop_target_register(DND_FILES)
            self.drop_area.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.drop_area.configure(text="Drag & drop unavailable (tkinterdnd2 not installed)")

        action_row = ttk.Frame(panel, style="Panel.TFrame")
        action_row.pack(fill="x", pady=(4, 8))
        self.transcribe_btn = ttk.Button(action_row, text="Start Transcription", style="Accent.TButton", command=self.start_transcription)
        self.transcribe_btn.pack(side="left")
        self.clear_btn = ttk.Button(action_row, text="Clear", style="Pill.TButton", command=self.clear_output)
        self.clear_btn.pack(side="left", padx=10)

        info_panel = ttk.Frame(panel, style="Panel.TFrame")
        info_panel.pack(fill="x", pady=(4, 10))
        self.info_var = tk.StringVar(value="Model: — | Language: — | Device: — | Progress: 0% | Time: 00:00")
        self.info_label = ttk.Label(info_panel, textvariable=self.info_var, foreground=SUBTLE_FG)
        self.info_label.pack(anchor="w")

        self.output_text = tk.Text(
            panel,
            bg=TEXT_BG,
            fg=TEXT_FG,
            wrap="word",
            font=("SF Pro Display", 11),
            height=18,
            bd=0,
            relief="flat",
        )
        self.output_text.pack(fill="both", expand=True, pady=(6, 10))

        output_actions = ttk.Frame(panel, style="Panel.TFrame")
        output_actions.pack(fill="x")
        ttk.Button(output_actions, text="Copy", style="Pill.TButton", command=self.copy_output).pack(side="left")
        ttk.Button(output_actions, text="Save TXT", style="Pill.TButton", command=self.save_txt).pack(side="left", padx=8)
        ttk.Button(output_actions, text="Save SRT", style="Pill.TButton", command=self.save_srt).pack(side="left")

    def build_status_bar(self):
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill="x", side="bottom", padx=18, pady=(0, 10))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_bar, textvariable=self.status_var, foreground=SUBTLE_FG).pack(anchor="w")

    def add_labeled_combobox(self, parent: ttk.Frame, label: str, variable: tk.StringVar, values: List[str], tooltip: str):
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=6)
        ttk.Label(frame, text=label, font=("SF Pro Display", 11, "bold")).pack(anchor="w")
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly")
        combo.pack(fill="x", pady=4)
        Tooltip(combo, tooltip)

    def add_slider(self, parent: ttk.Frame, label: str, variable: tk.DoubleVar, min_val: float, max_val: float, step: float, tooltip: str):
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=6)
        ttk.Label(frame, text=label, font=("SF Pro Display", 11, "bold")).pack(anchor="w")
        slider = ttk.Scale(frame, variable=variable, from_=min_val, to=max_val)
        slider.pack(fill="x", pady=4)
        Tooltip(slider, tooltip)

    def add_spinbox(self, parent: ttk.Frame, label: str, variable: tk.IntVar, min_val: int, max_val: int, tooltip: str):
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=6)
        ttk.Label(frame, text=label, font=("SF Pro Display", 11, "bold")).pack(anchor="w")
        spin = ttk.Spinbox(frame, from_=min_val, to=max_val, textvariable=variable, width=6)
        spin.pack(anchor="w", pady=4)
        Tooltip(spin, tooltip)

    def add_checkbox(self, parent: ttk.Frame, label: str, variable: tk.BooleanVar, tooltip: str):
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=4)
        checkbox = ttk.Checkbutton(frame, text=label, variable=variable)
        checkbox.pack(anchor="w")
        Tooltip(checkbox, tooltip)

    def detect_device(self) -> str:
        if torch and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def get_language_options(self) -> List[str]:
        options = ["Auto Detect"]
        if LANGUAGES:
            options.extend(sorted({name.title() for name in LANGUAGES.values()}))
        else:
            options.extend(["English", "Spanish", "French", "German", "Japanese"])
        return options

    def load_config(self) -> dict:
        if CONFIG_PATH.exists():
            try:
                return json.loads(CONFIG_PATH.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def save_config(self):
        data = {
            "model": self.model_var.get(),
            "language": self.language_var.get(),
            "task": self.task_var.get(),
            "temperature": self.temperature_var.get(),
            "beam_size": self.beam_size_var.get(),
            "best_of": self.best_of_var.get(),
            "patience": self.patience_var.get(),
            "fp16": self.fp16_var.get(),
            "word_timestamps": self.word_ts_var.get(),
            "condition_on_previous": self.condition_prev_var.get(),
            "max_line_width": self.max_line_width_var.get(),
            "max_line_count": self.max_line_count_var.get(),
        }
        CONFIG_PATH.write_text(json.dumps(data, indent=2))

    def load_config_into_ui(self):
        self.model_var.set(self.config.get("model", MODEL_OPTIONS[0]))
        self.language_var.set(self.config.get("language", "Auto Detect"))
        self.task_var.set(self.config.get("task", "transcribe"))
        self.temperature_var.set(self.config.get("temperature", 0.0))
        self.beam_size_var.set(self.config.get("beam_size", 5))
        self.best_of_var.set(self.config.get("best_of", 5))
        self.patience_var.set(self.config.get("patience", 1.0))
        self.fp16_var.set(self.config.get("fp16", True))
        self.word_ts_var.set(self.config.get("word_timestamps", False))
        self.condition_prev_var.set(self.config.get("condition_on_previous", True))
        self.max_line_width_var.set(self.config.get("max_line_width", 42))
        self.max_line_count_var.set(self.config.get("max_line_count", 2))

    def on_drop(self, event):
        path = event.data.strip("{}")
        self.set_file(Path(path))

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select audio/video file",
            filetypes=[
                ("Media files", "*.mp3 *.wav *.m4a *.mp4 *.mov *.flac *.aac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.set_file(Path(filename))

    def set_file(self, path: Path):
        self.file_path = path
        self.file_label.configure(text=path.name)
        self.update_status("File ready for transcription")

    def start_transcription(self):
        if not self.file_path:
            messagebox.showwarning("No file", "Please select an audio or video file.")
            return
        if whisper is None:
            messagebox.showerror("Missing dependency", "The 'whisper' package is not installed.")
            return

        if self.transcribe_thread and self.transcribe_thread.is_alive():
            messagebox.showinfo("Processing", "A transcription is already running.")
            return

        self.save_config()
        self.output_text.delete("1.0", tk.END)
        self.transcribe_btn.state(["disabled"])
        self.start_time = time.time()
        self.update_status("Transcribing...")
        self.queue.put(("progress", 0))
        self.transcribe_thread = threading.Thread(target=self.transcribe_worker, daemon=True)
        self.transcribe_thread.start()

    def transcribe_worker(self):
        try:
            model_name = self.model_var.get()
            model = self.load_model(model_name)
            self.queue.put(("model", model_name))

            options = {
                "temperature": float(self.temperature_var.get()),
                "beam_size": int(self.beam_size_var.get()),
                "best_of": int(self.best_of_var.get()),
                "patience": float(self.patience_var.get()),
                "fp16": bool(self.fp16_var.get()) if self.device == "cuda" else False,
                "condition_on_previous_text": bool(self.condition_prev_var.get()),
                "word_timestamps": bool(self.word_ts_var.get()),
                "task": self.task_var.get(),
            }

            language_label = self.language_var.get()
            if language_label != "Auto Detect" and LANGUAGES:
                language_code = self.language_name_to_code(language_label)
                if language_code:
                    options["language"] = language_code

            def progress_hook(progress):
                self.queue.put(("progress", progress))

            result = model.transcribe(
                str(self.file_path),
                verbose=False,
                **options,
            )
            self.queue.put(("done", result))
        except Exception as exc:  # pylint: disable=broad-except
            self.queue.put(("error", str(exc)))

    def load_model(self, name: str):
        if name in self.model_cache:
            return self.model_cache[name]
        model = whisper.load_model(name, device=self.device)
        self.model_cache[name] = model
        return model

    def poll_queue(self):
        try:
            while True:
                message, payload = self.queue.get_nowait()
                if message == "progress":
                    self.update_progress(payload)
                elif message == "model":
                    self.update_info(model=payload)
                elif message == "done":
                    self.handle_result(payload)
                elif message == "error":
                    self.handle_error(payload)
        except queue.Empty:
            pass
        self.root.after(200, self.poll_queue)

    def update_progress(self, progress: float):
        if self.start_time:
            elapsed = time.time() - self.start_time
            time_str = time.strftime("%M:%S", time.gmtime(elapsed))
        else:
            time_str = "00:00"
        progress_pct = int(progress * 100)
        self.update_info(progress=progress_pct, timer=time_str)

    def update_info(self, model: Optional[str] = None, language: Optional[str] = None, progress: Optional[int] = None, timer: Optional[str] = None):
        current = self.info_var.get().split(" | ")
        info = {
            "Model": model or current[0].split(": ")[1],
            "Language": language or current[1].split(": ")[1],
            "Device": current[2].split(": ")[1],
            "Progress": f"{progress if progress is not None else current[3].split(': ')[1]}%",
            "Time": timer or current[4].split(": ")[1],
        }
        self.info_var.set(
            f"Model: {info['Model']} | Language: {info['Language']} | Device: {info['Device']} | Progress: {info['Progress']} | Time: {info['Time']}"
        )

    def handle_result(self, result: dict):
        self.transcribe_btn.state(["!disabled"])
        self.update_status("Transcription complete")
        language = self.language_display(result.get("language", ""))
        self.update_info(language=language, progress=100)
        formatted = self.build_display_output(result)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, formatted)
        self.latest_result = TranscriptionResult(
            text=result.get("text", ""),
            segments=result.get("segments", []),
            language=language,
        )

    def handle_error(self, message: str):
        self.transcribe_btn.state(["!disabled"])
        self.update_status("Error")
        messagebox.showerror("Transcription failed", message)

    def update_status(self, text: str):
        self.status_var.set(text)
        self.info_var.set(
            f"Model: {self.model_var.get()} | Language: {self.language_var.get()} | Device: {self.device} | Progress: 0% | Time: 00:00"
        )

    def build_display_output(self, result: dict) -> str:
        srt_text = self.render_srt(result)
        blocks = self.parse_srt_blocks(srt_text)
        lines = []
        for start, end, text_lines in blocks:
            start_display = start.replace(",", ".")[3:]
            end_display = end.replace(",", ".")[3:]
            joined_text = "\n".join(text_lines)
            lines.append(f"[{start_display} --> {end_display}]\n{joined_text}\n")
        return "\n".join(lines).strip() + "\n"

    def copy_output(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.update_status("Copied to clipboard")

    def save_txt(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("No data", "Nothing to save yet.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if filename:
            Path(filename).write_text(text, encoding="utf-8")
            self.update_status("TXT saved")

    def save_srt(self):
        if not hasattr(self, "latest_result"):
            messagebox.showinfo("No data", "Run a transcription first.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".srt", filetypes=[("SubRip", "*.srt")])
        if filename:
            srt_text = self.build_srt(self.latest_result)
            Path(filename).write_text(srt_text, encoding="utf-8")
            self.update_status("SRT saved")

    def build_srt(self, result: TranscriptionResult) -> str:
        lines = []
        header_text = [
            f"FILE: {self.file_path.name if self.file_path else 'Unknown'}",
            f"MODEL: {self.model_var.get()}",
            f"LANGUAGE: {result.language}",
            "BY: PUSHKAR",
        ]
        lines.append("1")
        lines.append("00:00:00,000 --> 00:00:00,000")
        lines.extend(header_text)
        lines.append("")

        srt_text = self.render_srt(
            {
                "segments": result.segments,
            }
        )
        blocks = self.parse_srt_blocks(srt_text)
        index = 2
        for start, end, text_lines in blocks:
            lines.append(str(index))
            lines.append(f"{start} --> {end}")
            lines.extend(text_lines)
            lines.append("")
            index += 1
        return "\n".join(lines).strip() + "\n"

    def render_srt(self, result: dict) -> str:
        if whisper is None:
            return ""
        writer_factory = getattr(whisper.utils, "get_writer", None)
        if writer_factory and self.file_path:
            with tempfile.TemporaryDirectory() as temp_dir:
                writer = writer_factory("srt", temp_dir)
                writer(
                    result,
                    str(self.file_path),
                    max_line_width=int(self.max_line_width_var.get()),
                    max_line_count=int(self.max_line_count_var.get()),
                    highlight_words=False,
                )
                output_path = Path(temp_dir) / f"{self.file_path.stem}.srt"
                if output_path.exists():
                    return output_path.read_text(encoding="utf-8").strip()
        return self.manual_srt(result)

    def manual_srt(self, result: dict) -> str:
        lines = []
        for index, segment in enumerate(result.get("segments", []), start=1):
            start = self.format_srt_timestamp(segment.get("start", 0.0))
            end = self.format_srt_timestamp(segment.get("end", 0.0))
            text = segment.get("text", "").strip()
            lines.append(str(index))
            lines.append(f"{start} --> {end}")
            lines.append(text)
            lines.append("")
        return "\n".join(lines).strip()

    def format_srt_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millis = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"

    def parse_srt_blocks(self, srt_text: str) -> List[Tuple[str, str, List[str]]]:
        blocks: List[Tuple[str, str, List[str]]] = []
        if not srt_text:
            return blocks
        for block in srt_text.strip().split("\n\n"):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 2:
                continue
            time_line = lines[1] if "-->" in lines[1] else lines[0]
            if "-->" not in time_line:
                continue
            start, end = [part.strip() for part in time_line.split("-->")]
            text_lines = lines[2:] if "-->" in lines[1] else lines[1:]
            blocks.append((start, end, text_lines))
        return blocks

    def clear_output(self):
        self.output_text.delete("1.0", tk.END)
        self.file_label.configure(text="Drop a file or click Browse")
        self.file_path = None
        self.update_status("Cleared")

    def language_name_to_code(self, language_name: str) -> Optional[str]:
        for code, name in LANGUAGES.items():
            if name.lower() == language_name.lower():
                return code
        return None

    def language_display(self, code: str) -> str:
        if not code:
            return "Unknown"
        return LANGUAGES.get(code, code).title()


def main():
    root_class = TkinterDnD.Tk if TkinterDnD else tk.Tk
    root = root_class()
    app = WhisperApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
