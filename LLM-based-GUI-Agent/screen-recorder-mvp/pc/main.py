# -*- coding: utf-8 -*-
"""PC main program: screen recording + receive phone uploads + GUI Agent video analysis"""
import os
import re
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recorder import ScreenRecorder
from receiver import run_receiver
from config import (
    DASHSCOPE_API_KEY,
    FRAME_SAMPLE_INTERVAL_SEC,
    GUI_AGENT_BACKEND,
    LOCAL_GUI_OWL_MODEL_ID,
    SSIM_THRESHOLD,
)
from version import __version__

RECEIVER_PORT = 8765


def _configure_windows_dpi_before_tk():
    """Fix whole-window shrink on HiDPI when mss/OpenCV starts capture (Tk sees wrong pixels)."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = ctypes.c_bool
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def get_base_dir():
    """Return app data directory: exe dir when packaged, else script dir."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


class App:
    def __init__(self):
        _configure_windows_dpi_before_tk()
        self.root = tk.Tk()
        self.base_dir = get_base_dir()
        self.root.title(f"屏幕录制 & GUI Agent 分析 - PC 端 v{__version__}")
        self.root.geometry("640x820")
        self.root.minsize(640, 820)
        self._initial_tk_scaling = 1.0

        recordings_dir = os.path.join(self.base_dir, "recordings")
        self.recorder = ScreenRecorder(fps=15, output_dir=recordings_dir)
        self.receiver_thread = None
        self.receiver_running = False
        self._analysis_running = False
        self._selected_video = None
        self._locked_geometry = None
        self._suppress_configure = False
        self._record_start_pending = False
        self._record_after_id = None

        self._build_ui()
        self.root.update_idletasks()
        try:
            self._initial_tk_scaling = float(self.root.tk.call("tk", "scaling"))
        except Exception:
            self._initial_tk_scaling = 1.0
        self.root.bind("<Configure>", self._on_root_configure)

    def _build_ui(self):
        frame_rec = ttk.LabelFrame(self.root, text="本机屏幕录制", padding=10)
        frame_rec.pack(fill=tk.X, padx=10, pady=6)

        row_rec = ttk.Frame(frame_rec)
        row_rec.pack(fill=tk.X)
        self.btn_record = ttk.Button(
            row_rec, text="开始录制", command=self._toggle_record
        )
        self.btn_record.pack(side=tk.LEFT, padx=(0, 8))
        self.lbl_status = ttk.Label(row_rec, text="状态: 未录制")
        self.lbl_status.pack(side=tk.LEFT)
        ttk.Button(row_rec, text="打开录制文件夹", command=self._open_recordings).pack(
            side=tk.LEFT, padx=(16, 0)
        )

        frame_rcv = ttk.LabelFrame(
            self.root, text="接收手机视频（同 WiFi 下手机可上传到此电脑）", padding=10
        )
        frame_rcv.pack(fill=tk.X, padx=10, pady=6)

        self.lbl_server = ttk.Label(frame_rcv, text="上传服务: 未启动")
        self.lbl_server.pack(anchor=tk.W)
        self.lbl_url = ttk.Label(frame_rcv, text="", foreground="gray")
        self.lbl_url.pack(anchor=tk.W)
        row_rcv_btn = ttk.Frame(frame_rcv)
        row_rcv_btn.pack(anchor=tk.W, pady=(4, 0))
        self.btn_start_receiver = ttk.Button(
            row_rcv_btn, text="启动上传接收服务", command=self._start_receiver
        )
        self.btn_start_receiver.pack(side=tk.LEFT)
        self.btn_stop_receiver = ttk.Button(
            row_rcv_btn, text="停止接收服务", command=self._stop_receiver, state=tk.DISABLED
        )
        self.btn_stop_receiver.pack(side=tk.LEFT, padx=(8, 0))

        # --- GUI Agent analysis section ---
        frame_analyze = ttk.LabelFrame(
            self.root, text="GUI Agent 视频分析（基于 GUI-Owl）", padding=10
        )
        frame_analyze.pack(fill=tk.X, padx=10, pady=6)

        row_mode = ttk.Frame(frame_analyze)
        row_mode.pack(fill=tk.X, pady=(0, 4))
        row_mode.columnconfigure(1, weight=1)
        ttk.Label(row_mode, text="分析后端:").grid(row=0, column=0, sticky=tk.W)
        self.backend_var = tk.StringVar(
            value="local" if GUI_AGENT_BACKEND != "api" else "api"
        )
        self.combo_backend = ttk.Combobox(
            row_mode,
            state="readonly",
            values=("local", "api"),
            textvariable=self.backend_var,
        )
        self.combo_backend.grid(row=0, column=1, sticky=tk.EW, padx=(4, 8))
        self.combo_backend.bind("<<ComboboxSelected>>", lambda _e: self._refresh_backend_ui())
        self.btn_deploy_local = ttk.Button(
            row_mode, text="一键部署本地模型", command=self._deploy_local_model
        )
        self.btn_deploy_local.grid(row=0, column=2, sticky=tk.E)

        row_model = ttk.Frame(frame_analyze)
        row_model.pack(fill=tk.X, pady=(0, 4))
        row_model.columnconfigure(1, weight=1)
        ttk.Label(row_model, text="模型ID:").grid(row=0, column=0, sticky=tk.W)
        self.entry_model_id = ttk.Entry(row_model)
        self.entry_model_id.grid(row=0, column=1, sticky=tk.EW, padx=(4, 0))
        self.entry_model_id.insert(0, LOCAL_GUI_OWL_MODEL_ID)

        self.row_api = ttk.Frame(frame_analyze)
        self.row_api.pack(fill=tk.X, pady=(0, 4))
        self.row_api.columnconfigure(1, weight=1)
        ttk.Label(self.row_api, text="API Key:").grid(row=0, column=0, sticky=tk.W)
        self.entry_api_key = ttk.Entry(self.row_api, show="*")
        self.entry_api_key.grid(row=0, column=1, sticky=tk.EW, padx=(4, 0))
        if DASHSCOPE_API_KEY:
            self.entry_api_key.insert(0, DASHSCOPE_API_KEY)

        row_model_hint = ttk.Frame(frame_analyze)
        row_model_hint.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            row_model_hint,
            text="local: 首次会自动下载模型；api: 使用云端 OpenAI 兼容接口",
            foreground="gray",
        ).pack(anchor=tk.W)

        row_file = ttk.Frame(frame_analyze)
        row_file.pack(fill=tk.X, pady=(0, 4))
        row_file.columnconfigure(2, weight=1)
        self.btn_select_video = ttk.Button(
            row_file, text="选择视频", command=self._select_video
        )
        self.btn_select_video.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(row_file, text="当前:").grid(row=0, column=1, sticky=tk.W, padx=(8, 4))
        self.lbl_video_path = ttk.Label(
            row_file, text="未选择", foreground="gray", wraplength=320
        )
        self.lbl_video_path.grid(row=0, column=2, sticky=tk.W)

        row_actions = ttk.Frame(frame_analyze)
        row_actions.pack(fill=tk.X, pady=(0, 4))
        self.btn_analyze = ttk.Button(
            row_actions, text="开始分析", command=self._start_analysis
        )
        self.btn_analyze.pack(side=tk.LEFT)
        self.btn_open_result = ttk.Button(
            row_actions, text="打开分析结果", command=self._open_analysis_output,
            state=tk.DISABLED,
        )
        self.btn_open_result.pack(side=tk.LEFT, padx=(8, 0))

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            frame_analyze, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 2))
        self.lbl_progress = ttk.Label(
            frame_analyze, text="", foreground="gray"
        )
        self.lbl_progress.pack(anchor=tk.W)

        # --- Memory section (Week 3) ---
        frame_mem = ttk.LabelFrame(
            self.root, text="记忆（识别结果存储、检索）", padding=10
        )
        frame_mem.pack(fill=tk.X, padx=10, pady=6)

        row_mem_stats = ttk.Frame(frame_mem)
        row_mem_stats.pack(fill=tk.X)
        self.lbl_mem_stats = ttk.Label(
            row_mem_stats, text="原始: 0 | 压缩: 0",
            foreground="gray",
        )
        self.lbl_mem_stats.pack(side=tk.LEFT)
        ttk.Button(row_mem_stats, text="刷新统计", command=self._refresh_mem_stats).pack(side=tk.LEFT, padx=(12, 0))

        row_search = ttk.Frame(frame_mem)
        row_search.pack(fill=tk.X, pady=(6, 0))
        row_search.columnconfigure(1, weight=1)
        ttk.Label(row_search, text="搜索:").grid(row=0, column=0, sticky=tk.NW, pady=(0, 4))
        self.entry_search = ttk.Entry(row_search)
        self.entry_search.grid(row=0, column=1, sticky=tk.EW, padx=(4, 8), pady=(0, 4))
        row_search_btns = ttk.Frame(row_search)
        row_search_btns.grid(row=0, column=2, sticky=tk.NW, pady=(0, 4))
        ttk.Button(row_search_btns, text="检索", command=self._search_memory).pack(side=tk.LEFT)
        ttk.Button(row_search_btns, text="查看用户画像", command=self._show_user_profile).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Label(
            frame_mem,
            text="（支持关键词或自然语言，如：微信、昨天浏览设置）",
            foreground="gray",
            font=("", 8),
        ).pack(anchor=tk.W)

        self.mem_results = scrolledtext.ScrolledText(frame_mem, height=4, state=tk.DISABLED, wrap=tk.WORD)
        self.mem_results.pack(fill=tk.X, pady=(4, 0))

        # --- Log area (no expand=True: avoids Tk shrinking the whole window on relayout) ---
        frame_log = ttk.LabelFrame(self.root, text="日志", padding=6)
        frame_log.pack(fill=tk.X, padx=10, pady=6)
        self.log_text = scrolledtext.ScrolledText(
            frame_log, height=10, state=tk.DISABLED, wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.X)

        self._log("启动完成。可录屏、接收手机视频、或选择已有视频进行 GUI Agent 分析。")
        self._refresh_mem_stats()
        self._refresh_backend_ui()

    # ---- logging ----

    def _log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    # ---- recording ----

    @staticmethod
    def _parse_geometry(geom: str):
        """Parse 'WxH+X+Y' or 'WxH' -> (w, h, x, y). x,y may be 0 if missing."""
        m = re.match(r"(\d+)x(\d+)(?:\+(-?\d+)\+(-?\d+))?", geom)
        if not m:
            return 640, 820, 0, 0
        w, h = int(m.group(1)), int(m.group(2))
        x = int(m.group(3)) if m.group(3) else 0
        y = int(m.group(4)) if m.group(4) else 0
        return w, h, x, y

    def _restore_window_size_dpi(self):
        """Restore Tk scaling + geometry after mss (no SetWindowPos: Win32 move breaks Tk layout / ghosting)."""
        g = self._locked_geometry
        if not g:
            return
        try:
            cur = float(self.root.tk.call("tk", "scaling"))
            if abs(cur - self._initial_tk_scaling) > 0.001:
                self.root.tk.call("tk", "scaling", self._initial_tk_scaling)
        except Exception:
            pass
        self._suppress_configure = True
        self.root.geometry(g)
        self.root.update_idletasks()
        self.root.after_idle(lambda: setattr(self, "_suppress_configure", False))

    def _on_root_configure(self, event):
        if event.widget is not self.root:
            return
        if self._suppress_configure or not self._locked_geometry:
            return
        lw, lh, lx, ly = self._parse_geometry(self._locked_geometry)
        cw, ch, cx, cy = self._parse_geometry(self.root.geometry())
        if cw >= lw and ch >= lh:
            return
        self._restore_window_size_dpi()

    def _toggle_record(self):
        if self.recorder.is_recording():
            self._locked_geometry = None
            self.root.minsize(640, 820)
            path = self.recorder.stop()
            self.btn_record.configure(text="开始录制")
            self.lbl_status.configure(text="状态: 未录制")
            self._log(f"录制已停止，已保存: {path}")
            return

        if self._record_start_pending and self._record_after_id is not None:
            self.root.after_cancel(self._record_after_id)
            self._record_after_id = None
            self._record_start_pending = False
            self._locked_geometry = None
            self.root.minsize(640, 820)
            self.btn_record.configure(text="开始录制")
            self.lbl_status.configure(text="状态: 未录制")
            self._log("已取消启动录制")
            return

        self.root.update_idletasks()
        self._locked_geometry = self.root.geometry()
        gw, gh, gx, gy = self._parse_geometry(self._locked_geometry)
        if gw < 200 or gh < 200:
            self._locked_geometry = f"640x820+{gx}+{gy}"
            self.root.geometry(self._locked_geometry)
            gw, gh = 640, 820
        self.root.minsize(gw, gh)

        self._record_start_pending = True
        self.btn_record.configure(text="停止录制")
        self.lbl_status.configure(text="状态: 准备录制…")
        self._record_after_id = self.root.after(120, self._deferred_start_recording)

    def _deferred_start_recording(self):
        self._record_after_id = None
        self._record_start_pending = False
        if self.recorder.is_recording():
            return
        ok, out = self.recorder.start()
        if ok:
            self.lbl_status.configure(text="状态: 录制中…")
            self._log(f"开始录制: {out}")
            self._restore_window_size_dpi()
            self.root.after(150, self._restore_window_size_dpi)
        else:
            self._locked_geometry = None
            self.root.minsize(640, 820)
            self.btn_record.configure(text="开始录制")
            self.lbl_status.configure(text="状态: 未录制")
            self._log(out)
            messagebox.showwarning("提示", out)

    def _open_recordings(self):
        folder = os.path.join(self.base_dir, "recordings")
        os.makedirs(folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            import subprocess
            subprocess.run(["xdg-open", folder], check=False)

    # ---- receiver ----

    def _start_receiver(self):
        if self.receiver_running:
            return
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"
        self.receiver_running = True
        self.receiver_thread = threading.Thread(
            target=run_receiver,
            kwargs={"host": "0.0.0.0", "port": RECEIVER_PORT},
            daemon=True,
        )
        self.receiver_thread.start()
        self.lbl_server.configure(text="上传服务: 已启动")
        self.lbl_url.configure(
            text=f"手机上传地址: http://{local_ip}:{RECEIVER_PORT}/upload"
        )
        self.btn_start_receiver.configure(state=tk.DISABLED)
        self.btn_stop_receiver.configure(state=tk.NORMAL)
        self._log(f"接收服务已启动: http://{local_ip}:{RECEIVER_PORT}/upload")
        self._log("请在手机 App 中填写此电脑的 IP 和端口进行上传。")

    def _stop_receiver(self):
        if not self.receiver_running:
            return
        self.receiver_running = False
        self.lbl_server.configure(text="上传服务: 已停止")
        self.lbl_url.configure(text="")
        self.btn_start_receiver.configure(state=tk.NORMAL)
        self.btn_stop_receiver.configure(state=tk.DISABLED)
        self._log("上传接收服务已停止。可再次点击「启动上传接收服务」重新开启。")

    # ---- GUI Agent analysis ----

    def _refresh_backend_ui(self):
        backend = self.backend_var.get().strip().lower()
        if backend == "api":
            self.entry_api_key.configure(state=tk.NORMAL)
            self.btn_deploy_local.configure(state=tk.DISABLED)
        else:
            self.entry_api_key.configure(state=tk.DISABLED)
            self.btn_deploy_local.configure(state=tk.NORMAL)

    def _deploy_local_model(self):
        if self._analysis_running:
            messagebox.showinfo("提示", "分析进行中，请稍后再部署。")
            return
        model_id = self.entry_model_id.get().strip() or LOCAL_GUI_OWL_MODEL_ID
        self.btn_deploy_local.configure(state=tk.DISABLED)
        self.lbl_progress.configure(text="准备部署本地模型...")
        self._log(f"开始部署本地模型: {model_id}")

        thread = threading.Thread(
            target=self._run_local_deploy,
            args=(model_id,),
            daemon=True,
        )
        thread.start()

    def _run_local_deploy(self, model_id: str):
        try:
            from gui_agent_api import deploy_local_model

            def progress(msg: str):
                self.root.after(0, self._update_progress, 0, msg)
                self.root.after(0, self._log, msg)

            info = deploy_local_model(
                base_dir=self.base_dir,
                model_id=model_id,
                progress_cb=progress,
            )
            ok_msg = f"本地模型部署完成: {info['model_id']}"
            self.root.after(0, self._deploy_done, ok_msg, None)
        except Exception as e:
            self.root.after(0, self._deploy_done, None, str(e))

    def _deploy_done(self, message: str, error: str):
        self.btn_deploy_local.configure(state=tk.NORMAL)
        if error:
            self.lbl_progress.configure(text=f"部署失败: {error}")
            self._log(f"本地模型部署失败: {error}")
            messagebox.showerror("部署失败", error)
            return
        self.lbl_progress.configure(text=message)
        self._log(message)
        messagebox.showinfo("完成", message)

    def _select_video(self):
        initial_dir = os.path.join(self.base_dir, "recordings")
        if not os.path.isdir(initial_dir):
            initial_dir = self.base_dir
        path = filedialog.askopenfilename(
            title="选择要分析的视频",
            initialdir=initial_dir,
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
        )
        if path:
            self._selected_video = path
            display = path if len(path) < 60 else "..." + path[-57:]
            self.lbl_video_path.configure(text=display, foreground="black")
            self._log(f"已选择视频: {path}")

    def _start_analysis(self):
        if self._analysis_running:
            messagebox.showinfo("提示", "分析正在进行中，请稍候。")
            return
        if not self._selected_video or not os.path.isfile(self._selected_video):
            messagebox.showwarning("提示", "请先选择一个有效的视频文件。")
            return
        backend = self.backend_var.get().strip().lower()
        api_key = self.entry_api_key.get().strip()
        model_id = self.entry_model_id.get().strip() or LOCAL_GUI_OWL_MODEL_ID
        if backend == "api" and not api_key:
            messagebox.showwarning(
                "提示",
                "当前为 api 后端，请填写 API Key。\n"
                "可从百炼控制台获取: https://bailian.console.aliyun.com/",
            )
            return

        self._analysis_running = True
        self.btn_analyze.configure(state=tk.DISABLED)
        self.btn_deploy_local.configure(state=tk.DISABLED)
        self.btn_open_result.configure(state=tk.DISABLED)
        self.progress_var.set(0)
        self.lbl_progress.configure(text="准备分析...")
        self._log(
            f"开始分析视频: {self._selected_video} | backend={backend} | model={model_id}"
        )

        thread = threading.Thread(
            target=self._run_analysis,
            args=(self._selected_video, api_key, backend, model_id),
            daemon=True,
        )
        thread.start()

    def _run_analysis(self, video_path: str, api_key: str, backend: str, model_id: str):
        from analyzer import analyze_video
        if backend == "local":
            try:
                from gui_agent_api import ensure_local_model_ready

                self.root.after(
                    0, self._update_progress, 0, "[2/2] 检查本地模型部署状态..."
                )
                ensure_local_model_ready(
                    base_dir=self.base_dir,
                    model_id=model_id,
                    progress_cb=lambda msg: self.root.after(
                        0, self._update_progress, 0, f"[2/2] {msg}"
                    ),
                    allow_download=False,
                )
            except Exception as e:
                self.root.after(0, self._analysis_done, None, str(e))
                return

        def on_progress(current, total, message):
            if total > 0:
                pct = min(100.0, current / total * 100)
            else:
                pct = 0
            self.root.after(0, self._update_progress, pct, message)

        output_base = os.path.join(self.base_dir, "analysis_output")
        try:
            result_dir = analyze_video(
                video_path=video_path,
                api_key=api_key,
                backend=backend,
                app_base_dir=self.base_dir,
                model=model_id,
                interval_sec=FRAME_SAMPLE_INTERVAL_SEC,
                ssim_threshold=SSIM_THRESHOLD,
                output_dir=output_base,
                progress_cb=on_progress,
            )
            self.root.after(0, self._analysis_done, result_dir, None)
        except Exception as e:
            self.root.after(0, self._analysis_done, None, str(e))

    def _update_progress(self, pct: float, message: str):
        self.progress_var.set(pct)
        self.lbl_progress.configure(text=message)

    def _analysis_done(self, result_dir, error):
        self._analysis_running = False
        self.btn_analyze.configure(state=tk.NORMAL)
        self._refresh_backend_ui()
        if error:
            self.progress_var.set(0)
            self.lbl_progress.configure(text=f"分析失败: {error}")
            self._log(f"分析失败: {error}")
            messagebox.showerror("分析失败", error)
        else:
            self._analysis_result_dir = result_dir
            self.progress_var.set(100)
            self.lbl_progress.configure(text="分析完成!")
            self.btn_open_result.configure(state=tk.NORMAL)
            self._log(f"分析完成，结果保存在: {result_dir}")
            self._store_into_memory(result_dir)

    def _store_into_memory(self, result_dir: str):
        """Store analysis report into memory, run clean/compress, build profile."""
        try:
            import json
            from memory import store_report_into_memory
            report_path = os.path.join(result_dir, "report.json")
            if not os.path.isfile(report_path):
                return
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            memory_dir = os.path.join(self.base_dir, "memory")
            os.makedirs(memory_dir, exist_ok=True)
            stats = store_report_into_memory(report, source_path=result_dir, memory_dir=memory_dir)
            self._log(f"已写入记忆: +{stats.get('added', 0)} 条, 压缩后 {stats.get('compressed', 0)} 条")
            self._refresh_mem_stats()
        except Exception as e:
            self._log(f"写入记忆失败: {e}")

    def _refresh_mem_stats(self):
        try:
            from memory import MemoryStore
            mem_dir = os.path.join(self.base_dir, "memory")
            os.makedirs(mem_dir, exist_ok=True)
            store = MemoryStore(os.path.join(mem_dir, "memory.db"))
            s = store.get_stats()
            self.lbl_mem_stats.configure(text=f"原始: {s['raw_records']} | 压缩: {s['compressed_records']}")
        except Exception:
            self.lbl_mem_stats.configure(text="原始: 0 | 压缩: 0")

    def _search_memory(self):
        query = self.entry_search.get().strip()
        if not query:
            self._show_mem_results("请输入关键词或自然语言查询")
            return
        try:
            from memory import MemoryStore
            mem_dir = os.path.join(self.base_dir, "memory")
            store = MemoryStore(os.path.join(mem_dir, "memory.db"))
            results = store.search_natural_language(query, limit=30)
            lines = []
            for r in results:
                lines.append(f"[{r.get('timestamp_sec', 0):.0f}s] {r.get('app_name', '')} / {r.get('page_name', '')}")
                lines.append(f"  {r.get('description', '')[:80]}...")
                lines.append("")
            self._show_mem_results("\n".join(lines) if lines else "无匹配结果")
        except Exception as e:
            self._show_mem_results(f"检索失败: {e}")

    def _show_mem_results(self, text: str):
        self.mem_results.configure(state=tk.NORMAL)
        self.mem_results.delete("1.0", tk.END)
        self.mem_results.insert(tk.END, text)
        self.mem_results.configure(state=tk.DISABLED)

    def _show_user_profile(self):
        try:
            from memory import MemoryStore
            mem_dir = os.path.join(self.base_dir, "memory")
            store = MemoryStore(os.path.join(mem_dir, "memory.db"))
            profile = store.get_user_profile()
            if not profile:
                profile = store.build_user_profile()
            lines = ["=== 用户画像 ===\n"]
            lines.append(f"总事件数: {profile.get('total_events', 0)}\n")
            lines.append("最常用应用 (Top 10):")
            for item in profile.get("top_apps", [])[:10]:
                lines.append(f"  - {item['app']}: {item['count']} 次")
            lines.append("\n各应用常用页面:")
            for app, pages in list(profile.get("app_pages", {}).items())[:8]:
                lines.append(f"  [{app}]")
                for page, cnt in list(pages.items())[:3]:
                    lines.append(f"    - {page}: {cnt} 次")
            msg = "\n".join(lines)
            self._show_mem_results(msg)
        except Exception as e:
            self._show_mem_results(f"加载画像失败: {e}")

    def _open_analysis_output(self):
        folder = getattr(self, "_analysis_result_dir", None)
        if not folder or not os.path.isdir(folder):
            folder = os.path.join(self.base_dir, "analysis_output")
        os.makedirs(folder, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)
        else:
            import subprocess
            subprocess.run(["xdg-open", folder], check=False)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
