"""
daytime_processor.py — 白天实时处理模块（短期记忆 / 闪存）
======================================================================
09-24 阶段 D。对应设计说明书的双阶段处理流程：

    短期记忆（闪存）：白天实时快速存储与提取，为 Agent 提供上下文，
    场景结束（一段连续活动告一段落）后即可删除。

三条职责：
1. 实时写入：视频片段 → Layer1 临时记忆 → 按场景分流
             Layer2（同环境场景）/ Layer3（当日，上限 1000）/ Layer7（永久主存储）
2. 实时提取：短期记忆优先检索，链路不调用 LLM，保证 ≤1s
3. 场景结束：清理 Layer1 / Layer2 短期记忆；Layer3 当日窗口与 Layer7 永久记忆保留
4. 后台翻译：非英文文本放入后台队列异步翻译为英文，回填 moment["normalized"]，
             不阻塞写入与检索（与 09-20 的「查询零 LLM」一致）

用法：
    from memory import PersonMemory
    from daytime_processor import DaytimeProcessor

    mem = PersonMemory(memory_dir="memory")
    day = DaytimeProcessor(mem, translate_fn=my_translate)

    day.start_scene("morning_kitchen")
    day.ingest(scene="我在厨房切菜", user_action="切菜", location="kitchen")
    hits = day.query("切菜")
    cleared = day.end_scene()          # 场景结束 → 清理短期记忆
    day.flush_translations()           # 等待后台翻译完成（可选）
"""

import queue
import threading
import time
from typing import Callable, Dict, List, Optional

from memory import _contains_cjk


class DaytimeProcessor:
    """白天实时处理：短期记忆（闪存）的快速存储与提取。"""

    def __init__(self, memory, translate_fn: Optional[Callable[[str], str]] = None):
        """
        memory      : PersonMemory 实例
        translate_fn: 非英文 → 英文的翻译函数；None 时不做后台翻译
                      （09-20 起检索走中英混合分词，翻译只是离线增强，非必需）
        """
        self.mem = memory
        self.translate_fn = translate_fn

        self.scene_id: Optional[str] = None
        self.scene_start: Optional[float] = None

        self._queue: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.stats = {
            "scenes": 0,
            "ingested": 0,
            "queries": 0,
            "short_term_cleared": 0,
            "translated": 0,
            "translate_failed": 0,
        }

    # ------------------------------------------------------------------
    # 场景生命周期
    # ------------------------------------------------------------------

    def start_scene(self, scene_id: str = None) -> str:
        """开始一个新场景（一段连续活动）。自动清理上一场景残留的短期记忆。"""
        if self.scene_id is not None:
            self.end_scene()
        self.scene_id = scene_id or f"scene_{int(time.time())}"
        self.scene_start = time.time()
        self.stats["scenes"] += 1
        return self.scene_id

    def end_scene(self) -> int:
        """
        场景结束：清理 Layer1 / Layer2 短期记忆（闪存），
        保留 Layer3（当日窗口）与 Layer7（永久主存储）。

        返回本次清理掉的引用条数。
        注意：Layer1/2 存的是引用，真实内容一直在 Layer7.moments，清理不会丢数据。
        """
        if self.scene_id is None:
            return 0
        with self._lock:
            cleared = len(self.mem.memory.get("layer1", [])) + \
                      len(self.mem.memory.get("layer2", []))
            self.mem.memory["layer1"] = []
            self.mem.memory["layer2"] = []
            self.mem._save()
        self.stats["short_term_cleared"] += cleared
        self.scene_id = None
        self.scene_start = None
        return cleared

    # ------------------------------------------------------------------
    # 实时写入
    # ------------------------------------------------------------------

    def ingest(self,
               scene: str,
               user_action: str,
               needs: List[Dict] = None,
               solutions: List[Dict] = None,
               people: List[str] = None,
               location: str = None,
               activity: str = None,
               environments: List[str] = None,
               objects: List[str] = None,
               extra_notes: str = "") -> str:
        """
        实时写入一条记忆。

        写入链路：Layer1（当前时刻）→ 按环境场景分流 Layer2 / Layer3 → Layer7 主存储 + 六维索引。
        非英文文本不在此处同步翻译，而是丢进后台队列（避免阻塞实时写入）。
        """
        # 与后台翻译线程互斥：两者的 _save() 会争用同一个临时文件
        with self._lock:
            moment_id = self.mem.add(
                scene=scene,
                user_action=user_action,
                needs=needs or [],
                solutions=solutions or [],
                people=people,
                location=location,
                activity=activity,
                environments=environments,
                objects=objects,
                extra_notes=extra_notes,
            )
        self.stats["ingested"] += 1

        # 后台离线翻译：只把含中文的字段入队
        if self.translate_fn:
            payload = {}
            for field, raw in (("scene", scene), ("user_action", user_action),
                               ("location", location), ("activity", activity),
                               ("notes", extra_notes)):
                if raw and _contains_cjk(str(raw)):
                    payload[field] = str(raw)
            if payload:
                self._queue.put((moment_id, payload))
                self._ensure_worker()

        return moment_id

    # ------------------------------------------------------------------
    # 实时提取
    # ------------------------------------------------------------------

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        """
        实时检索。走 memory.query()，链路不调用 LLM（09-20 中英混合分词），
        保证白天实时场景下的响应速度。
        """
        self.stats["queries"] += 1
        return self.mem.query(text, top_k=top_k)

    def context(self, top_k: int = 5) -> List[Dict]:
        """取当前场景上下文：Layer1（当前）→ Layer2（同场景）→ Layer3（当日）优先。"""
        l1 = list(self.mem.memory.get("layer1", []))
        l2 = list(self.mem.memory.get("layer2", []))
        l3 = list(self.mem.memory.get("layer3", []))
        return (l1 + l2 + l3)[:top_k]

    # ------------------------------------------------------------------
    # 后台离线翻译
    # ------------------------------------------------------------------

    def _ensure_worker(self):
        """确保后台翻译线程在运行。"""
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._translation_worker, daemon=True, name="translate-worker")
            self._worker.start()

    def _translation_worker(self):
        """后台翻译：把队列里的中文文本翻成英文，回填 moment["normalized"]。"""
        while True:
            item = self._queue.get()
            if item is None:                      # 退出信号
                self._queue.task_done()
                return
            moment_id, payload = item
            try:
                normalized = {}
                for field, raw in payload.items():
                    try:
                        translated = self.translate_fn(raw)
                        if translated:
                            normalized[field] = translated
                    except Exception:
                        self.stats["translate_failed"] += 1
                if normalized:
                    with self._lock:
                        moment = self.mem.memory["layer7"]["moments"].get(moment_id)
                        if moment is not None:
                            existing = moment.get("normalized") or {}
                            existing.update(normalized)
                            moment["normalized"] = existing
                            try:
                                self.mem._save()
                                self.stats["translated"] += 1
                            except Exception:
                                # 落盘失败不致命：内容已在内存里，下次 save 会带上
                                self.stats["translate_failed"] += 1
            finally:
                self._queue.task_done()

    def flush_translations(self, timeout: float = 10.0) -> bool:
        """等待后台翻译队列清空（测试或场景切换时调用）。超时返回 False。"""
        if self._worker is None:
            return True
        if not self._worker.is_alive() and not self._queue.empty():
            self._worker = None
            self._ensure_worker()          # 线程异常退出后重新拉起
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        return self._queue.empty()

    def stop(self):
        """停止后台翻译线程。"""
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(None)
            self._worker.join(timeout=3)
        self._worker = None

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict:
        """当前短期记忆状态（供可视化）。"""
        return {
            "scene_id": self.scene_id,
            "layer1": len(self.mem.memory.get("layer1", [])),
            "layer2": len(self.mem.memory.get("layer2", [])),
            "layer3": len(self.mem.memory.get("layer3", [])),
            "layer7_moments": len(self.mem.memory.get("layer7", {}).get("moments", {})),
            "pending_translations": self._queue.qsize(),
            "stats": dict(self.stats),
        }
