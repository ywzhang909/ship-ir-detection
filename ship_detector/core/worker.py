import threading
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QThread, Signal


class DetectionWorker(QThread):
    """后台推理线程 — 避免 UI 冻结"""
    finished = Signal(list)       # List[DetectedShip]
    error = Signal(str)
    progress = Signal(str)        # 状态文本

    def __init__(self, detector, frame, conf: float, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.frame = frame
        self.conf = conf

    def run(self):
        try:
            self.progress.emit("正在推理...")
            ships = self.detector.detect_frame(self.frame, conf=self.conf)
            self.finished.emit(ships)
        except Exception as e:
            self.error.emit(str(e))


class VideoDetectWorker(QThread):
    """视频连续检测线程 — 后台逐帧推理，供播放中实时识别使用。

    为什么用「最新帧槽位」而不是队列:
        单帧推理 (YOLO11l @1280) 约 0.3~1s，而播放是 30fps。任何队列都会以
        10~30 倍速积累、无限增长。既然无论如何都追不上播放速度，唯一正确的
        实时语义就是「永远只处理最新的那一帧，中间的丢掉」。

    为什么 submit() 是 UI 线程直接调用、而不是通过信号槽:
        本类重写了 run()，用手写 threading.Condition 循环代替 Qt 事件循环，
        因此该线程**没有 Qt 事件循环**。若把帧用 QueuedConnection 发给本对象，
        事件永远不会被派发。所以生产者必须直接调用 submit()（内部加锁保证
        跨线程安全）。反向的 result_ready 走正常队列连接回到 UI 线程 —— 
        UI 线程有事件循环，这是成立的。改动这里前请先读懂这段话。
    """

    result_ready = Signal(int, list)   # (generation, List[DetectedShip])
    error = Signal(str)

    def __init__(self, detector, conf: float = 0.25, parent=None):
        super().__init__(parent)
        self.detector = detector
        self._cond = threading.Condition()
        self._pending: Optional[Tuple[np.ndarray, int, float, int]] = None
        self._stop = False
        self._generation = 0
        self.conf = float(conf)

    @property
    def generation(self) -> int:
        return self._generation

    def advance_generation(self) -> int:
        """数据源切换时调用，返回新的代次；旧代次的结果会被上层丢弃。"""
        with self._cond:
            self._generation += 1
            gen = self._generation
            self._cond.notify_all()
        return gen

    def set_conf(self, conf: float) -> None:
        with self._cond:
            self.conf = float(conf)

    def submit(self, frame: np.ndarray, frame_id: int, conf: float,
               generation: int) -> None:
        """提交一帧（UI 线程直接调用）。覆盖未处理的旧帧。

        这里必须 copy: 调用方（如 cv2 解码）可能复用同一块内存，而推理要等到
        本线程下一次循环才发生。不 copy 的话检测器可能读到已被覆写的像素。
        """
        payload = (np.array(frame, copy=True), int(frame_id), float(conf),
                   int(generation))
        with self._cond:
            self._pending = payload
            self._cond.notify_all()

    def clear_pending(self) -> None:
        with self._cond:
            self._pending = None

    def run(self) -> None:
        while True:
            with self._cond:
                while (self._pending is None and not self._stop
                       and not self.isInterruptionRequested()):
                    self._cond.wait(timeout=0.1)
                if self._stop or self.isInterruptionRequested():
                    break
                payload = self._pending
                self._pending = None

            frame, _frame_id, conf, generation = payload
            try:
                ships = self.detector.detect_frame(frame, conf=conf)
            except Exception as exc:  # noqa: BLE001 — 检测失败不能弄死线程
                self.error.emit(str(exc))
                continue
            self.result_ready.emit(generation, ships)

    def shutdown(self, timeout_ms: int = 2000) -> None:
        """停止线程。必须唤醒 cond.wait()，否则线程会一直挂着。"""
        with self._cond:
            self._stop = True
            self._pending = None
            self._cond.notify_all()
        self.requestInterruption()
        if self.isRunning():
            self.wait(timeout_ms)
