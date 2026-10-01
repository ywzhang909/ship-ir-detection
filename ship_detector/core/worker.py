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
        self._busy = False
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

    def submit(self, frame: np.ndarray, frame_id: int, conf: Optional[float] = None,
               generation: int = 0) -> None:
        """提交一帧（UI 线程直接调用）。覆盖未处理的旧帧。

        这里必须 copy: 调用方（如 cv2 解码）可能复用同一块内存，而推理要等到
        本线程下一次循环才发生。不 copy 的话检测器可能读到已被覆写的像素。
        """
        if conf is None:
            with self._cond:
                conf = self.conf
        payload = (np.array(frame, copy=True), int(frame_id), float(conf),
                   int(generation))
        with self._cond:
            self._pending = payload
            self._cond.notify_all()

    def is_busy(self) -> bool:
        """是否还有未处理帧或正在推理。

        调用方用它保证「同一时刻只有一个推理」——共享的 Ultralytics 模型不是
        线程安全的，并发调用可能直接崩掉 CUDA 上下文。
        """
        with self._cond:
            return self._pending is not None or self._busy

    def is_inferencing(self) -> bool:
        """是否正处在一次推理调用中间（不含仅排队、尚未开始的帧）。"""
        with self._cond:
            return self._busy

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
                self._busy = True

            try:
                frame, _frame_id, conf, generation = payload
                try:
                    ships = self.detector.detect_frame(frame, conf=conf)
                except Exception as exc:  # noqa: BLE001 — 检测失败不能弄死线程
                    self.error.emit(str(exc))
                    continue
                self.result_ready.emit(generation, ships)
            finally:
                with self._cond:
                    self._busy = False

    def shutdown(self, timeout_ms: int = 5000) -> None:
        """停止线程。必须唤醒 cond.wait()，否则线程会一直挂着。

        超时预算给得较宽（默认 5s）: 首帧推理含 CUDA 预热可能较慢，若此处提前
        返回，窗口销毁后线程仍在跑，Qt 会报 "QThread: Destroyed while thread is
        still running" 并可能崩溃。超时未退出时至少留下明确日志。
        """
        with self._cond:
            self._stop = True
            self._pending = None
            self._cond.notify_all()
        self.requestInterruption()
        if self.isRunning():
            if not self.wait(timeout_ms):
                logger.warning(
                    "VideoDetectWorker 在 %d ms 内未退出（推理可能仍在进行）", timeout_ms
                )
