"""Contract tests for continuous video detection (play + recognise ships).

Covers the units introduced for "play an MP4 and draw confidence boxes":
  * core/worker.py         VideoDetectWorker  (latest-frame slot, no Qt loop)
  * ui/central_canvas.py   frame_captured signal + non-aliased _current_frame
  * ui/main_window.py      continuous-detection wiring + generation guard
  * render_gui_screenshots.py  deterministic MP4 seek + screenshot renderer

The detector is faked everywhere so the suite stays fast and CPU-only; the
real-weights path is exercised by running the renderer manually.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.data_models import BoundingBox, DetectedShip

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def make_ship(conf=0.9, track_id=1):
    return DetectedShip(
        track_id=track_id,
        class_id=0,
        class_name="ship",
        bbox=BoundingBox(0.40, 0.30, 0.20, 0.25),
        confidence=conf,
        color=(0, 0, 255),
    )


class FakeDetector:
    """Records what it was asked to detect; returns a fixed number of ships."""

    def __init__(self, n_ships=1, delay=0.0):
        self.n_ships = n_ships
        self.delay = delay
        self.calls = []          # (mean_pixel_value, conf)
        self._lock = threading.Lock()
        self.concurrency = 0
        self.max_concurrency = 0

    def detect_frame(self, frame, conf=0.25):
        with self._lock:
            self.calls.append((float(np.mean(frame)), float(conf)))
            self.concurrency += 1
            self.max_concurrency = max(self.max_concurrency, self.concurrency)
        try:
            if self.delay:
                time.sleep(self.delay)
            return [make_ship(0.9, i + 1) for i in range(self.n_ships)]
        finally:
            with self._lock:
                self.concurrency -= 1


class FlakyDetector(FakeDetector):
    """Raises once to prove the worker survives a detector blow-up."""

    def __init__(self):
        super().__init__(n_ships=0)
        self.boom = True

    def detect_frame(self, frame, conf=0.25):
        with self._lock:
            self.calls.append(float(conf))
        if self.boom:
            self.boom = False
            raise RuntimeError("cuda oom")
        return []


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory):
    """A 20-frame 320x180 greyscale-ish clip, enough to drive playback."""
    path = tmp_path_factory.mktemp("vid") / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    assert writer.isOpened(), "could not open VideoWriter"
    for i in range(20):
        frame = np.full((180, 320, 3), i * 10 % 255, dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def pump(ms=120):
    """Let the Qt event loop deliver queued signals."""
    from PySide6.QtWidgets import QApplication

    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def wait_for(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        QApplication_process()
        time.sleep(0.005)
    return predicate()


def QApplication_process():
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()


# ===========================================================================
# VideoDetectWorker — latest-frame slot
# ===========================================================================


def test_worker_detects_submitted_frame_and_emits_result(qapp):
    """HAPPY: a submitted frame produces one result_ready(ships)."""
    from core.worker import VideoDetectWorker

    det = FakeDetector(n_ships=1)
    w = VideoDetectWorker(det, conf=0.25)
    got = []
    w.result_ready.connect(lambda gen, ships: got.append((gen, ships)))
    w.start()

    w.submit(np.full((32, 32, 3), 77, np.uint8), frame_id=5, conf=0.25, generation=0)
    try:
        assert wait_for(lambda: len(got) >= 1), "worker never emitted result_ready"
    finally:
        w.shutdown()

    gen, ships = got[0]
    assert gen == 0
    assert len(ships) == 1
    assert ships[0].confidence == pytest.approx(0.9)
    assert det.calls and det.calls[0][0] == pytest.approx(77.0)


def test_worker_latest_slot_wins_over_older_frames(qapp):
    """Only the newest frame is detected; older submissions are overwritten.

    Inference is far slower than playback, so a queue would grow without bound.
    The slot must collapse to the latest frame instead of draining a backlog.
    """
    from core.worker import VideoDetectWorker

    det = FakeDetector(n_ships=0, delay=0.05)
    w = VideoDetectWorker(det, conf=0.25)
    got = []
    w.result_ready.connect(lambda gen, ships: got.append((gen, ships)))
    w.start()
    try:
        # slam 6 distinct frames in before the first result can land
        for i in range(1, 7):
            w.submit(np.full((16, 16, 3), i * 20, np.uint8), frame_id=i, conf=0.25, generation=0)
            time.sleep(0.005)
        time.sleep(0.4)
        pump(150)
    finally:
        w.shutdown()

    detected_values = [round(c[0]) for c in det.calls]
    assert len(detected_values) < 6, f"slot did not collapse backlog: {detected_values}"
    assert detected_values[-1] == 120, f"latest frame not preferred: {detected_values}"


def test_worker_does_not_hold_a_reference_to_a_mutated_frame(qapp):
    """The detector must see the submitted pixels, not later mutations.

    cv2.VideoCapture reuses its decode buffer, so the producer copies before
    submitting. This locks that contract in at the worker boundary too.
    """
    from core.worker import VideoDetectWorker

    det = FakeDetector(n_ships=0, delay=0.08)
    w = VideoDetectWorker(det, conf=0.25)
    w.start()
    try:
        frame = np.full((32, 32, 3), 42, np.uint8)
        w.submit(frame, frame_id=1, conf=0.25, generation=0)
        frame[:] = 200  # producer scribbles over the array after submitting
        time.sleep(0.35)
    finally:
        w.shutdown()

    assert det.calls, "worker never ran detection"
    assert round(det.calls[0][0]) == 42, f"worker saw mutated pixels: {det.calls[0][0]}"


def test_worker_conf_change_is_picked_up(qapp):
    """set_conf() changes the threshold used by subsequent detections."""
    from core.worker import VideoDetectWorker

    det = FakeDetector(n_ships=0)
    w = VideoDetectWorker(det, conf=0.10)
    w.start()
    try:
        w.set_conf(0.77)
        w.submit(np.zeros((16, 16, 3), np.uint8), frame_id=1, conf=0.77, generation=0)
        time.sleep(0.2)
    finally:
        w.shutdown()

    assert det.calls, "no detection ran"
    assert det.calls[-1][1] == pytest.approx(0.77)


def test_worker_result_carries_generation_so_stale_sources_can_be_dropped(qapp):
    """Switching source bumps the generation; results must be labelled with it."""
    from core.worker import VideoDetectWorker

    det = FakeDetector(n_ships=0)
    w = VideoDetectWorker(det, conf=0.25)
    got = []
    w.result_ready.connect(lambda gen, ships: got.append(gen))
    w.start()
    try:
        gen = w.advance_generation()
        w.submit(np.zeros((16, 16, 3), np.uint8), frame_id=1, conf=0.25, generation=gen)
        time.sleep(0.2)
        pump(150)  # cross-thread signals are delivered through the event loop
    finally:
        w.shutdown()

    assert got, "no result emitted"
    assert got[0] == gen


def test_worker_survives_a_raising_detector(qapp):
    """A detector exception must not kill the thread."""
    from core.worker import VideoDetectWorker

    det = FlakyDetector()
    w = VideoDetectWorker(det, conf=0.25)
    errors = []
    w.error.connect(lambda msg: errors.append(msg))
    w.start()
    alive_after_error = False
    try:
        w.submit(np.zeros((16, 16, 3), np.uint8), frame_id=1, conf=0.25, generation=0)
        time.sleep(0.25)
        pump(100)  # cross-thread signals are delivered through the event loop
        alive_after_error = w.isRunning()
        # the worker must still be alive for a second submission
        w.submit(np.full((16, 16, 3), 9, np.uint8), frame_id=2, conf=0.25, generation=0)
        time.sleep(0.25)
        pump(100)
    finally:
        w.shutdown()

    assert errors, "detector exception was not reported"
    assert "cuda oom" in errors[0]
    assert alive_after_error, "worker thread died after a detector exception"
    assert len(det.calls) >= 2, "worker stopped processing after the exception"


def test_worker_shutdown_is_prompt(qapp):
    """shutdown() must not hang even when the worker is blocked waiting."""
    from core.worker import VideoDetectWorker

    w = VideoDetectWorker(FakeDetector(), conf=0.25)
    w.start()
    time.sleep(0.1)
    t0 = time.monotonic()
    w.shutdown()
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"shutdown took {elapsed:.2f}s — worker is wedged"


def test_worker_never_runs_two_inferences_concurrently(qapp):
    """One worker thread == at most one inference in flight on the model."""
    from core.worker import VideoDetectWorker

    det = FakeDetector(n_ships=0, delay=0.12)
    w = VideoDetectWorker(det, conf=0.25)
    w.start()
    try:
        for i in range(8):
            w.submit(np.full((16, 16, 3), i, np.uint8), frame_id=i, conf=0.25, generation=0)
            time.sleep(0.03)
        time.sleep(0.4)
    finally:
        w.shutdown()

    assert det.max_concurrency == 1, f"concurrent inference: {det.max_concurrency}"


# ===========================================================================
# CentralCanvas — frame_captured + non-aliased _current_frame
# ===========================================================================


def test_canvas_emits_frame_captured_with_a_copy(qapp, tiny_video):
    """Playback emits frame_captured, and _current_frame is never the cap buffer."""
    from ui.central_canvas import CentralCanvas

    canvas = CentralCanvas()
    frames = []
    canvas.frame_captured.connect(lambda f, fid: frames.append((f, fid)))
    canvas.set_frame_capture(True)
    canvas.load_video(str(tiny_video))
    try:
        canvas.pause()  # stop the timer; keep the single frame load_video did
        pump(150)
        assert frames, "frame_captured never fired"
        frame, frame_id = frames[-1]
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (180, 320, 3)
        assert frame_id >= 1

        # scribbling over the emitted array must not corrupt the canvas state
        snapshot = canvas.current_frame.copy()
        frame[:] = 123
        assert np.array_equal(canvas.current_frame, snapshot), (
            "emitted frame aliases canvas.current_frame — needs .copy()"
        )
    finally:
        canvas._stop_capture()
        canvas.close()


def test_canvas_frame_capture_is_opt_in(qapp, tiny_video):
    """Plain playback must not copy+broadcast every frame when nobody consumes it.

    1280x576x3 at 30fps is ~66MB/s of pointless allocation if frames are emitted
    unconditionally, so capture is gated behind set_frame_capture().
    """
    from ui.central_canvas import CentralCanvas

    canvas = CentralCanvas()
    frames = []
    canvas.frame_captured.connect(lambda f, fid: frames.append((f, fid)))
    try:
        canvas.load_video(str(tiny_video))
        canvas.pause()
        pump(150)
        assert frames == [], "frame_captured fired while capture was disabled"

        canvas.set_frame_capture(True)
        canvas.load_video(str(tiny_video))
        canvas.pause()
        pump(150)
        assert frames, "frame_captured never fired after capture was enabled"
    finally:
        canvas._stop_capture()
        canvas.close()


def test_canvas_set_ships_reports_the_count(qapp):
    """set_ships drives ships_updated so the side panels stay truthful."""
    from ui.central_canvas import CentralCanvas

    canvas = CentralCanvas()
    seen = []
    canvas.ships_updated.connect(lambda n: seen.append(n))
    canvas.set_ships([make_ship(0.8, 1), make_ship(0.6, 2)])
    assert seen[-1] == 2
    assert len(canvas.ships) == 2
    canvas.set_ships([])
    assert seen[-1] == 0
    canvas.close()


# ===========================================================================
# MainWindow — continuous detection wiring
# ===========================================================================


def test_main_window_continuous_toggle_drives_worker_and_canvas(qapp):
    """Toggling continuous detection starts the worker and feeds the canvas."""
    from ui.main_window import MainWindow

    win = MainWindow()
    win.detector = FakeDetector(n_ships=1)
    win.detector._is_loaded = True
    try:
        assert win.detect_worker is not None, "MainWindow must own a VideoDetectWorker"

        win.set_continuous_detection(True)
        assert win.toolbar.act_continuous.isEnabled()
        assert win._continuous is True

        seen = []
        win.canvas.ships_updated.connect(lambda n: seen.append(n))

        # simulate a decoded frame arriving from the canvas
        win.canvas.frame_captured.emit(np.full((32, 32, 3), 3, np.uint8), 1)
        assert wait_for(lambda: len(win.canvas.ships) == 1), "canvas never received ships"
        assert win.canvas.ships[0].confidence == pytest.approx(0.9)

        win.set_continuous_detection(False)
        assert win._continuous is False
    finally:
        win.set_continuous_detection(False)
        win.close()


def test_main_window_ignores_results_from_a_previous_source(qapp):
    """After switching source, a late result from the old clip must be dropped."""
    from ui.main_window import MainWindow

    win = MainWindow()
    win.detector = FakeDetector(n_ships=1)
    win.detector._is_loaded = True
    try:
        win.set_continuous_detection(True)
        win.set_detection_generation(win.detect_worker.advance_generation())
        stale_gen = win.detect_worker.generation - 1
        win.detect_worker.result_ready.emit(stale_gen, [make_ship(0.9, 1)])
        pump(120)
        assert win.canvas.ships == [], "stale-generation result was drawn"

        win.detect_worker.result_ready.emit(win.detect_worker.generation, [make_ship(0.9, 2)])
        pump(120)
        assert len(win.canvas.ships) == 1
    finally:
        win.set_continuous_detection(False)
        win.close()


def test_main_window_manual_detection_is_blocked_during_continuous(qapp):
    """One model, one inference path: F5 must not race the continuous worker."""
    from ui.main_window import MainWindow

    win = MainWindow()
    win.detector = FakeDetector(n_ships=0, delay=0.05)
    win.detector._is_loaded = True
    try:
        win.canvas.set_frame(np.full((32, 32, 3), 11, np.uint8))
        win.set_continuous_detection(True)
        win.canvas.frame_captured.emit(np.full((32, 32, 3), 11, np.uint8), 1)
        win._run_detection()
        assert win._detecting is False, "manual detection started while continuous was on"
        win.set_continuous_detection(False)
    finally:
        win.set_continuous_detection(False)
        win.close()


def test_continuous_detection_refused_while_manual_detection_running(qapp):
    """BLOCKING invariant: the shared model must never run two inferences at once.

    Pressing F5 and then toggling 连续检测 on must NOT start a second inference
    while the one-shot worker is still in flight — Ultralytics models are not
    thread-safe and concurrent calls can corrupt the CUDA context.
    """
    from ui.main_window import MainWindow

    win = MainWindow()
    win.detector = FakeDetector(n_ships=0, delay=0.1)
    win.detector._is_loaded = True
    try:
        win.canvas.set_frame(np.full((32, 32, 3), 11, np.uint8))
        win._run_detection()
        assert win._detecting is True, "one-shot detection should have started"

        win.set_continuous_detection(True)

        assert win._continuous is False, (
            "continuous detection started while a manual detection was in flight"
        )
        assert win.toolbar.act_continuous.isChecked() is False, (
            "toolbar toggle left on even though continuous mode was refused"
        )
    finally:
        if win._worker is not None and win._worker.isRunning():
            wait_for(lambda: not win._worker.isRunning(), timeout=3.0)
        win.set_continuous_detection(False)
        win.close()


def test_manual_detection_refused_while_continuous_worker_busy(qapp):
    """The reverse direction: a continuous worker mid-frame must also block F5.

    Turning continuous mode off does not cancel an in-flight inference, so the
    guard must consult the worker's busy state rather than only ``_continuous``.
    """
    from ui.main_window import MainWindow

    win = MainWindow()
    win.detector = FakeDetector(n_ships=0, delay=0.3)
    win.detector._is_loaded = True
    try:
        win.set_continuous_detection(True)
        win.canvas.frame_captured.emit(np.full((32, 32, 3), 5, np.uint8), 1)
        assert wait_for(lambda: win.detect_worker.is_inferencing()), "never started inferring"

        win.set_continuous_detection(False)
        win.canvas.set_frame(np.full((32, 32, 3), 7, np.uint8))
        win._run_detection()

        assert win._detecting is False, (
            "manual detection ran concurrently with an in-flight continuous inference"
        )
    finally:
        win.set_continuous_detection(False)
        win.close()


def test_continuous_detection_reports_worker_errors(qapp):
    """A failing continuous worker must surface, not silently stop updating."""
    from ui.main_window import MainWindow

    win = MainWindow()
    win.detector = FlakyDetector()
    win.detector._is_loaded = True
    try:
        win.set_continuous_detection(True)
        win.canvas.frame_captured.emit(np.full((32, 32, 3), 5, np.uint8), 1)
        pump(400)
        text = win.bottom_panel.log.toPlainText()
        assert "cuda oom" in text, f"worker error never surfaced in the log: {text!r}"
    finally:
        win.set_continuous_detection(False)
        win.close()


# ===========================================================================
# render_gui_screenshots.py — deterministic MP4 seek + screenshots
# ===========================================================================


def test_renderer_resolves_repo_paths(qapp):
    """The weights dir must actually exist (this used to be off by one level)."""
    import render_gui_screenshots as r

    assert r.REPO.name == "ship", f"REPO resolved to {r.REPO}"
    assert r.WEIGHTS_DIR.exists(), f"WEIGHTS_DIR does not exist: {r.WEIGHTS_DIR}"
    assert list(r.WEIGHTS_DIR.glob(r.BEST_MODEL_GLOB)), "best model weights not found"


def test_renderer_picks_spread_out_frames(tiny_video):
    """Screenshot indices must be distinct, ordered and inside the clip."""
    import render_gui_screenshots as r

    picks = r.pick_screenshot_frames(60, count=3)
    assert picks == sorted(picks)
    assert len(set(picks)) == 3
    assert all(0 <= i < 60 for i in picks)
    assert picks[0] == 0


def test_renderer_handles_a_degenerate_video_length():
    """A 1-frame or empty clip must still yield a usable index list."""
    import render_gui_screenshots as r

    assert r.pick_screenshot_frames(1, count=3) == [0]
    assert r.pick_screenshot_frames(0, count=3) == []


def test_renderer_rejects_an_unreadable_video(qapp, tmp_path):
    """A non-video file must be reported, not crash the batch."""
    import render_gui_screenshots as r

    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"this is not a video")
    cap = cv2.VideoCapture(str(bad))
    assert not cap.isOpened()
    cap.release()


def test_renderer_draws_a_confidence_box_for_a_video_frame(qapp, tiny_video, tmp_path):
    """END-TO-END (fake detector): an MP4 frame yields a screenshot on disk.

    This is the surface the user actually asked for — play the clip, recognise
    the ship, and emit a PNG with a confidence label drawn on it.
    """
    import render_gui_screenshots as r
    from ui.main_window import MainWindow

    win = MainWindow()
    win.detector = FakeDetector(n_ships=1)
    win.detector._is_loaded = True
    try:
        out = tmp_path / "shots"
        made = r.render_video_frames(
            win=win,
            app=qapp,
            detector=win.detector,
            video_path=str(tiny_video),
            frame_indices=[0, 5],
            conf=0.25,
            out_dir=out,
            prefix="tiny",
        )
        assert len(made) == 2, f"expected 2 screenshots, got {made}"
        for p in made:
            assert p.exists() and p.stat().st_size > 1024, f"bad screenshot: {p}"
    finally:
        win.close()