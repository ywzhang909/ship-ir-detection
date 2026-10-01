"""Contract tests for annotated-video export + the detection results report.

Covers `export_results.py`, which burns confidence boxes into every frame of a
clip and emits per-frame / aggregate statistics.

The detector is faked so the suite stays fast and CPU-only; the real-weights
path is exercised by running the exporter manually.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from core.data_models import BoundingBox, DetectedShip

cv2 = pytest.importorskip("cv2")


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
    """Returns a fixed ship count; optionally scaling confidence with brightness."""

    def __init__(self, n_ships=1):
        self.n_ships = n_ships
        self.calls = []
        self._lock = threading.Lock()

    def detect_frame(self, frame, conf=0.25):
        with self._lock:
            self.calls.append(conf)
        # vary the confidence so aggregate stats are not degenerate
        base = float(np.mean(frame)) / 255.0
        return [make_ship(min(0.99, 0.5 + base / 2.0), i + 1) for i in range(self.n_ships)]


@pytest.fixture(scope="session")
def clip(tmp_path_factory):
    """20-frame 320x180 clip with a per-frame brightness ramp."""
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    assert writer.isOpened()
    for i in range(20):
        frame = np.full((180, 320, 3), i * 10, dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


# ===========================================================================
# annotated video
# ===========================================================================


def test_annotated_video_is_written_with_every_source_frame(clip, tmp_path):
    """HAPPY: output video exists and holds one frame per processed source frame."""
    import export_results as ex

    out = tmp_path / "annotated.mp4"
    stats = ex.export_annotated_video(str(clip), FakeDetector(1), str(out), conf=0.25)

    assert out.exists(), "annotated video was not written"
    assert out.stat().st_size > 1024
    assert stats["frames_processed"] == 20
    assert stats["source_frames"] == 20
    assert count_frames(out) == 20, "annotated video lost frames"


def test_annotated_frames_visibly_differ_from_the_source(clip, tmp_path):
    """The box must actually be burned in, not silently skipped.

    This is the surface the user asked for: recognisable ships, drawn with
    confidence. Reading the output back and diffing against the original proves it.
    """
    import export_results as ex

    out = tmp_path / "annotated.mp4"
    ex.export_annotated_video(str(clip), FakeDetector(1), str(out), conf=0.25)

    src = cv2.VideoCapture(str(clip))
    dst = cv2.VideoCapture(str(out))
    ok_s, first_src = src.read()
    ok_d, first_dst = dst.read()
    src.release()
    dst.release()

    assert ok_s and ok_d, "could not read the first frame back"
    assert first_src.shape == first_dst.shape, "annotation changed the frame geometry"
    diff = cv2.absdiff(first_src, first_dst)
    assert diff.max() > 0, "annotated frame is pixel-identical to the source — nothing drawn"


def test_stride_controls_sampling(clip, tmp_path):
    """stride=2 must sample every other frame, and say so in the stats."""
    import export_results as ex

    out = tmp_path / "strided.mp4"
    stats = ex.export_annotated_video(str(clip), FakeDetector(1), str(out), conf=0.25, stride=2)

    assert stats["stride"] == 2
    assert stats["frames_processed"] == 10, stats["frames_processed"]
    assert count_frames(out) == 10
    indices = [f["frame_index"] for f in stats["per_frame"]]
    assert indices == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18], indices


def test_zero_detection_frames_still_produce_a_video_and_report(clip, tmp_path):
    """EDGE: a clip with no ships must still export cleanly, not raise."""
    import export_results as ex

    out = tmp_path / "empty.mp4"
    stats = ex.export_annotated_video(str(clip), FakeDetector(0), str(out), conf=0.25)

    assert out.exists() and out.stat().st_size > 1024
    assert stats["frames_processed"] == 20
    assert stats["total_detections"] == 0
    assert stats["frames_with_detections"] == 0
    assert all(f["count"] == 0 for f in stats["per_frame"])


def test_missing_source_video_is_reported_not_raised(tmp_path):
    """EDGE: a bad path yields a stats dict carrying an error, never an exception."""
    import export_results as ex

    out = tmp_path / "never.mp4"
    stats = ex.export_annotated_video(str(tmp_path / "nope.mp4"), FakeDetector(1), str(out))

    assert stats.get("error"), "missing video was not reported"
    assert stats["per_frame"] == []
    assert not out.exists(), "an output video was written despite a missing source"


def test_stride_zero_or_negative_falls_back_to_one(clip, tmp_path):
    """EDGE: nonsense stride must not produce an empty/invalid video."""
    import export_results as ex

    out = tmp_path / "s.mp4"
    stats = ex.export_annotated_video(str(clip), FakeDetector(1), str(out), stride=0)
    assert stats["stride"] == 1
    assert stats["frames_processed"] == 20


# ===========================================================================
# report
# ===========================================================================


def test_per_frame_stats_carry_timestamps_confidences_and_boxes(clip):
    """Every sampled frame gets an auditable record."""
    import export_results as ex

    stats = ex.export_annotated_video(str(clip), FakeDetector(1), None, conf=0.25)

    assert len(stats["per_frame"]) == 20
    rec = stats["per_frame"][0]
    for key in ("frame_index", "timestamp", "count", "detections"):
        assert key in rec, f"missing per-frame key: {key}"
    assert rec["timestamp"] == pytest.approx(0.0)
    assert stats["per_frame"][3]["timestamp"] == pytest.approx(3 / 30.0, abs=1e-3)
    det = rec["detections"][0]
    for key in ("track_id", "class_name", "confidence", "bbox"):
        assert key in det, f"missing detection key: {key}"
    assert 0.0 <= det["confidence"] <= 1.0
    assert set(det["bbox"]) == {"x", "y", "w", "h"}


def test_summary_aggregates_are_consistent(clip):
    """Summary numbers must actually follow from the per-frame records."""
    import export_results as ex

    stats = ex.export_annotated_video(str(clip), FakeDetector(2), None, conf=0.25)

    summary = stats["summary"]
    assert summary["frames_processed"] == 20
    assert summary["frames_with_detections"] == 20
    assert summary["detection_rate"] == pytest.approx(1.0)
    assert summary["total_detections"] == 40
    assert summary["max_detections_in_a_frame"] == 2
    assert summary["mean_confidence"] == pytest.approx(
        np.mean([d["confidence"] for f in stats["per_frame"] for d in f["detections"]])
    )
    assert 0.0 <= summary["min_confidence"] <= summary["max_confidence"] <= 1.0


def test_report_records_the_run_configuration(clip):
    """Provenance: which model/threshold produced these numbers must be recorded."""
    import export_results as ex

    det = FakeDetector(1)
    stats = ex.export_annotated_video(
        str(clip), det, None, conf=0.30, stride=2, use_sahi=True, model_name="T5_yolo11l_fusion"
    )
    meta = stats["run"]
    assert meta["confidence"] == pytest.approx(0.30)
    assert meta["stride"] == 2
    assert meta["use_sahi"] is True
    assert meta["model_name"] == "T5_yolo11l_fusion"
    assert meta["source"].endswith("clip.mp4")
    assert meta["fps"] == pytest.approx(30.0)
    assert meta["frame_size"] == [320, 180]
    assert meta["source_frames"] == 20
    assert stats["source_frames"] == meta["source_frames"], "run/top-level disagree"


def test_conf_threshold_is_forwarded_to_the_detector(clip):
    """The threshold the user set must be the threshold actually used."""
    import export_results as ex

    det = FakeDetector(1)
    ex.export_annotated_video(str(clip), det, None, conf=0.42)
    assert det.calls, "detector was never called"
    assert all(c == pytest.approx(0.42) for c in det.calls)


def test_use_sahi_toggles_detector_sahi(clip):
    """SAHI must actually be switched on/off on the detector, not just recorded."""
    import export_results as ex

    class SahiSpy(FakeDetector):
        def __init__(self):
            super().__init__(1)
            self.sahi = []

        def set_sahi(self, enabled):
            self.sahi.append(enabled)

    spy = SahiSpy()
    ex.export_annotated_video(str(clip), spy, None, use_sahi=True)
    assert spy.sahi == [True]

    spy2 = SahiSpy()
    ex.export_annotated_video(str(clip), spy2, None, use_sahi=False)
    assert spy2.sahi == [False]


# ===========================================================================
# report writing
# ===========================================================================


def test_write_report_emits_both_json_and_markdown(clip, tmp_path):
    """D2: a machine-readable JSON and a human-readable Markdown side by side."""
    import export_results as ex

    stats = ex.export_annotated_video(str(clip), FakeDetector(1), None, conf=0.25)
    paths = ex.write_report(stats, tmp_path, stem="report")
    kinds = {p.suffix for p in paths}
    assert kinds == {".json", ".md"}, kinds

    json_path = next(p for p in paths if p.suffix == ".json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["frames_processed"] == 20
    assert len(payload["per_frame"]) == 20

    md_path = next(p for p in paths if p.suffix == ".md")
    text = md_path.read_text(encoding="utf-8")
    assert "frames_processed" in text or "帧" in text
    assert "20" in text


def test_write_report_rejects_a_stats_dict_without_summary(tmp_path):
    """EDGE: refuse to emit a misleading report instead of writing junk."""
    import export_results as ex

    with pytest.raises(ValueError):
        ex.write_report({"per_frame": []}, tmp_path)


def test_module_imports_without_a_display():
    """The module must be importable for pure-logic use without Qt/cv2 loaded eagerly."""
    import importlib

    mod = importlib.import_module("export_results")
    assert hasattr(mod, "export_annotated_video")
    assert hasattr(mod, "write_report")


def test_cli_fails_fast_on_an_unreadable_video(tmp_path):
    """A present-but-corrupt video must yield a non-zero exit code.

    Also guards the ordering: input validation must happen BEFORE the (slow) GPU
    model load, so bad input fails immediately instead of after a wasted warmup.
    """
    import time

    import export_results as ex

    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"definitely not a video")
    out = tmp_path / "out"

    start = time.monotonic()
    rc = ex.main(["--video", str(bad), "--out-dir", str(out)])
    elapsed = time.monotonic() - start

    assert rc == 1, "CLI reported success for an unreadable video"
    assert not list(out.glob("annotated_*.mp4")), "wrote an annotated video for a bad source"
    assert elapsed < 15.0, f"took {elapsed:.1f}s — model was loaded before validating input"


def test_cli_fails_fast_on_a_missing_video(tmp_path):
    import export_results as ex

    out = tmp_path / "out"
    rc = ex.main(["--video", str(tmp_path / "nope.mp4"), "--out-dir", str(out)])
    assert rc == 1
    assert not out.exists(), "output dir created despite invalid input"