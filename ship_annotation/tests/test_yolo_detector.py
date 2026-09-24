"""YoloDetector 单元测试与真实推理冒烟测试。

单元测试不加载权重（快速）；冒烟测试加载真实最优模型并对
NSLSR 测试集图片执行完整推理链路（灰度归一化 → 训练分布
预处理 → YOLO 推理 → TargetObject 转换）。
"""
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.yolo_detector import YoloDetector  # noqa: E402
from core.data_models import BoundingBox, TargetObject  # noqa: E402
import detector as _core  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_sample_image() -> Path | None:
    """优先取训练域内 NSLSR 测试图，退回 customn 实拍图。"""
    for pattern in (
        "datasets/NSLSR/data/NSLSR_test/**/*.jpg",
        "datasets/NSLSR/data/NSLSR_test/**/*.png",
        "datasets/customn/**/*.bmp",
        "datasets/customn/**/*.jpg",
    ):
        found = sorted(REPO_ROOT.glob(pattern))
        if found:
            return found[0]
    return None


class TestModelDiscovery:
    """模型发现逻辑（不加载权重）"""

    def test_discover_returns_sorted_models(self):
        models = _core.discover_models()
        assert len(models) > 0, "runs/detect/ship-detection 下应存在 best.pt"
        known = [m.map50_95 for m in models if m.map50_95 is not None]
        assert known == sorted(known, reverse=True), "应按 mAP50-95 降序排列"

    def test_best_model_is_t5_fusion(self):
        models = YoloDetector.available_models()
        if not models:
            pytest.skip("本地无模型权重")
        assert models[0].name == "T5_yolo11l_fusion"
        assert models[0].prep == "dual"

    def test_model_info_label_format(self):
        models = _core.discover_models()
        if not models:
            pytest.skip("本地无模型权重")
        label = models[0].label
        assert "T5_yolo11l_fusion" in label
        assert "[dual]" in label


class TestUnloadedBehavior:
    """未加载模型时的安全行为"""

    def test_detect_before_load_returns_empty(self):
        det = YoloDetector()
        assert det.is_loaded is False
        assert det.detect("whatever.jpg") == []

    def test_model_label_unloaded(self):
        det = YoloDetector()
        assert det.model_label == "未加载"

    def test_prep_unloaded_is_none(self):
        det = YoloDetector()
        assert det.prep is None


class TestTargetConversion:
    """Detection → TargetObject 归一化坐标转换"""

    def test_bbox_normalization_math(self):
        bbox = BoundingBox(0.25, 0.5, 0.125, 0.25)
        x1, y1, x2, y2 = bbox.to_pixels(640, 512)
        assert (x1, y1, x2, y2) == (160, 256, 240, 384)

    def test_target_object_fields(self):
        t = TargetObject(
            id=1,
            class_id=0,
            class_name="Ada",
            bbox=BoundingBox(0.1, 0.1, 0.2, 0.2),
            confidence=0.87,
            color=(255, 0, 0),
        )
        assert t.class_name == "Ada"
        assert 0.0 <= t.confidence <= 1.0
        assert t.is_manual is False


@pytest.mark.slow
class TestRealInference:
    """真实推理冒烟测试：自动加载最优模型并生成检测结果"""

    def test_load_best_model(self):
        det = YoloDetector()
        if not YoloDetector.available_models():
            pytest.skip("本地无模型权重")
        assert det.load_best_model() is True
        assert det.is_loaded is True
        assert "T5_yolo11l_fusion" in det.model_label
        assert det.prep == "dual"

    def test_detect_generates_results(self):
        sample = _find_sample_image()
        if sample is None:
            pytest.skip("无可用样例图片")
        det = YoloDetector()
        if not YoloDetector.available_models():
            pytest.skip("本地无模型权重")
        assert det.load_best_model() is True

        targets = det.detect(str(sample), conf=0.25)

        assert isinstance(targets, list)
        assert det.last_latency_ms > 0
        class_names = det.get_class_names()
        for t in targets:
            assert isinstance(t, TargetObject)
            assert 0.0 <= t.bbox.x <= 1.0 and 0.0 <= t.bbox.w <= 1.0
            assert 0.0 <= t.confidence <= 1.0
            # 类别名必须来自 checkpoint 内置 names（T5 为单类 "ship"）
            assert t.class_name == class_names[t.class_id]
            assert t.class_name

    def test_class_names_from_checkpoint(self):
        """加载后类别名应与 checkpoint 一致（T5: nc=1, {0: ship}）"""
        det = YoloDetector()
        if not YoloDetector.available_models():
            pytest.skip("本地无模型权重")
        assert det.load_best_model() is True
        assert det.get_class_names() == ["ship"]
