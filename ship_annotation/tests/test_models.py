"""数据模型单元测试"""
import pytest
from core.data_models import (
    BoundingBox, TargetObject, FrameData, ProjectConfig,
    SourceType, AnnotationTool,
)


class TestBoundingBox:
    def test_creation(self):
        bbox = BoundingBox(0.1, 0.2, 0.3, 0.4)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.w == 0.3
        assert bbox.h == 0.4

    def test_to_pixels(self):
        bbox = BoundingBox(0.1, 0.2, 0.3, 0.4)
        x1, y1, x2, y2 = bbox.to_pixels(1000, 500)
        assert x1 == 100
        assert y1 == 100
        assert x2 == 400
        assert y2 == 300

    def test_to_pixels_full_image(self):
        bbox = BoundingBox(0.0, 0.0, 1.0, 1.0)
        x1, y1, x2, y2 = bbox.to_pixels(800, 600)
        assert x1 == 0
        assert y1 == 0
        assert x2 == 800
        assert y2 == 600


class TestTargetObject:
    def test_creation(self):
        bbox = BoundingBox(0.1, 0.2, 0.3, 0.4)
        target = TargetObject(
            id=1, class_id=0, class_name="船只",
            bbox=bbox, confidence=0.95,
        )
        assert target.id == 1
        assert target.class_name == "船只"
        assert target.confidence == 0.95
        assert target.is_manual is False

    def test_manual_annotation(self):
        bbox = BoundingBox(0.5, 0.5, 0.1, 0.1)
        target = TargetObject(
            id=1, class_id=0, class_name="船只",
            bbox=bbox, confidence=1.0, is_manual=True,
        )
        assert target.is_manual is True


class TestSourceType:
    def test_enum_values(self):
        assert SourceType.IMAGE is not None
        assert SourceType.CAMERA is not None
        assert SourceType.IMAGE != SourceType.CAMERA


class TestAnnotationTool:
    def test_enum_values(self):
        assert AnnotationTool.SELECT is not None
        assert AnnotationTool.RECTANGLE is not None
        assert AnnotationTool.ARROW is not None
        assert AnnotationTool.DELETE is not None


class TestProjectConfig:
    def test_default_config(self):
        config = ProjectConfig()
        assert config.name == "未命名项目"
        assert 0 in config.class_map
        assert config.confidence_threshold == 0.5

    def test_custom_config(self):
        config = ProjectConfig(
            name="测试项目",
            confidence_threshold=0.3,
        )
        assert config.name == "测试项目"
        assert config.confidence_threshold == 0.3
