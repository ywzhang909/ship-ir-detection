# -*- coding: utf-8 -*-
"""视频/图像输入的训练域对齐预处理（与仓库既有约定一致）。

本模块是根目录 `detector.py` 推理链在 ship_detector 内的忠实复用，分两步：

1. **灰度归一化（BGR→GRAY→BGR）** —— 无条件执行。
   仓库内所有模型都在单波段红外域上训练，参考脚本
   (``inference_customn_fusion.py``) 先 ``IMREAD_GRAYSCALE`` 再 ``GRAY2BGR``。
   因此无论输入视频是彩色还是准灰度，都必须先折叠到单波段，保证色彩空间与
   训练一致。对应 ``detector.read_image_bgr``。

2. **按训练输入分布路由的预处理流水线** —— ``prep`` 选择：
   ``raw`` / ``tophat`` / ``butterworth`` / ``dual`` / ``wavelet`` / ``rpca``
   对应 ``detector.preprocess_for``，参数逐项对齐，避免"另发明一套"。

几何不变性: 上述流水线都是逐像素/逐通道映射, 输出与输入同尺寸同通道数,
因此检测框坐标可直接映射回原始画面。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent.parent
_YT_DIR = _REPO / "yolo-training"

KNOWN_PREPS = ("raw", "tophat", "butterworth", "dual", "wavelet", "rpca")


def normalize_ir(frame: np.ndarray) -> np.ndarray:
    """BGR→GRAY→BGR 灰度归一化（与 detector.read_image_bgr 完全一致）。"""
    import cv2

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[2] != 3:
        raise ValueError(f"期望 1/3/4 通道输入, 实际 {frame.shape}")
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _import_pipeline_types():
    """延迟导入 yolo-training/preprocessing_module（含它时才需要）。"""
    if str(_YT_DIR) not in sys.path:
        sys.path.insert(0, str(_YT_DIR))
    from preprocessing_module import (  # noqa: PLC0415
        ColorCorrectionMethod,
        EnhancementMethod,
        PreprocessingConfig,
        PreprocessingPipeline,
        PreprocessingPipelineDual,
        SeaClutterMethod,
    )
    return (ColorCorrectionMethod, EnhancementMethod, PreprocessingConfig,
            PreprocessingPipeline, PreprocessingPipelineDual, SeaClutterMethod)


def apply_prep(frame: np.ndarray, prep: str = "raw") -> np.ndarray:
    """按训练输入分布应用预处理流水线（参数与 detector.preprocess_for 对齐）。

    输入应为已灰度归一化的 3 通道 BGR；``raw`` 直接返回。
    """
    prep = (prep or "raw").lower()
    if prep == "raw":
        return frame
    if prep not in KNOWN_PREPS:
        raise ValueError(f"未知预处理 {prep!r}, 可选: {KNOWN_PREPS}")

    (CCM, EM, PC, PP, PPD, SCM) = _import_pipeline_types()

    if prep == "dual":
        return PPD()(frame)
    if prep == "wavelet":
        return PPD(freq_method=SCM.WAVELET)(frame)
    if prep == "rpca":
        return PPD(freq_method=SCM.RPCA)(frame)

    common = dict(
        color_correction_method=CCM.SEA_WHITE_BALANCE,
        wb_corner_fraction=0.06,
        wb_gray_target=128.0,
        enhancement_method=EM.CLAHE,
        clahe_clip_limit=3.0,
        edge_enhance=True,
        unsharp_strength=0.5,
    )
    if prep == "tophat":
        config = PC(sea_clutter_method=SCM.TOPHAT, tophat_kernel=31, **common)
    else:  # butterworth
        config = PC(sea_clutter_method=SCM.BUTTERWORTH_BPF,
                    butterworth_cutoff_low=8.0, butterworth_cutoff_high=80.0,
                    **common)
    return PP(config)(frame)


def preprocess_frame(frame: np.ndarray, prep: str = "raw",
                     normalize: bool = True) -> np.ndarray:
    """完整链: 灰度归一化 -> 训练域流水线。"""
    out = normalize_ir(frame) if normalize else frame
    return apply_prep(out, prep)


def is_prep_available(prep: str) -> bool:
    """该预处理是否可在当前环境运行（raw 恒可用）。"""
    if (prep or "raw").lower() == "raw":
        return True
    try:
        _import_pipeline_types()
        return True
    except Exception:  # noqa: BLE001 — 缺依赖时如实报告不可用
        return False
