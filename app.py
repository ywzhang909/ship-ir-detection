"""Streamlit UI for YOLO infrared ship detection.

Thin presentation layer: all model discovery, loading, inference, and metric
math live in ``detector`` (which also bootstraps ``sys.path`` for fusion
checkpoints). This module only handles widgets, session state, and rendering.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import detector

st.set_page_config(page_title="Ship Detection", page_icon="🚢", layout="wide")

MODELS = detector.discover_models()
if not MODELS:
    st.error(
        "No trained models found. Expected at least one checkpoint at "
        "`runs/detect/ship-detection/*/weights/best.pt`. Train a model or "
        "point the runs directory at an existing experiment folder."
    )
    st.stop()

MODEL_BY_PATH = {str(m.path): m for m in MODELS}

# Download buttons trigger a rerun; results must survive it.
if "results" not in st.session_state:
    st.session_state["results"] = None


@st.cache_resource(show_spinner="Loading model…")
def get_model(path_str: str) -> detector.LoadedModel:
    """Load and cache a YOLO model, keyed ONLY by its path string."""
    return detector.load_model(path_str)


def _metrics_table(metrics: dict) -> pd.DataFrame:
    """Two-column Metric/Value view of a per-image metrics dict (no filename)."""
    rows = []
    for key, value in metrics.items():
        if key == "filename":
            continue
        if key == "per_class" and isinstance(value, dict):
            value = ", ".join(f"{n}: {c}" for n, c in value.items()) or "-"
        rows.append({"Metric": key, "Value": value})
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _render_summary(results: list[dict]) -> None:
    st.subheader("Summary")
    successful = [r["metrics"] for r in results if r["metrics"] is not None]
    agg = detector.aggregate_metrics(successful)

    col_total, col_hit, col_dets, col_lat = st.columns(4)
    col_total.metric("Total images", agg["total_images"])
    col_hit.metric("With detections", agg["images_with_detections"])
    col_dets.metric("Total detections", agg["total_detections"])
    col_lat.metric("Mean latency ms", f"{float(agg['mean_latency_ms']):.1f}")

    distribution = agg.get("class_distribution") or {}
    if distribution:
        dist_df = pd.DataFrame({"count": pd.Series(distribution, dtype=int)})
        dist_df.index.name = "class"
        st.bar_chart(dist_df)

    if successful:
        per_image_df = pd.DataFrame(
            [
                {
                    "filename": m["filename"],
                    "detections": m["num_detections"],
                    "latency_ms": m["latency_ms"],
                    "conf_max": m["conf_max"],
                }
                for m in successful
            ]
        )
        st.dataframe(per_image_df, hide_index=True)


def _render_per_image(results: list[dict]) -> None:
    for idx, row in enumerate(results):
        if row["error"] is not None:
            expander_label = f"📷 {row['name']} — error"
        else:
            expander_label = f"📷 {row['name']} — {row['metrics']['num_detections']} detections"

        with st.expander(expander_label):
            if row["error"] is not None:
                st.error(f"{row['name']}: {row['error']}")
                continue

            left, right = st.columns(2)
            with left:
                png_bgr = cv2.imdecode(
                    np.frombuffer(row["png"], dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if png_bgr is None:  # bytes came from cv2.imencode, so unreachable
                    st.warning(f"Could not decode stored PNG for {row['name']}.")
                    continue
                st.image(cv2.cvtColor(png_bgr, cv2.COLOR_BGR2RGB))
            with right:
                st.dataframe(_metrics_table(row["metrics"]), hide_index=True)
                st.download_button(
                    "Download result PNG",
                    data=row["png"],
                    file_name=f"det_{row['name']}.png",
                    mime="image/png",
                    key=f"download_png_{idx}",
                )


with st.sidebar:
    st.header("⚙️ Inference Settings")

    selected_path = st.selectbox(
        "Model",
        options=[str(m.path) for m in MODELS],
        index=0,
        format_func=lambda p: MODEL_BY_PATH[p].label,
        key="model",
    )

    info = MODEL_BY_PATH[selected_path]
    if info.map50_95 is not None or info.map50 is not None:
        bits = []
        if info.map50_95 is not None:
            bits.append(f"mAP50-95: {info.map50_95:.2f}")
        if info.map50 is not None:
            bits.append(f"mAP50: {info.map50:.2f}")
        st.caption(" · ".join(bits))
    else:
        st.caption("No validation metrics available")

    conf = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        key="conf",
    )
    imgsz = st.selectbox(
        "Image size",
        options=[320, 480, 640, 960, 1280],
        index=2,
        key="imgsz",
    )

    device = detector.resolve_device()
    if device.startswith("cuda"):
        st.success(f"GPU: {detector.gpu_display_name()} ({device})")
    else:
        st.warning("CPU mode — CUDA not available")


st.header("🚢 Infrared Ship Detection")

uploads = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    key="uploader",
)

run_clicked = st.button("Run detection", type="primary", disabled=not uploads)

if run_clicked and uploads:
    # Heavy work starts here only: booting the app never touches YOLO weights.
    loaded = get_model(selected_path)
    rows: list[dict] = []
    for file in uploads:
        try:
            bgr = detector.read_image_bgr(file.getvalue())
            dets, annotated, latency_ms = detector.predict_image(
                loaded, bgr, conf=conf, imgsz=int(imgsz)
            )
            rows.append(
                {
                    "name": file.name,
                    "metrics": detector.compute_image_metrics(file.name, dets, latency_ms),
                    "png": cv2.imencode(".png", annotated)[1].tobytes(),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
            rows.append(
                {"name": file.name, "metrics": None, "png": None, "error": str(exc)}
            )
    st.session_state["results"] = rows

results = st.session_state["results"]
if results:
    _render_summary(results)
    _render_per_image(results)
elif not uploads:
    st.info(
        "Upload one or more infrared images (.jpg / .jpeg / .png / .bmp), "
        "then click **Run detection**."
    )
