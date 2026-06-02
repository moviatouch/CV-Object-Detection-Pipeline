from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from config import DETECTION_MODEL_CONFIDENCE, MODEL_PATH, OUTPUT_PATHS, ROI_CONFIG_DIR
from main import PipelineStatus, run_pipeline


st.set_page_config(
    page_title="Viatouch Processing UI",
    page_icon="VS",
    layout="wide",
)


def _default_session_id() -> str:
    """Generate a simple session id for the UI."""
    return f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _save_upload(uploaded_file, target_path: Path) -> Path:
    """Persist an uploaded file so the pipeline can read it like a normal video."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def _product_rows(status: PipelineStatus) -> list[dict]:
    """Build product metrics rows for a dataframe."""
    products = set(status.pickup_by_product) | set(status.putback_by_product) | set(status.net_by_product)
    return [
        {
            "Product": product,
            "Pickups": status.pickup_by_product.get(product, 0),
            "Putbacks": status.putback_by_product.get(product, 0),
            "Net Inventory": status.net_by_product.get(product, 0),
        }
        for product in sorted(products)
    ]


st.title("Viatouch CV Processing UI")
st.caption("Upload the top-left and top-right camera videos, then run the existing pipeline with a live preview.")

with st.sidebar:
    st.header("Run Settings")
    session_id = st.text_input("Session ID", value=_default_session_id())
    device = st.text_input("Device", value="cuda")
    det_conf = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=1.00,
        value=float(DETECTION_MODEL_CONFIDENCE),
        step=0.05,
    )
    roi_dir = st.text_input("ROI directory", value=ROI_CONFIG_DIR)
    model_path = st.text_input("Model path", value=MODEL_PATH)
    callback_every_n_frames = st.number_input("UI refresh every N frames", min_value=1, max_value=30, value=1, step=1)

video0_file = None
video1_file = None

upload_col_0, upload_col_1 = st.columns(2)
with upload_col_0:
    video0_file = st.file_uploader("Top Left Cam Video", type=["mp4", "avi", "mov", "mkv"], key="video0")
with upload_col_1:
    video1_file = st.file_uploader("Top Right Cam Video", type=["mp4", "avi", "mov", "mkv"], key="video1")

command_box = st.empty()
progress_placeholder = st.empty()
preview_placeholder = st.empty()
status_placeholder = st.empty()
metric_cols = st.columns(5)
product_placeholder = st.empty()
event_placeholder = st.empty()
result_placeholder = st.empty()

ready_to_run = video0_file is not None and video1_file is not None
start_processing = st.button("Start Processing", type="primary", disabled=not ready_to_run)

if start_processing:
    upload_root = OUTPUT_PATHS.root / "streamlit_uploads" / session_id
    video0_path = _save_upload(video0_file, upload_root / f"cam0_{video0_file.name}")
    video1_path = _save_upload(video1_file, upload_root / f"cam1_{video1_file.name}")

    command_box.code(
        "python main.py "
        f"--video0 {video0_path} "
        f"--video1 {video1_path} "
        f"--session_id {session_id}",
        language="bash",
    )

    progress_bar = progress_placeholder.progress(0.0, text="Waiting for first frame...")

    def render_status(status: PipelineStatus) -> bool:
        """Update the Streamlit placeholders from the latest pipeline status."""
        progress_bar.progress(
            status.progress,
            text=(
                f"Frames processed: {status.frames_processed}/{status.total_frames or '?'} "
                f"(frame index {status.current_frame_index})"
            ),
        )

        metric_cols[0].metric("Pickups Count", status.pickup_total)
        metric_cols[1].metric("Putback Count", status.putback_total)
        metric_cols[2].metric("Frames Processed", status.frames_processed)
        metric_cols[3].metric("Net Inventory", status.net_inventory)
        metric_cols[4].metric("Sync", "OK" if status.sync_ok else "Frozen")

        status_bits = []
        if not status_bits:
            status_bits.append("Processing normally")
        status_placeholder.info(" | ".join(status_bits))

        if status.preview_bgr is not None:
            preview_rgb = cv2.cvtColor(status.preview_bgr, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(preview_rgb, caption="Live processing preview", use_container_width=True)

        rows = _product_rows(status)
        if rows:
            product_placeholder.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            product_placeholder.info("No product events detected yet.")

        if status.events:
            event_placeholder.table(pd.DataFrame(status.events))

        return True

    try:
        result = run_pipeline(
            video0=str(video0_path),
            video1=str(video1_path),
            session_id=session_id,
            device=device,
            roi_dir=roi_dir,
            model_path=model_path,
            det_conf=float(det_conf),
            show_preview=False,
            status_callback=render_status,
            callback_every_n_frames=int(callback_every_n_frames),
        )
        progress_bar.progress(1.0, text=f"Completed {result.frames_processed} frames.")
        result_placeholder.success(
            f"Session complete. Pickups: {result.pickup_total}, Putbacks: {result.putback_total}, Net Inventory: {result.net_inventory}"
        )
        st.write(f"Session JSON: `{result.session_path}`")
        st.write(f"Log file: `{result.log_path}`")
    except Exception as exc:
        result_placeholder.error(f"Processing failed: {exc}")
        st.code(traceback.format_exc())
