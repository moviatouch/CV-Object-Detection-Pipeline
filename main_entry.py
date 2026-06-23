import os
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from main import run_pipeline, PipelineResult
from utils.media_uploader import upload_video
from utils.grace_period_check import run_grace_period_check, get_current_line_items
from utils import login
from config import VICKI_APP, MODEL_TO_DASHBOARD_MAPPING

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

def get_video_paths(transid_folder: str) -> tuple[str, str]:
    """Find video0 (wide-angle/media4) and video1 (close-up/media0) in the transaction folder."""
    folder = Path(transid_folder)
    videos = list(folder.glob("*.mp4"))
    if len(videos) < 2:
        raise ValueError(f"Not enough videos in {transid_folder}")
    
    video0 = None  # wide-angle (media4 / video4 / video2)
    video1 = None  # close-up (media0 / video0 / video1)
    
    # Try pattern match
    for v in videos:
        name = v.name.lower()
        if "media4" in name or "video4" in name or "video2" in name:
            video0 = str(v)
        elif "media0" in name or "video0" in name or "video1" in name:
            video1 = str(v)
            
    # Fallback if patterns aren't explicitly matched
    if not video0 or not video1:
        # Alphabetical sorting puts media0 first and media4 second.
        # We want video0 to be wide-angle (media4) and video1 to be close-up (media0).
        sorted_videos = sorted(videos)
        video0 = str(sorted_videos[1])
        video1 = str(sorted_videos[0])
        
    return video0, video1

def get_all_config_products() -> List[str]:
    """Return all dashboard product names (planogram names)."""
    return list(MODEL_TO_DASHBOARD_MAPPING.values())

def get_order_line_items(trans_id: str) -> List[Dict]:
    """Fetch current order line items from loyalty API."""
    base_url, machine_id, machine_token, machine_api_key = login.get_custom_machine_settings(VICKI_APP, logger)
    access_token = login.get_current_access_token(base_url, machine_id, machine_token, machine_api_key, logger)
    order = get_current_line_items(base_url, trans_id, access_token)
    # The response structure may vary; we need the line_items list.
    # Adjust based on actual response (e.g., order.get('line_items', []))
    return order.get('line_items', [])

def create_user_activities_json(trans_id: str, transid_folder: str, line_items: List[Dict]) -> str:
    """Create user_actvities.json in transaction folder."""
    data = {
        "user_activity_instance": {
            "report_id": trans_id,
            "line_items": line_items
        }
    }
    filepath = os.path.join(transid_folder, "user_actvities.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return filepath

def build_cv_activities(pipeline_result: PipelineResult) -> List[Dict[str, str]]:
    """Convert pipeline summary to list of pickup/putback events with dashboard product names."""
    activities = []
    for model_name, count in pipeline_result.pickup_by_product.items():
        dash_name = MODEL_TO_DASHBOARD_MAPPING.get(model_name, model_name)
        for _ in range(count):
            activities.append({"USER_PICKUP": dash_name})
    for model_name, count in pipeline_result.putback_by_product.items():
        dash_name = MODEL_TO_DASHBOARD_MAPPING.get(model_name, model_name)
        for _ in range(count):
            activities.append({"USER_PUTBACK": dash_name})
    return activities

def process_transaction(trans_id: str, transid_folder: str, customer_trans: str = 'True'):
    """Main orchestration for a transaction."""
    logger.info(f"Processing transaction {trans_id} in {transid_folder}")

    # Step 1: Locate videos
    try:
        video0, video1 = get_video_paths(transid_folder)
    except Exception as e:
        logger.error(f"Error locating videos: {e}")
        return

    # Step 2: Run CV pipeline
    logger.info(f"Running pipeline on {video0} and {video1}")
    try:
        pipeline_result = run_pipeline(
            video0=video0,
            video1=video1,
            session_id=trans_id,
            device="cuda",                  # or "cpu" if no GPU
            roi_dir="config/roi",
            model_path="models/best_10_6.pt",
            det_conf=0.6,
            show_preview=False,
            status_callback=None,
            callback_every_n_frames=1,
            preview_panel_size=(960, 540)
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return

    if not pipeline_result.completed:
        logger.error(f"Pipeline did not complete successfully for {trans_id}")
        return

    # Step 3: Build activities and net summary
    cv_activities = build_cv_activities(pipeline_result)
    net_summary = {
        "pickup_by_product": pipeline_result.pickup_by_product,
        "putback_by_product": pipeline_result.putback_by_product,
        "net_by_product": pipeline_result.net_by_product,
        "total_pickup": pipeline_result.pickup_total,
        "total_putback": pipeline_result.putback_total,
        "net_inventory": pipeline_result.net_inventory,
    }
    logger.info(f"Net summary: {net_summary}")

    # Step 4: Fetch order and create user_activities.json
    try:
        line_items = get_order_line_items(trans_id)
    except Exception as e:
        logger.error(f"Failed to fetch order line items: {e}")
        # Proceed with empty line_items? The grace period will fetch again anyway.
        line_items = []
    user_activities_path = create_user_activities_json(trans_id, transid_folder, line_items)

    # Step 5: Grace period check (dashboard update)
    all_config_products = get_all_config_products()
    try:
        run_grace_period_check(
            transaction_id=trans_id,
            json_file=user_activities_path,
            cv_activities=cv_activities,
            all_config_products=all_config_products
        )
    except Exception as e:
        logger.error(f"Grace period check failed: {e}")

    # Step 6: Upload media (zip + upload)
    try:
        upload_video(
            logger=logger,
            trans_id=trans_id,
            post_transid=trans_id,
            transid_folder=transid_folder,
            customer_trans=customer_trans
        )
    except Exception as e:
        logger.error(f"Media upload failed: {e}")

    # Mark as processed (to avoid re-processing)
    with open(os.path.join(transid_folder, "processed.txt"), 'w') as f:
        f.write("processed")
    logger.info(f"Transaction {trans_id} processing finished.")