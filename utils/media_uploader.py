import threading
import time
import requests
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from utils import login
from utils.send_alert import send_alert
import config
import logging

os.makedirs(os.path.join(config.BASE_PATH, 'post_archive'), exist_ok=True)


def compute_picked_products(base_url, post_transid):
    """
    Fetch order details and calculate total product pickup count for the transaction.
    """
    url = "{}/loyalty/orders/{}".format(base_url, post_transid)
    headers = {"invoice_number": post_transid}
    total_picked_products = 0
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            activities = response.json()
            for activity in activities.get('invoice', {}).get('line_items', []):
                total_picked_products += activity.get('quantity', 0)
    except Exception as e:
        logging.error(f"Error fetching product pickup count: {e}")
    return total_picked_products


def convert_to_h264(input_path, output_path):
    """
    Re-encode an existing mp4v video to real H.264/yuv420p, no resizing
    (annotated videos are already correctly sized at write-time).
    """
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-i", str(input_path),
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logging.error(f"ffmpeg convert failed for {input_path}: {result.stderr.decode(errors='ignore')}")
        return False
    return True


def upload_video(logger, trans_id, post_transid, transid_folder, customer_trans):
    """
    Zip the transaction folder after processing only the annotated computer vision videos,
    then upload the zip to the loyalty API.
    """
    # Resolve paths for the annotated processed videos from the CV output
    annotated_video0 = config.OUTPUT_PATHS.annotated_videos / f"{trans_id}_cam0.mp4"
    annotated_video1 = config.OUTPUT_PATHS.annotated_videos / f"{trans_id}_cam1.mp4"

    # Check if the annotated computer vision videos exist
    if not (annotated_video0.exists() and annotated_video1.exists()):
        logger.error(f"Annotated CV videos missing for transaction {trans_id}. Skipping media upload entirely.")
        return False

    # ------------------------------------------------------------
    # 1. Create temporary directory for the zip contents
    # ------------------------------------------------------------
    temp_dir = tempfile.mkdtemp(prefix=f'resized_{trans_id}_')
    logger.info(f"Created temporary directory: {temp_dir}")

    try:
        # ------------------------------------------------------------
        # 2. Convert the already-resized annotated CV videos to H.264
        #    and place them in the temp dir under their expected output names
        # ------------------------------------------------------------
        logger.info(f"Converting annotated videos to H.264 for {trans_id}...")
        try:
            # Avoid circular import at startup by importing inside function
            from main import get_video_paths
            video0_path, video1_path = get_video_paths(transid_folder)
            video0_name = Path(video0_path).name
            video1_name = Path(video1_path).name
        except Exception as e:
            logger.warning(f"Could not map annotated videos to original names: {e}. Defaulting to media4.mp4 and media0.mp4")
            video0_name = "media4.mp4"
            video1_name = "media0.mp4"

        videos_to_process = [
            (annotated_video0, video0_name),
            (annotated_video1, video1_name)
        ]

        for src_vid_path, out_name in videos_to_process:
            out_path = os.path.join(temp_dir, out_name)
            logger.info(f"Converting {src_vid_path} -> {out_path}")
            success = convert_to_h264(str(src_vid_path), out_path)
            if not success:
                logger.warning(f"H.264 conversion failed for {src_vid_path}, copying as-is.")
                shutil.copy2(str(src_vid_path), out_path)

        # Copy over metadata and non-video files (.json, logs, etc.) from original transaction folder
        source_path = Path(transid_folder)
        for file_path in source_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() != '.mp4':
                shutil.copy2(str(file_path), temp_dir)
                logger.debug(f"Copied non-video file: {file_path.name}")

        zip_source = temp_dir

        # ------------------------------------------------------------
        # 3. Create zip archive
        # ------------------------------------------------------------
        zip_filename = f'{post_transid}.zip'
        zip_dest = os.path.join(config.BASE_PATH, 'post_archive', zip_filename)
        logger.info(f"Creating archive: {zip_dest}")

        base_name = os.path.join(config.BASE_PATH, 'post_archive', post_transid)
        shutil.make_archive(base_name, 'zip', zip_source)

        # ------------------------------------------------------------
        # 4. Upload the zip
        # ------------------------------------------------------------
        file_size_bytes = os.path.getsize(zip_dest)
        file_size_mb = file_size_bytes / (1024 * 1024)
        logger.info(f"Archive size: {file_size_mb:.2f} MB")

        base_url, machine_id, machine_token, machine_api_key = login.get_custom_machine_settings(config.VICKI_APP, logger)
        access_token = login.get_current_access_token(base_url, machine_id, machine_token, machine_api_key, logger)

        with open(zip_dest, 'rb') as fileobj:
            if customer_trans == 'False':
                url = f"{base_url}/loyalty/upload-media/cv?media_event_type=TECHNICIAN_MODE&invoice_id={post_transid}"
            else:
                url = f"{base_url}/loyalty/upload-media/cv?media_event_type=COMPUTER_VISION&invoice_id={post_transid}"
            logger.info(url)
            headers = {"Authorization": f"Bearer {access_token}"}
            response_media = requests.post(url, files={'file': fileobj}, headers=headers)

            if response_media.status_code == 200:
                status = f'Media Uploaded Successfully (Transaction: {post_transid}) File-size: {file_size_mb:.2f} MB'
                logger.info(f"Uploaded {post_transid}")
                # os.remove(zip_dest)
                # for video_path in (annotated_video0, annotated_video1):
                #     try:
                #         video_path.unlink()
                #         logger.info(f"Deleted saved video: {video_path}")
                #     except FileNotFoundError:
                #         pass
                #     except Exception as e:
                #         logger.warning(f"Could not delete {video_path}: {e}")
                logger.info("Cleaned Transaction")
                threading.Thread(target=send_alert, args=(logger, config.VICKI_APP, status, False)).start()
            elif response_media.status_code == 504:
                status = f'Endpoint Time-out Error for Media Upload ({response_media.status_code}) (Transaction: {post_transid})'
                logger.error(status)
                threading.Thread(target=send_alert, args=(logger, config.VICKI_APP, status)).start()
            else:
                status = f'Media Upload Failed (Transaction: {post_transid})'
                logger.error(f"Upload failed: {response_media.status_code} - {response_media.text}")
                threading.Thread(target=send_alert, args=(logger, config.VICKI_APP, status)).start()

    except Exception as e:
        logger.exception(f"Exception during upload workflow: {e}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")