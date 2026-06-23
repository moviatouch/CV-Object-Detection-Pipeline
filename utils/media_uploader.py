import threading
import time
import requests
import os
import shutil
import tempfile
from pathlib import Path
from moviepy import VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from utils import login
import cv2
from utils.send_alert import send_alert
import config
import logging

os.makedirs(os.path.join(config.BASE_PATH, 'post_archive'), exist_ok= True)
# os.makedirs(os.path.join(config.BASE_PATH, 'temp'), exist_ok= True)
# os.makedirs(os.path.join(config.BASE_PATH, 'temp_archive'), exist_ok= True)


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


# def resize_video(input_path, output_path, target_width=640, target_height=480):
#     """
#     Resize a video to fit within target dimensions while preserving aspect ratio,
#     compress with lower bitrate, strip audio, and enforce yuv420p for web compatibility.
#     """
#     try:
#         clip = VideoFileClip(input_path)
#         # Resize to fit within target box
#         if hasattr(clip, 'resized'):
#             resized = clip.resized(height=target_height)
#             if resized.w > target_width:
#                 resized = resized.resized(width=target_width)
#         else:
#             resized = clip.resize(height=target_height)
#             if resized.w > target_width:
#                 resized = resized.resize(width=target_width)
        
#         # Enforcing yuv420p makes it natively playable on web dashboards
#         resized.write_videofile(
#             output_path,
#             codec='libx264',
#             bitrate='500k',
#             audio=False,
#             fps=10,
#             ffmpeg_params=['-pix_fmt', 'yuv420p'],
#             logger=None
#         )
#         clip.close()
#         return True
#     except Exception as e:
#         logging.error(f"Error resizing {input_path}: {e}")
#         return False

def resize_video(input_path, output_path, target_width=640, target_height=480):
    """
    Resize a video to fit within target dimensions while preserving aspect ratio,
    reading frames via OpenCV, resizing each in memory, then encoding via moviepy's
    ImageSequenceClip (same proven pipeline used for the dashboard) with
    h264/yuv420p for web compatibility.
    """
    try:
        fps = 10
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error(f"Could not open video {input_path}")
            return False

        source_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frame_interval = max(int(round(source_fps / fps)), 1)

        resized_frames = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV reads BGR; moviepy/ffmpeg expect RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            # Resize to fit within target box, preserving aspect ratio
            scale = target_height / h
            new_w, new_h = int(w * scale), target_height
            if new_w > target_width:
                scale = target_width / w
                new_w, new_h = target_width, int(h * scale)

            resized_frame = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized_frames.append(resized_frame)

            frame_idx += 1

        cap.release()

        if not resized_frames:
            logging.error(f"No frames extracted from {input_path}")
            return False

        out_clip = ImageSequenceClip(resized_frames, fps=fps)
        out_clip.write_videofile(output_path,fps=fps)
        return True

    except Exception as e:
        logging.error(f"Error resizing {input_path}: {e}")
        return False


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
    # 1. Create temporary directory for resized content
    # ------------------------------------------------------------
    temp_dir = tempfile.mkdtemp(prefix=f'resized_{trans_id}_')
    logger.info(f"Created temporary directory: {temp_dir}")

    try:
        # ------------------------------------------------------------
        # 2. Map and process ONLY the annotated CV videos
        # ------------------------------------------------------------
        logger.info(f"Processing annotated videos for {trans_id}...")
        try:
            # Avoid circular import at startup by importing inside function
            from main_entry import get_video_paths
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

        # Resize and compress the target CV videos
        for src_vid_path, out_name in videos_to_process:
            out_path = os.path.join(temp_dir, out_name)
            logger.info(f"Resizing/compressing {src_vid_path} -> {out_path}")
            success = resize_video(str(src_vid_path), out_path)
            if not success:
                logger.warning(f"Resize failed for {src_vid_path}, copying annotated file as-is.")
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