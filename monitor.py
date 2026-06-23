import os
import time
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from main_entry import process_transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WAIT_TIME = 10  # seconds to wait for files to be fully written


class TransactionHandler(FileSystemEventHandler):
    def __init__(self, watch_dir):
        self.watch_dir = watch_dir
        self.processing = set()

    def on_created(self, event):
        if event.is_directory:
            self.handle_directory(event.src_path)

    def on_modified(self, event):
        # Also catch file modifications inside folders
        if event.is_directory:
            # Optionally, handle modification of the directory itself
            # (e.g., when files are added)
            self.handle_directory(event.src_path)

    def handle_directory(self, dir_path):
        trans_id = os.path.basename(dir_path)
        if trans_id in self.processing:
            return

        # Wait for files to settle
        time.sleep(WAIT_TIME)

        if os.path.exists(os.path.join(dir_path, "processed.txt")):
            logger.info(f"Transaction {trans_id} already processed, skipping.")
            return

        videos = list(Path(dir_path).glob("*.mp4"))
        if len(videos) < 2:
            logger.warning(f"Not enough videos in {dir_path}, skipping for now.")
            return

        self.processing.add(trans_id)
        try:
            logger.info(f"Processing new transaction: {trans_id}")
            process_transaction(trans_id, dir_path)
        except Exception as e:
            logger.error(f"Error processing {trans_id}: {e}")
        finally:
            self.processing.remove(trans_id)


def process_existing_folders(watch_dir):
    """Process any folders already present at startup."""
    for item in Path(watch_dir).iterdir():
        if item.is_dir():
            trans_id = item.name
            if os.path.exists(item / "processed.txt"):
                logger.info(f"Transaction {trans_id} already processed, skipping.")
                continue
            videos = list(item.glob("*.mp4"))
            if len(videos) >= 2:
                logger.info(f"Found existing unprocessed transaction: {trans_id}")
                handler = TransactionHandler(watch_dir)
                handler.handle_directory(str(item))
            else:
                logger.info(f"Transaction {trans_id} has <2 videos, will wait for more.")


def monitor_post_archive(post_archive_path):
    # First, process any existing folders
    process_existing_folders(post_archive_path)

    event_handler = TransactionHandler(post_archive_path)
    observer = Observer()
    observer.schedule(event_handler, post_archive_path, recursive=False)
    observer.start()
    logger.info(f"Monitoring {post_archive_path} for new transactions...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    post_archive = os.environ.get("POST_ARCHIVE_PATH", "/app/post_archive")
    monitor_post_archive(post_archive)