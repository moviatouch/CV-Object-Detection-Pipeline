
import argparse
from pipeline.pipeline import run_pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vending pickup/putback detector.")
    parser.add_argument("--video0", required=True, help="Path to camera 0 video")
    parser.add_argument("--video1", required=True, help="Path to camera 1 video")
    parser.add_argument("--session_id", required=True, help="Session identifier")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--roi_dir", default=ROI_CONFIG_DIR, help="ROI config directory")
    parser.add_argument("--model_path", default=MODEL_PATH, help="RT-DETR model path")
    parser.add_argument("--det_conf", type=float, default=DETECTION_MODEL_CONFIDENCE, help="Detection confidence")
    parser.add_argument("--show_preview", action=argparse.BooleanOptionalAction, default=SHOW_PREVIEW)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    run_pipeline(
        video0=args.video0,
        video1=args.video1,
        session_id=args.session_id,
        device=args.device,
        roi_dir=args.roi_dir,
        model_path=args.model_path,
        det_conf=args.det_conf,
        show_preview=args.show_preview,
    )


if __name__ == "__main__":
    main()