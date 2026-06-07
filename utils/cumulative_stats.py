from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class CumulativeStats:
    def __init__(self, stats_path: Path):
        self.stats_path = stats_path
        self.data = self._load()

    def _load(self) -> dict:
        if self.stats_path.exists():
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        return {
            "videos_processed": 0,
            "total_pickups": 0,
            "total_putbacks": 0,
            "sessions": []
        }

    def save(self) -> None:
        with open(self.stats_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add_session(self, video_name: str, pickups: int, putbacks: int, session_id: str = "") -> None:
        self.data["videos_processed"] += 1
        self.data["total_pickups"] += pickups
        self.data["total_putbacks"] += putbacks
        self.data["sessions"].append({
            "video_name": video_name,
            "pickups": pickups,
            "putbacks": putbacks,
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        })
        self.save()

    def get_summary_line(self) -> str:
        return f"Videos processed: {self.data['videos_processed']} | Total pickups: {self.data['total_pickups']} | Total putbacks: {self.data['total_putbacks']}"

    def get_display_lines(self) -> List[str]:
        return [
            f"All Videos: {self.data['videos_processed']} processed",
            f"Total Pickups: {self.data['total_pickups']}",
            f"Total Putbacks: {self.data['total_putbacks']}",
            f"Net: {self.data['total_pickups'] - self.data['total_putbacks']}"
        ]