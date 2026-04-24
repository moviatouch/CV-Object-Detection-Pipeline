"""
Pipeline metrics tracking and reporting.

Tracks performance metrics including:
- Video properties (FPS, frames, resolution)
- Timing statistics (total time, avg per frame)
- System resource utilization (CPU, GPU)
- Pipeline throughput (FPS)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import GPUtil
except ImportError:
    GPUtil = None


@dataclass
class CameraMetrics:
    """Metrics for a single camera."""
    camera_id: int
    video_fps: float = 0.0
    total_frames: int = 0
    resolution: tuple[int, int] = (0, 0)
    frame_count: int = 0
    frames_processed: int = 0
    total_time_ms: float = 0.0
    
    def get_summary(self) -> Dict[str, float | int | tuple]:
        """Get summary of camera metrics."""
        avg_time_per_frame = (self.total_time_ms / self.frames_processed) if self.frames_processed > 0 else 0.0
        pipeline_fps = (1000.0 / avg_time_per_frame) if avg_time_per_frame > 0 else 0.0
        
        return {
            "camera_id": self.camera_id,
            "video_fps": self.video_fps,
            "total_frames": self.total_frames,
            "resolution": self.resolution,
            "frames_processed": self.frames_processed,
            "avg_time_per_frame_ms": round(avg_time_per_frame, 2),
            "pipeline_fps": round(pipeline_fps, 2),
        }


@dataclass
class SystemMetrics:
    """System resource utilization metrics."""
    cpu_percent: float = 0.0
    cpu_count: int = 0
    memory_percent: float = 0.0
    memory_available_mb: float = 0.0
    gpu_utilized: bool = False
    gpu_devices: list[Dict[str, float]] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, float | int | bool | list]:
        """Get summary of system metrics."""
        return {
            "cpu_utilization_percent": round(self.cpu_percent, 2),
            "cpu_count": self.cpu_count,
            "memory_utilization_percent": round(self.memory_percent, 2),
            "memory_available_mb": round(self.memory_available_mb, 2),
            "gpu_available": self.gpu_utilized,
            "gpu_devices": self.gpu_devices,
        }


@dataclass
class PipelineMetrics:
    """Overall pipeline metrics and performance tracking."""
    session_id: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    camera_metrics: Dict[int, CameraMetrics] = field(default_factory=dict)
    system_metrics: SystemMetrics = field(default_factory=SystemMetrics)
    frame_times: Dict[int, list[float]] = field(default_factory=dict)  # ms per frame
    peak_cpu_percent: float = 0.0
    peak_memory_percent: float = 0.0
    
    def __post_init__(self):
        """Initialize camera metrics dictionaries."""
        self.frame_times = {0: [], 1: []}
    
    def record_frame_time(self, camera_id: int, elapsed_time_ms: float) -> None:
        """Record frame processing time for a camera."""
        if camera_id not in self.frame_times:
            self.frame_times[camera_id] = []
        self.frame_times[camera_id].append(elapsed_time_ms)
        
        if camera_id in self.camera_metrics:
            self.camera_metrics[camera_id].total_time_ms += elapsed_time_ms
            self.camera_metrics[camera_id].frames_processed += 1
    
    def update_system_metrics(self) -> None:
        """Update system resource utilization metrics."""
        if psutil is not None:
            try:
                self.system_metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
                self.system_metrics.cpu_count = psutil.cpu_count()
                
                vm = psutil.virtual_memory()
                self.system_metrics.memory_percent = vm.percent
                self.system_metrics.memory_available_mb = vm.available / (1024 * 1024)
                
                self.peak_cpu_percent = max(self.peak_cpu_percent, self.system_metrics.cpu_percent)
                self.peak_memory_percent = max(self.peak_memory_percent, self.system_metrics.memory_percent)
            except Exception:
                pass
        
        if GPUtil is not None:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.system_metrics.gpu_utilized = True
                    self.system_metrics.gpu_devices = [
                        {
                            "id": gpu.id,
                            "name": gpu.name,
                            "load_percent": round(gpu.load * 100, 2),
                            "memory_used_mb": round(gpu.memoryUsed, 2),
                            "memory_total_mb": round(gpu.memoryTotal, 2),
                            "memory_percent": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 2),
                            "temperature": gpu.temperature,
                        }
                        for gpu in gpus
                    ]
            except Exception:
                pass
    
    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.end_time = time.time()
    
    def get_total_time_seconds(self) -> float:
        """Get total execution time in seconds."""
        if self.end_time is None:
            return (time.time() - self.start_time)
        return (self.end_time - self.start_time)
    
    def print_summary(self, logger) -> None:
        """Print comprehensive pipeline metrics summary."""
        total_time = self.get_total_time_seconds()
        
        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        # Overall timing
        logger.info("\n--- OVERALL TIMING ---")
        logger.info("Total Execution Time: %.2f seconds", total_time)
        logger.info("Start Time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_time)))
        if self.end_time:
            logger.info("End Time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.end_time)))
        
        # Per-camera metrics
        logger.info("\n--- PER-CAMERA METRICS ---")
        for camera_id, metrics in sorted(self.camera_metrics.items()):
            summary = metrics.get_summary()
            logger.info("\nCamera %d:", camera_id)
            logger.info("  Video FPS: %.2f", summary["video_fps"])
            logger.info("  Resolution: %dx%d", summary["resolution"][0], summary["resolution"][1])
            logger.info("  Total Frames: %d", summary["total_frames"])
            logger.info("  Frames Processed: %d", summary["frames_processed"])
            logger.info("  Total Time: %.2f seconds", metrics.total_time_ms / 1000.0)
            logger.info("  Average Time per Frame: %.2f ms", summary["avg_time_per_frame_ms"])
            logger.info("  Pipeline FPS: %.2f", summary["pipeline_fps"])
        
        # System metrics
        logger.info("\n--- SYSTEM RESOURCE UTILIZATION ---")
        sys_summary = self.system_metrics.get_summary()
        logger.info("CPU Utilization: %.2f%% (%d cores)", sys_summary["cpu_utilization_percent"], sys_summary["cpu_count"])
        logger.info("Peak CPU Utilization: %.2f%%", self.peak_cpu_percent)
        logger.info("Memory Utilization: %.2f%%", sys_summary["memory_utilization_percent"])
        logger.info("Peak Memory Utilization: %.2f%%", self.peak_memory_percent)
        logger.info("Available Memory: %.2f MB", sys_summary["memory_available_mb"])
        
        if sys_summary["gpu_available"]:
            logger.info("\nGPU Information:")
            for gpu in sys_summary["gpu_devices"]:
                logger.info("  Device %d - %s:", gpu["id"], gpu["name"])
                logger.info("    Load: %.2f%%", gpu["load_percent"])
                logger.info("    Memory: %.2f MB / %.2f MB (%.2f%%)", 
                           gpu["memory_used_mb"], gpu["memory_total_mb"], gpu["memory_percent"])
                if gpu["temperature"]:
                    logger.info("    Temperature: %.1f°C", gpu["temperature"])
        else:
            logger.info("GPU: Not available or not detected")
        
        logger.info("\n" + "=" * 80)
    
    def print_summary_console(self) -> None:
        """Print comprehensive pipeline metrics summary to console."""
        total_time = self.get_total_time_seconds()
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        # Overall timing
        print("\n--- OVERALL TIMING ---")
        print(f"Total Execution Time: {total_time:.2f} seconds")
        print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time))}")
        if self.end_time:
            print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.end_time))}")
        
        # Per-camera metrics
        print("\n--- PER-CAMERA METRICS ---")
        for camera_id, metrics in sorted(self.camera_metrics.items()):
            summary = metrics.get_summary()
            print(f"\nCamera {camera_id}:")
            print(f"  Video FPS: {summary['video_fps']:.2f}")
            print(f"  Resolution: {summary['resolution'][0]}x{summary['resolution'][1]}")
            print(f"  Total Frames: {summary['total_frames']}")
            print(f"  Frames Processed: {summary['frames_processed']}")
            print(f"  Total Time: {metrics.total_time_ms / 1000.0:.2f} seconds")
            print(f"  Average Time per Frame: {summary['avg_time_per_frame_ms']:.2f} ms")
            print(f"  Pipeline FPS: {summary['pipeline_fps']:.2f}")
        
        # System metrics
        print("\n--- SYSTEM RESOURCE UTILIZATION ---")
        sys_summary = self.system_metrics.get_summary()
        print(f"CPU Utilization: {sys_summary['cpu_utilization_percent']:.2f}% ({sys_summary['cpu_count']} cores)")
        print(f"Peak CPU Utilization: {self.peak_cpu_percent:.2f}%")
        print(f"Memory Utilization: {sys_summary['memory_utilization_percent']:.2f}%")
        print(f"Peak Memory Utilization: {self.peak_memory_percent:.2f}%")
        print(f"Available Memory: {sys_summary['memory_available_mb']:.2f} MB")
        
        if sys_summary["gpu_available"]:
            print("\nGPU Information:")
            for gpu in sys_summary["gpu_devices"]:
                print(f"  Device {gpu['id']} - {gpu['name']}:")
                print(f"    Load: {gpu['load_percent']:.2f}%")
                print(f"    Memory: {gpu['memory_used_mb']:.2f} MB / {gpu['memory_total_mb']:.2f} MB ({gpu['memory_percent']:.2f}%)")
                if gpu["temperature"]:
                    print(f"    Temperature: {gpu['temperature']:.1f}°C")
        else:
            print("GPU: Not available or not detected")
        
        print("\n" + "=" * 80 + "\n")


class MetricsTracker:
    """Context manager for tracking pipeline metrics."""
    
    def __init__(self, session_id: str, logger=None):
        """Initialize metrics tracker."""
        self.metrics = PipelineMetrics(session_id=session_id)
        self.logger = logger
        self._frame_start_time = None
    
    def initialize_camera(self, camera_id: int, video_fps: float, resolution: tuple[int, int], total_frames: int) -> None:
        """Initialize metrics for a camera."""
        self.metrics.camera_metrics[camera_id] = CameraMetrics(
            camera_id=camera_id,
            video_fps=video_fps,
            resolution=resolution,
            total_frames=total_frames,
        )
    
    def start_frame(self) -> None:
        """Start timing a frame."""
        self._frame_start_time = time.time()
    
    def end_frame(self, camera_id: int) -> None:
        """End timing a frame."""
        if self._frame_start_time is not None:
            elapsed_ms = (time.time() - self._frame_start_time) * 1000
            self.metrics.record_frame_time(camera_id, elapsed_ms)
    
    def update_system_metrics(self) -> None:
        """Update system metrics."""
        self.metrics.update_system_metrics()
    
    def finalize(self) -> None:
        """Finalize metrics collection."""
        self.metrics.finalize()
    
    def print_summary(self) -> None:
        """Print metrics summary."""
        if self.logger:
            self.metrics.print_summary(self.logger)
        self.metrics.print_summary_console()
