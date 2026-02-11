#!/usr/bin/env python3
"""
Calibration Tool
----------------
Interactive distance calibration wizard for the navigation system.

Features:
- Step-by-step distance calibration
- Reference object measurement
- Validation testing
- Config file generation

Usage:
    python src/calibration_tool.py
"""

import os
import sys
import time
import yaml
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from camera_capture import CameraCapture
from object_detector import ObjectDetector


@dataclass
class CalibrationPoint:
    """A single calibration measurement."""
    actual_distance_m: float
    bbox_height_px: int
    object_class: str
    timestamp: float


@dataclass  
class CalibrationResult:
    """Result of calibration process."""
    object_class: str
    reference_height_m: float
    calibration_points: List[CalibrationPoint]
    
    # Calculated parameters
    focal_length: float = 0.0
    accuracy_percent: float = 0.0
    
    def calculate_focal_length(self) -> float:
        """Calculate focal length from calibration points."""
        if not self.calibration_points:
            return 0.0
        
        focal_lengths = []
        for point in self.calibration_points:
            # focal_length = (bbox_height * distance) / actual_height
            fl = (point.bbox_height_px * point.actual_distance_m) / self.reference_height_m
            focal_lengths.append(fl)
        
        self.focal_length = sum(focal_lengths) / len(focal_lengths)
        return self.focal_length


class CalibrationWizard:
    """Interactive calibration wizard."""
    
    # Default object heights (meters)
    REFERENCE_HEIGHTS = {
        'person': 1.7,
        'chair': 0.45,
        'bottle': 0.25,
        'cup': 0.10,
        'car': 1.5
    }
    
    # Calibration distances to test (meters)
    CALIBRATION_DISTANCES = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.camera: Optional[CameraCapture] = None
        self.detector: Optional[ObjectDetector] = None
        
        self._calibration_points: List[CalibrationPoint] = []
        self._current_class: str = "person"
        self._progress_file: str = "data/calibration_progress.json"
    
    def _save_progress(self, result: 'CalibrationResult') -> None:
        """Save intermediate calibration results to prevent data loss."""
        import json
        from dataclasses import asdict
        
        os.makedirs(os.path.dirname(self._progress_file), exist_ok=True)
        
        data = {
            'object_class': result.object_class,
            'reference_height_m': result.reference_height_m,
            'points': [asdict(p) for p in result.calibration_points],
            'timestamp': time.time()
        }
        
        with open(self._progress_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  [Progress saved]")
    
    def _load_progress(self) -> Optional['CalibrationResult']:
        """Load interrupted calibration (if < 1 hour old)."""
        import json
        
        if not os.path.exists(self._progress_file):
            return None
        
        with open(self._progress_file, 'r') as f:
            data = json.load(f)
        
        # Check if recent (< 1 hour old)
        if time.time() - data['timestamp'] > 3600:
            return None
        
        result = CalibrationResult(
            object_class=data['object_class'],
            reference_height_m=data['reference_height_m'],
            calibration_points=[
                CalibrationPoint(**p) for p in data['points']
            ]
        )
        
        print(f"[Calibration] Found {len(result.calibration_points)} saved points")
        return result
    
    def _start_modules(self) -> bool:
        """Initialize camera and detector."""
        print("[Calibration] Starting camera and detector...")
        
        self.camera = CameraCapture(config_path=self.config_path)
        self.detector = ObjectDetector(config_path=self.config_path)
        
        if not self.camera.start():
            print("[Calibration] ERROR: Failed to start camera")
            return False
        
        return True
    
    def _stop_modules(self) -> None:
        """Stop camera and detector."""
        if self.camera:
            self.camera.stop()
    
    def _get_detection(self, target_class: str, 
                       timeout_sec: float = 10.0) -> Optional[Dict]:
        """Get detection of target class with averaging."""
        print(f"[Calibration] Looking for {target_class}...")
        
        bbox_heights = []
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            frame = self.camera.get_frame(timeout=1.0)
            if frame is None:
                continue
            
            result = self.detector.detect(frame.data, frame_id=0)
            
            for det in result.detections:
                if det.class_name == target_class:
                    bbox = det.bbox
                    height = bbox[3] - bbox[1]  # y2 - y1
                    bbox_heights.append(height)
                    
                    # Show frame with detection
                    display = frame.data.copy()
                    cv2.rectangle(display, 
                                 (bbox[0], bbox[1]), 
                                 (bbox[2], bbox[3]),
                                 (0, 255, 0), 2)
                    cv2.putText(display, 
                               f"{target_class}: {height}px",
                               (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                               (0, 255, 0), 2)
                    cv2.putText(display,
                               f"Samples: {len(bbox_heights)}/10",
                               (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                               (255, 255, 0), 2)
                    cv2.imshow("Calibration", display)
                    cv2.waitKey(1)
            
            # Collect 10 samples for averaging
            if len(bbox_heights) >= 10:
                break
        
        if not bbox_heights:
            return None
        
        return {
            'class_name': target_class,
            'avg_bbox_height': int(np.mean(bbox_heights)),
            'std_bbox_height': np.std(bbox_heights),
            'samples': len(bbox_heights)
        }
    
    def run_distance_calibration(self, target_class: str = "person") -> CalibrationResult:
        """Run interactive distance calibration."""
        print("\n" + "=" * 60)
        print("DISTANCE CALIBRATION WIZARD")
        print("=" * 60)
        
        self._current_class = target_class
        reference_height = self.REFERENCE_HEIGHTS.get(target_class, 1.7)
        
        print(f"\nTarget object: {target_class}")
        print(f"Reference height: {reference_height}m")
        
        # Ask for custom height
        custom = input(f"Enter actual height (or press Enter for {reference_height}m): ").strip()
        if custom:
            try:
                reference_height = float(custom)
            except ValueError:
                pass
        
        result = CalibrationResult(
            object_class=target_class,
            reference_height_m=reference_height,
            calibration_points=[]
        )
        
        if not self._start_modules():
            return result
        
        try:
            for distance in self.CALIBRATION_DISTANCES:
                print(f"\n{'='*40}")
                print(f"Step: Place {target_class} at {distance} meters")
                print("='*40")
                input("Press ENTER when ready...")
                
                detection = self._get_detection(target_class)
                
                if detection is None:
                    print(f"[Calibration] WARNING: No {target_class} detected")
                    retry = input("Retry? (y/n): ").lower()
                    if retry == 'y':
                        detection = self._get_detection(target_class, timeout_sec=20.0)
                
                if detection:
                    point = CalibrationPoint(
                        actual_distance_m=distance,
                        bbox_height_px=detection['avg_bbox_height'],
                        object_class=target_class,
                        timestamp=time.time()
                    )
                    result.calibration_points.append(point)
                    
                    print(f"✓ Recorded: {distance}m → {detection['avg_bbox_height']}px")
                    print(f"  (std: {detection['std_bbox_height']:.1f}px, samples: {detection['samples']})")
                else:
                    print(f"✗ Skipped: {distance}m")
            
            # Calculate focal length
            if result.calibration_points:
                result.calculate_focal_length()
                print(f"\n[Calibration] Calculated focal length: {result.focal_length:.1f}")
            
        finally:
            cv2.destroyAllWindows()
            self._stop_modules()
        
        return result
    
    def validate_calibration(self, result: CalibrationResult) -> Dict:
        """Validate calibration accuracy."""
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)
        
        if result.focal_length == 0:
            print("[Calibration] ERROR: No calibration data")
            return {'error': 'No calibration data'}
        
        if not self._start_modules():
            return {'error': 'Failed to start modules'}
        
        validation_results = []
        
        try:
            for test_distance in [1.5, 2.0, 3.0]:
                print(f"\nPlace {result.object_class} at {test_distance}m")
                input("Press ENTER when ready...")
                
                detection = self._get_detection(result.object_class)
                
                if detection:
                    # Calculate estimated distance
                    estimated = (result.focal_length * result.reference_height_m) / detection['avg_bbox_height']
                    error = abs(estimated - test_distance) / test_distance * 100
                    
                    validation_results.append({
                        'actual_m': test_distance,
                        'estimated_m': estimated,
                        'error_percent': error
                    })
                    
                    status = "✓" if error < 20 else "✗"
                    print(f"{status} Actual: {test_distance}m, Estimated: {estimated:.2f}m (Error: {error:.1f}%)")
        
        finally:
            cv2.destroyAllWindows()
            self._stop_modules()
        
        if validation_results:
            avg_error = sum(v['error_percent'] for v in validation_results) / len(validation_results)
            result.accuracy_percent = 100 - avg_error
            print(f"\n[Calibration] Overall accuracy: {result.accuracy_percent:.1f}%")
        
        return {
            'validation_points': validation_results,
            'accuracy_percent': result.accuracy_percent
        }
    
    def save_calibration(self, result: CalibrationResult, 
                        output_path: str = None) -> str:
        """Save calibration to config file."""
        if output_path is None:
            output_path = self.config_path
        
        # Load existing config
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}
        
        # Update distance estimation config
        if 'safety' not in config:
            config['safety'] = {}
        
        if 'distance_estimation' not in config['safety']:
            config['safety']['distance_estimation'] = {}
        
        dist_cfg = config['safety']['distance_estimation']
        
        # Set calibrated values
        dist_cfg['method'] = 'bbox_height'
        dist_cfg['focal_length'] = result.focal_length
        
        if 'reference_heights' not in dist_cfg:
            dist_cfg['reference_heights'] = {}
        
        dist_cfg['reference_heights'][result.object_class] = result.reference_height_m
        
        # Add calibration metadata
        dist_cfg['calibration'] = {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'object_class': result.object_class,
            'accuracy_percent': result.accuracy_percent,
            'points': len(result.calibration_points)
        }
        
        # Save config
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n[Calibration] Saved to {output_path}")
        return output_path


def main():
    """Run calibration from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distance Calibration Tool")
    parser.add_argument('--object', type=str, default='person',
                       help='Object class to calibrate (default: person)')
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                       help='Config file path')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation, skip calibration')
    
    args = parser.parse_args()
    
    wizard = CalibrationWizard(config_path=args.config)
    
    print("\n" + "=" * 60)
    print("NAVIGATION SYSTEM CALIBRATION TOOL")
    print("=" * 60)
    print("\nThis wizard will help you calibrate distance estimation.")
    print("You will need:")
    print("  1. A clear space (at least 4 meters)")
    print("  2. A reference object (person, chair, etc.)")
    print("  3. A measuring tape or marked distances")
    
    input("\nPress ENTER to begin...")
    
    # Run calibration
    result = wizard.run_distance_calibration(target_class=args.object)
    
    if result.calibration_points:
        # Validate
        validate = input("\nRun validation? (y/n): ").lower()
        if validate == 'y':
            wizard.validate_calibration(result)
        
        # Save
        save = input("\nSave calibration to config? (y/n): ").lower()
        if save == 'y':
            wizard.save_calibration(result)
    
    print("\n[Calibration] Complete!")


if __name__ == "__main__":
    main()
