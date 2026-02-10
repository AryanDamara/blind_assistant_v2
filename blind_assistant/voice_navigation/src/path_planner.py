"""
Path Planner Module
-------------------
Intelligent path planning and obstacle avoidance for navigation.

Features:
- Occupancy grid from depth + detections
- A* path planning algorithm
- Safe corridor detection
- Turn-by-turn guidance generation
- Dynamic path updates
- Cost function (distance + safety + smoothness)

Upgrade from: Simple zone-based alerts
Enhancement: Intelligent pathfinding with obstacle avoidance
"""

import time
import numpy as np
import cv2
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import heapq
import math


@dataclass
class PathPoint:
    """Point in a navigation path."""
    x: int
    y: int
    cost: float = 0.0
    parent: Optional['PathPoint'] = None
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def distance_to(self, other: 'PathPoint') -> float:
        """Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)


@dataclass
class NavigationPath:
    """Complete navigation path with waypoints."""
    waypoints: List[PathPoint]
    total_cost: float
    is_safe: bool
    clearance: float
    length: float
    
    def get_next_waypoint(self, current_pos: Tuple[int, int],
                         lookahead: int = 50) -> Optional[PathPoint]:
        """Get next waypoint for navigation."""
        if not self.waypoints:
            return None
        cx, cy = current_pos
        for wp in self.waypoints:
            if wp.y > cy and wp.distance_to(PathPoint(cx, cy)) < lookahead:
                return wp
        return self.waypoints[-1] if self.waypoints else None
    
    def to_dict(self) -> dict:
        """Convert to dict for telemetry."""
        return {
            'num_waypoints': len(self.waypoints),
            'total_cost': round(self.total_cost, 2),
            'is_safe': self.is_safe,
            'clearance_px': round(self.clearance, 1),
            'length_px': round(self.length, 1)
        }


@dataclass
class NavigationGuidance:
    """Turn-by-turn navigation instructions."""
    direction: str
    distance_to_next_m: float
    obstacle_ahead: Optional[str] = None
    confidence: float = 1.0
    urgency: str = 'normal'
    
    def get_instruction(self, verbosity: str = "standard") -> str:
        """Generate human-readable instruction."""
        if verbosity == "minimal":
            return self.direction
        elif verbosity == "standard":
            if self.direction == 'stop':
                return f"Stop! {self.obstacle_ahead or 'Obstacle'} ahead"
            elif self.direction == 'forward':
                return f"Continue forward for {self.distance_to_next_m:.1f} meters"
            else:
                return f"Turn {self.direction} to avoid obstacles"
        else:
            parts = [f"Navigate {self.direction}"]
            if self.obstacle_ahead:
                parts.append(f"avoiding {self.obstacle_ahead}")
            parts.append(f"for {self.distance_to_next_m:.1f} meters")
            parts.append(f"({self.urgency} priority)")
            return " ".join(parts)


class PathPlanner:
    """
    A* path planner with occupancy grid.
    
    Usage:
        planner = PathPlanner(config_path="config/settings.yaml")
        occupancy = planner.create_occupancy_grid(depth_result, alerts)
        path = planner.plan_path(occupancy, start=(320, 400), goal=(320, 100))
        guidance = planner.get_guidance(path, current_position=(320, 400))
    """
    
    DIRECTIONS = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1)
    ]
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self._load_config(config_path)
        self._total_plans = 0
        self._successful_plans = 0
        self._planning_time_total = 0.0
    
    def _load_config(self, config_path: str) -> None:
        self._grid_resolution = 10
        self._obstacle_threshold = 0.7
        self._safe_clearance_px = 50
        self._max_iterations = 5000
        self._diagonal_cost = 1.414
        self._straight_cost = 1.0
        self._safety_weight = 2.0
        self._smoothness_weight = 0.5
        self._default_goal_distance_m = 3.0
        self._goal_zone = 'center'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                path_cfg = config.get('path_planning', {})
                self._grid_resolution = path_cfg.get('grid_resolution', 10)
                self._obstacle_threshold = path_cfg.get('obstacle_threshold', 0.7)
                self._safe_clearance_px = path_cfg.get('safe_clearance_px', 50)
                self._max_iterations = path_cfg.get('max_iterations', 5000)
                self._safety_weight = path_cfg.get('safety_weight', 2.0)
                self._smoothness_weight = path_cfg.get('smoothness_weight', 0.5)
                self._default_goal_distance_m = path_cfg.get('default_goal_distance_m', 3.0)
        
        print(f"[PathPlanner] Initialized")
        print(f"[PathPlanner] Grid resolution: {self._grid_resolution}px")
        print(f"[PathPlanner] Safe clearance: {self._safe_clearance_px}px")
    
    def create_occupancy_grid(self, depth_result, alerts: List,
                              frame_shape: Tuple[int, int]) -> np.ndarray:
        height, width = frame_shape
        occupancy = np.zeros((height, width), dtype=np.float32)
        
        if depth_result is not None:
            close_threshold = 2.0
            close_mask = depth_result.depth_map_raw < close_threshold
            occupancy[close_mask] = 0.8
        
        for alert in alerts:
            if isinstance(alert, dict):
                bbox = alert.get('bbox', (0, 0, 0, 0))
                danger = alert.get('danger_level', 'info')
            else:
                bbox = alert.bbox
                danger = alert.danger_level
            
            x1, y1, x2, y2 = bbox
            
            if danger == 'critical':
                value = 1.0
            elif danger == 'warning':
                value = 0.8
            else:
                value = 0.5
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            occupancy[y1:y2, x1:x2] = np.maximum(occupancy[y1:y2, x1:x2], value)
            
            margin = self._safe_clearance_px
            x1_m, y1_m = max(0, x1 - margin), max(0, y1 - margin)
            x2_m, y2_m = min(width, x2 + margin), min(height, y2 + margin)
            safety_value = value * 0.5
            occupancy[y1_m:y1, x1_m:x2_m] = np.maximum(occupancy[y1_m:y1, x1_m:x2_m], safety_value)
            occupancy[y2:y2_m, x1_m:x2_m] = np.maximum(occupancy[y2:y2_m, x1_m:x2_m], safety_value)
            occupancy[y1:y2, x1_m:x1] = np.maximum(occupancy[y1:y2, x1_m:x1], safety_value)
            occupancy[y1:y2, x2:x2_m] = np.maximum(occupancy[y1:y2, x2:x2_m], safety_value)
        
        return occupancy
    
    def plan_path(self, occupancy: np.ndarray, start: Tuple[int, int],
                  goal: Tuple[int, int]) -> Optional[NavigationPath]:
        start_time = time.time()
        self._total_plans += 1
        height, width = occupancy.shape
        
        if not (0 <= start[0] < width and 0 <= start[1] < height):
            print(f"[PathPlanner] Invalid start position: {start}")
            return None
        if not (0 <= goal[0] < width and 0 <= goal[1] < height):
            print(f"[PathPlanner] Invalid goal position: {goal}")
            return None
        
        if occupancy[start[1], start[0]] > self._obstacle_threshold:
            print("[PathPlanner] Start position is in obstacle")
            return None
        if occupancy[goal[1], goal[0]] > self._obstacle_threshold:
            goal = self._find_nearest_free_cell(occupancy, goal)
            if goal is None:
                print("[PathPlanner] Goal position unreachable")
                return None
        
        start_node = PathPoint(start[0], start[1])
        goal_node = PathPoint(goal[0], goal[1])
        
        open_set = []
        heapq.heappush(open_set, (0, start_node))
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_node] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[start_node] = self._heuristic(start_node, goal_node)
        closed_set = set()
        iterations = 0
        
        while open_set and iterations < self._max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current == goal_node:
                path = self._reconstruct_path(current)
                planning_time = (time.time() - start_time) * 1000
                self._planning_time_total += planning_time
                self._successful_plans += 1
                print(f"[PathPlanner] Path found: {len(path.waypoints)} waypoints "
                      f"({planning_time:.0f}ms, {iterations} iterations)")
                return path
            
            closed_set.add(current)
            
            for dx, dy in self.DIRECTIONS:
                nx, ny = current.x + dx, current.y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                neighbor = PathPoint(nx, ny)
                if neighbor in closed_set:
                    continue
                occupancy_cost = occupancy[ny, nx]
                if occupancy_cost > self._obstacle_threshold:
                    continue
                move_cost = self._diagonal_cost if (dx != 0 and dy != 0) else self._straight_cost
                safety_penalty = occupancy_cost * self._safety_weight
                tentative_g = g_score[current] + move_cost + safety_penalty
                if tentative_g < g_score[neighbor]:
                    neighbor.parent = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        planning_time = (time.time() - start_time) * 1000
        self._planning_time_total += planning_time
        print(f"[PathPlanner] No path found ({iterations} iterations)")
        return None
    
    def _heuristic(self, a: PathPoint, b: PathPoint) -> float:
        dx = abs(a.x - b.x)
        dy = abs(a.y - b.y)
        return math.sqrt(dx*dx + dy*dy)
    
    def _reconstruct_path(self, goal: PathPoint) -> NavigationPath:
        path_points = []
        current = goal
        while current is not None:
            path_points.append(current)
            current = current.parent
        path_points.reverse()
        
        total_cost = sum(p.cost for p in path_points)
        length = sum(path_points[i].distance_to(path_points[i+1])
                    for i in range(len(path_points)-1))
        
        if self._smoothness_weight > 0:
            path_points = self._smooth_path(path_points)
        
        return NavigationPath(
            waypoints=path_points, total_cost=total_cost,
            is_safe=True, clearance=self._safe_clearance_px, length=length
        )
    
    def _smooth_path(self, waypoints: List[PathPoint]) -> List[PathPoint]:
        if len(waypoints) < 3:
            return waypoints
        smoothed = [waypoints[0]]
        for i in range(1, len(waypoints) - 1):
            prev = smoothed[-1]
            curr = waypoints[i]
            next_pt = waypoints[i + 1]
            angle1 = math.atan2(curr.y - prev.y, curr.x - prev.x)
            angle2 = math.atan2(next_pt.y - curr.y, next_pt.x - curr.x)
            if abs(angle1 - angle2) > 0.3:
                smoothed.append(curr)
        smoothed.append(waypoints[-1])
        return smoothed
    
    def _find_nearest_free_cell(self, occupancy: np.ndarray,
                                position: Tuple[int, int],
                                max_search_radius: int = 100) -> Optional[Tuple[int, int]]:
        height, width = occupancy.shape
        x, y = position
        for radius in range(1, max_search_radius, 5):
            for dx in range(-radius, radius + 1, 5):
                for dy in range(-radius, radius + 1, 5):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if occupancy[ny, nx] < self._obstacle_threshold:
                            return (nx, ny)
        return None
    
    def get_guidance(self, path: NavigationPath, current_position: Tuple[int, int],
                    frame_width: int = 640) -> NavigationGuidance:
        if path is None or not path.waypoints:
            return NavigationGuidance(direction='stop', distance_to_next_m=0.0,
                                     obstacle_ahead='No clear path', urgency='urgent')
        
        next_wp = path.get_next_waypoint(current_position, lookahead=100)
        if next_wp is None:
            return NavigationGuidance(direction='forward', distance_to_next_m=1.0, urgency='normal')
        
        cx, cy = current_position
        center_x = frame_width // 2
        tolerance = frame_width * 0.1
        
        if abs(next_wp.x - center_x) < tolerance:
            direction = 'forward'
        elif next_wp.x < center_x - tolerance:
            direction = 'left'
        else:
            direction = 'right'
        
        distance_px = next_wp.distance_to(PathPoint(cx, cy))
        distance_m = distance_px / 200
        
        if path.clearance < 30:
            urgency = 'urgent'
        elif path.clearance < 50:
            urgency = 'caution'
        else:
            urgency = 'normal'
        
        return NavigationGuidance(direction=direction, distance_to_next_m=distance_m,
                                 confidence=0.8, urgency=urgency)
    
    def visualize_path(self, frame: np.ndarray, path: NavigationPath,
                      occupancy: Optional[np.ndarray] = None) -> np.ndarray:
        vis_frame = frame.copy()
        if occupancy is not None:
            occupancy_colored = (occupancy * 255).astype(np.uint8)
            occupancy_colored = cv2.applyColorMap(occupancy_colored, cv2.COLORMAP_HOT)
            vis_frame = cv2.addWeighted(vis_frame, 0.7, occupancy_colored, 0.3, 0)
        if path is None or not path.waypoints:
            return vis_frame
        points = np.array([(wp.x, wp.y) for wp in path.waypoints], dtype=np.int32)
        cv2.polylines(vis_frame, [points], False, (0, 255, 0), 3)
        for i, wp in enumerate(path.waypoints):
            if i == 0:
                color = (255, 0, 0)
            elif i == len(path.waypoints) - 1:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            cv2.circle(vis_frame, (wp.x, wp.y), 5, color, -1)
        info_text = f"Path: {len(path.waypoints)} waypoints, {path.length:.0f}px"
        cv2.putText(vis_frame, info_text, (10, vis_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return vis_frame
    
    def detect_safe_corridors(self, occupancy: np.ndarray,
                             min_width_px: int = 100) -> List[Tuple[int, int, int]]:
        height, width = occupancy.shape
        corridors = []
        for y in range(0, height, 20):
            row = occupancy[y, :]
            in_corridor = False
            start_x = 0
            for x in range(width):
                if row[x] < self._obstacle_threshold:
                    if not in_corridor:
                        in_corridor = True
                        start_x = x
                else:
                    if in_corridor:
                        corridor_width = x - start_x
                        if corridor_width >= min_width_px:
                            center_x = (start_x + x) // 2
                            corridors.append((center_x, y, corridor_width))
                        in_corridor = False
            if in_corridor:
                corridor_width = width - start_x
                if corridor_width >= min_width_px:
                    center_x = (start_x + width) // 2
                    corridors.append((center_x, y, corridor_width))
        return corridors
    
    def get_stats(self) -> dict:
        success_rate = (self._successful_plans / max(1, self._total_plans)) * 100
        avg_time = self._planning_time_total / max(1, self._total_plans)
        return {
            'total_plans': self._total_plans,
            'successful_plans': self._successful_plans,
            'success_rate': round(success_rate, 1),
            'avg_planning_time_ms': round(avg_time, 2)
        }


# ============================================
# Testing / Demo
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("Path Planner Module Test")
    print("=" * 50)
    
    grid_size = (480, 640)
    occupancy = np.zeros(grid_size, dtype=np.float32)
    occupancy[100:200, 200:300] = 1.0
    occupancy[250:350, 400:500] = 1.0
    occupancy[350:400, 100:550] = 1.0
    
    print("\nOccupancy grid created with 3 obstacles")
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.yaml')
    planner = PathPlanner(config_path=config_path)
    
    start = (320, 450)
    goal = (320, 50)
    
    print(f"\nPlanning path from {start} to {goal}...")
    path = planner.plan_path(occupancy, start, goal)
    
    if path:
        print(f"Path found! Waypoints: {len(path.waypoints)}, Length: {path.length:.1f}px")
        guidance = planner.get_guidance(path, start, frame_width=640)
        print(f"Guidance: {guidance.get_instruction()}")
    else:
        print("No path found")
    
    corridors = planner.detect_safe_corridors(occupancy, min_width_px=80)
    print(f"\nFound {len(corridors)} safe corridors")
    
    stats = planner.get_stats()
    print(f"\nStats: {stats}")
    print("\nTest complete!")
