import math
from typing import List, Tuple, Dict

class Tracker:
    def __init__(self):
        # Dictionary to store the center positions of the objects and their IDs
        self.center_points: Dict[int, Tuple[int, int]] = {}
        # Counter for generating unique IDs
        self.id_count: int = 0

    def update(self, objects_rect: List[Tuple[int, int, int, int]]) -> List[List[int]]:
        # List to store objects with their bounding boxes and IDs
        objects_bbs_ids: List[List[int]] = []

        # Iterate through each object bounding box
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2  # Center x-coordinate
            cy = (y + y + h) // 2  # Center y-coordinate

            # Flag to check if the object was already detected
            same_object_detected = False

            # Check if this object is already being tracked
            for obj_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])  # Calculate Euclidean distance

                if dist < 35:  # If the object is within the threshold distance
                    # Update the position of the existing object
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    same_object_detected = True
                    break

            # If the object is new, assign a new ID
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Create a new dictionary to remove IDs of objects that were not detected in the current frame
        new_center_points: Dict[int, Tuple[int, int]] = {}
        for _, _, _, _, obj_id in objects_bbs_ids:
            new_center_points[obj_id] = self.center_points[obj_id]

        # Update the dictionary with current IDs
        self.center_points = new_center_points

        # Debug information
        print("Center Points:", self.center_points)
        print("Tracked Objects:", objects_bbs_ids)

        return objects_bbs_ids
