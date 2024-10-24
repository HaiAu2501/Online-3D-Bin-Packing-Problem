# ems.py 

import numpy as np
from typing import List, Tuple

class EMSManager:
    """
    Class to manage Empty Maximal Spaces (EMS) in the bin.
    """
    def __init__(self, bin_size: Tuple[int, int, int]):
        """
        Initialize the EMSManager with the entire bin as the initial EMS.

        :param bin_size: A tuple (W, L, H) representing the size of the bin.
        """
        self.W, self.L, self.H = bin_size
        # Each EMS is represented as (x_min, y_min, z_min, x_max, y_max, z_max)
        self.ems_list = [(0, 0, 0, self.W, self.L, self.H)]

    def reset(self):
        """
        Reset the EMSManager to the initial state.
        """
        self.ems_list = [(0, 0, 0, self.W, self.L, self.H)]

    def find_containing_ems(self, box: Tuple[int, int, int, int, int, int]) -> Tuple[int, int, int, int, int, int]:
        """
        Find the EMS that fully contains the given box.

        :param box: A tuple (x, y, z, w, l, h) representing the box's position and rotated size.
        :return: The EMS tuple that contains the box, or None if not found.
        """
        for ems in self.ems_list:
            if self.ems_contains_box(ems, box):
                return ems
        return None

    def ems_contains_box(self, ems: Tuple[int, int, int, int, int, int], box: Tuple[int, int, int, int, int, int]) -> bool:
        """
        Check if the EMS fully contains the box.

        :param ems: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :param box: Box tuple (x, y, z, w, l, h).
        :return: True if EMS contains the box, False otherwise.
        """
        box_x, box_y, box_z, box_w, box_l, box_h = box
        return (ems[0] <= box_x and
                ems[1] <= box_y and
                ems[2] <= box_z and
                (box_x + box_w) <= ems[3] and
                (box_y + box_l) <= ems[4] and
                (box_z + box_h) <= ems[5])

    def ems_intersects_box(self, ems: Tuple[int, int, int, int, int, int], box: Tuple[int, int, int, int, int, int]) -> bool:
        """
        Check if the EMS intersects with the box.

        :param ems: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :param box: Box tuple (x, y, z, w, l, h).
        :return: True if EMS intersects with the box, False otherwise.
        """
        ems_x_min, ems_y_min, ems_z_min, ems_x_max, ems_y_max, ems_z_max = ems
        box_x, box_y, box_z, box_w, box_l, box_h = box
        box_x_max = box_x + box_w
        box_y_max = box_y + box_l
        box_z_max = box_z + box_h

        # Check for overlap in all three dimensions
        overlap = not (box_x_max <= ems_x_min or box_x >= ems_x_max or
                       box_y_max <= ems_y_min or box_y >= ems_y_max or
                       box_z_max <= ems_z_min or box_z >= ems_z_max)
        return overlap

    def split_ems(self, ems: Tuple[int, int, int, int, int, int], box: Tuple[int, int, int, int, int, int]) -> List[Tuple[int, int, int, int, int, int]]:
        """
        Split the EMS into sub-EMSs after placing the box.
        This method assumes that the EMS fully contains the box.

        :param ems: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :param box: Box tuple (x, y, z, w, l, h).
        :return: A list of new EMS Tuples.
        """
        new_ems = []
        x_min, y_min, z_min, x_max, y_max, z_max = ems
        box_x, box_y, box_z, box_w, box_l, box_h = box
        box_x_max = box_x + box_w
        box_y_max = box_y + box_l
        box_z_max = box_z + box_h

        # 1. EMS to the right of the box
        if box_x_max < x_max:
            sub = (box_x_max, y_min, z_min, x_max, y_max, z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 2. EMS in front of the box
        if box_y_max < y_max:
            sub = (x_min, box_y_max, z_min, x_max, y_max, z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 3. EMS above the box
        if box_z_max < z_max:
            sub = (x_min, y_min, box_z_max, x_max, y_max, z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 4. EMS to the left of the box
        if box_x > x_min:
            sub = (x_min, y_min, z_min, box_x, y_max, z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 5. EMS behind the box
        if box_y > y_min:
            sub = (x_min, y_min, z_min, x_max, box_y, z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        return new_ems

    def subtract_box_from_ems(self, ems: Tuple[int, int, int, int, int, int], box: Tuple[int, int, int, int, int, int]) -> List[Tuple[int, int, int, int, int, int]]:
        """
        Subtract the overlapping region of the box from the EMS.
        This method handles cases where the EMS and box partially overlap.

        :param ems: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :param box: Box tuple (x, y, z, w, l, h).
        :return: A list of new EMS Tuples after subtraction.
        """
        new_ems = []
        ems_x_min, ems_y_min, ems_z_min, ems_x_max, ems_y_max, ems_z_max = ems
        box_x, box_y, box_z, box_w, box_l, box_h = box
        box_x_max = box_x + box_w
        box_y_max = box_y + box_l
        box_z_max = box_z + box_h

        # Determine the overlapping region
        overlap_x_min = max(ems_x_min, box_x)
        overlap_y_min = max(ems_y_min, box_y)
        overlap_z_min = max(ems_z_min, box_z)
        overlap_x_max = min(ems_x_max, box_x_max)
        overlap_y_max = min(ems_y_max, box_y_max)
        overlap_z_max = min(ems_z_max, box_z_max)

        # Create EMSs for regions not overlapped by the box
        # 1. EMS to the left of the overlapping region
        if ems_x_min < overlap_x_min:
            sub = (ems_x_min, ems_y_min, ems_z_min, overlap_x_min, ems_y_max, ems_z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 2. EMS to the right of the overlapping region
        if overlap_x_max < ems_x_max:
            sub = (overlap_x_max, ems_y_min, ems_z_min, ems_x_max, ems_y_max, ems_z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 3. EMS in front of the overlapping region
        if ems_y_min < overlap_y_min:
            sub = (overlap_x_min, ems_y_min, ems_z_min, overlap_x_max, overlap_y_min, ems_z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 4. EMS behind the overlapping region
        if overlap_y_max < ems_y_max:
            sub = (overlap_x_min, overlap_y_max, ems_z_min, overlap_x_max, ems_y_max, ems_z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 5. EMS below the overlapping region
        if ems_z_min < overlap_z_min:
            sub = (overlap_x_min, overlap_y_min, ems_z_min, overlap_x_max, overlap_y_max, overlap_z_min)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 6. EMS above the overlapping region
        if overlap_z_max < ems_z_max:
            sub = (overlap_x_min, overlap_y_min, overlap_z_max, overlap_x_max, overlap_y_max, ems_z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        return new_ems

    def is_valid_ems(self, ems: Tuple[int, int, int, int, int, int]) -> bool:
        """
        Check if the EMS has positive volume.

        :param ems: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :return: True if EMS is valid, False otherwise.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = ems
        return (x_max > x_min) and (y_max > y_min) and (z_max > z_min)

    def remove_contained_ems(self):
        """
        Remove EMS that are completely contained within another EMS.
        """
        to_remove = set()
        ems_count = len(self.ems_list)
        for i in range(ems_count):
            if i in to_remove:
                continue
            for j in range(ems_count):
                if i == j or j in to_remove:
                    continue
                if self.is_ems_contained(self.ems_list[i], self.ems_list[j]):
                    to_remove.add(i)
                    break
        # Remove in reverse order to avoid indexing issues
        self.ems_list = [ems for idx, ems in enumerate(self.ems_list) if idx not in to_remove]

    def is_ems_contained(self, ems1: Tuple[int, int, int, int, int, int], ems2: Tuple[int, int, int, int, int, int]) -> bool:
        """
        Check if ems1 is completely contained within ems2.

        :param ems1: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :param ems2: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :return: True if ems1 is contained within ems2, False otherwise.
        """
        return (ems2[0] <= ems1[0] and
                ems2[1] <= ems1[1] and
                ems2[2] <= ems1[2] and
                ems1[3] <= ems2[3] and
                ems1[4] <= ems2[4] and
                ems1[5] <= ems2[5])

    def update_ems_after_placement(self, box: Tuple[int, int, int, int, int, int]):
        """
        Update the EMS list after placing a box.

        :param box: Box tuple (x, y, z, w, l, h).
        """
        # Find all EMSs that intersect with the placed box
        overlapping_ems = [ems for ems in self.ems_list if self.ems_intersects_box(ems, box)]

        if not overlapping_ems:
            print("Error: No EMS intersects with the placed box.")
            return

        new_ems = []  # List to hold new EMSs generated after splitting

        for ems in overlapping_ems:
            if self.ems_contains_box(ems, box):
                # Case 1: EMS fully contains the box
                # Split the EMS into sub-EMSs excluding the box
                splitted = self.split_ems(ems, box)
                new_ems.extend(splitted)
                self.ems_list.remove(ems)
            else:
                # Case 2: EMS partially overlaps with the box
                # Subtract the overlapping region from the EMS
                splitted = self.subtract_box_from_ems(ems, box)
                new_ems.extend(splitted)
                self.ems_list.remove(ems)

        # Add all new EMSs generated from splitting
        self.ems_list.extend(new_ems)

        # Remove any EMSs that are completely contained within another EMS
        self.remove_contained_ems()

    def get_ems_list(self) -> List[Tuple[int, int, int, int, int, int]]:
        """
        Get the current list of EMS.

        :return: List of EMS tuples.
        """
        return self.ems_list.copy()

    def print_ems_list(self):
        """
        Print the current list of EMS.
        """
        print("\nCurrent EMS List:")
        for idx, ems in enumerate(self.ems_list):
            print(f"EMS {idx + 1}: (x_min={ems[0]}, y_min={ems[1]}, z_min={ems[2]}, "
                  f"x_max={ems[3]}, y_max={ems[4]}, z_max={ems[5]})")

    def clone(self):
        """
        Create a deep copy of the EMSManager object.

        :return: A deep copy of the EMSManager object.
        """
        cloned_manager = EMSManager(bin_size=(self.W, self.L, self.H))
        cloned_manager.ems_list = self.ems_list.copy()

        return cloned_manager