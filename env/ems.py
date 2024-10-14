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

    def find_containing_ems(self, box: Tuple):
        """
        Find the EMS that contains the given box.

        :param box: A tuple (x, y, z, w, l, h) representing the box's position and size.
        :return: The EMS tuple that contains the box, or None if not found.
        """
        for ems in self.ems_list:
            if self.ems_contains_box(ems, box):
                return ems
        return None

    def ems_contains_box(self, ems: Tuple, box: Tuple) -> bool:
        """
        Check if the EMS contains the box.

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

    def split_ems(self, ems: Tuple, box: Tuple) -> list:
        """
        Split the EMS into sub-EMSs after placing the box.

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

    def is_valid_ems(self, ems: Tuple) -> bool:
        """
        Check if the EMS has positive volume.

        :param ems: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :return: True if EMS is valid, False otherwise.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = ems
        return (x_max > x_min) and (y_max > y_min) and (z_max > z_min)

    def update_ems_after_placement(self, box: Tuple):
        """
        Update the EMS list after placing a box.

        :param box: Box tuple (x, y, z, w, l, h).
        """
        containing_ems = self.find_containing_ems(box)
        if containing_ems is None:
            print("Error: No EMS contains the placed box.")
            return

        # Split the EMS into sub-EMS
        sub_ems = self.split_ems(containing_ems, box)

        # Remove the original EMS and add the new sub-EMS
        self.ems_list.remove(containing_ems)
        self.ems_list.extend(sub_ems)

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
