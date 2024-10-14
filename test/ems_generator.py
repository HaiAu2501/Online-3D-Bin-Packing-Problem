import numpy as np

class EMSManager:
    """
    Class to manage Empty Maximal Spaces (EMS) in the bin.
    """
    def __init__(self, bin_size: tuple):
        """
        Initialize the EMSManager with the entire bin as the initial EMS.

        :param bin_size: A tuple (W, L, H) representing the size of the bin.
        """
        self.W, self.L, self.H = bin_size
        # Each EMS is represented as (x_min, y_min, z_min, x_max, y_max, z_max)
        self.ems_list = [(0, 0, 0, self.W, self.L, self.H)]

    def find_containing_ems(self, box: tuple):
        """
        Find the EMS that contains the given box.

        :param box: A tuple (x, y, z, w, l, h) representing the box's position and size.
        :return: The EMS tuple that contains the box, or None if not found.
        """
        box_x, box_y, box_z, box_w, box_l, box_h = box
        for ems in self.ems_list:
            if self.ems_contains_box(ems, box):
                return ems
        return None

    def ems_contains_box(self, ems: tuple, box: tuple) -> bool:
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

    def split_ems(self, ems: tuple, box: tuple) -> list:
        """
        Split the EMS into sub-EMSs after placing the box.

        :param ems: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :param box: Box tuple (x, y, z, w, l, h).
        :return: A list of new EMS tuples.
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

        # 4. EMS to the left of the box (if any)
        if box_x > x_min:
            sub = (x_min, y_min, z_min, box_x, y_max, z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        # 5. EMS behind the box (if any)
        if box_y > y_min:
            sub = (x_min, y_min, z_min, x_max, box_y, z_max)
            if self.is_valid_ems(sub):
                new_ems.append(sub)

        return new_ems

    def is_valid_ems(self, ems: tuple) -> bool:
        """
        Check if the EMS has positive volume.

        :param ems: EMS tuple (x_min, y_min, z_min, x_max, y_max, z_max).
        :return: True if EMS is valid, False otherwise.
        """
        x_min, y_min, z_min, x_max, y_max, z_max = ems
        return (x_max > x_min) and (y_max > y_min) and (z_max > z_min)

    def update_ems_after_placement(self, box: tuple):
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

    def print_ems_list(self):
        """
        Print the current list of EMS.
        """
        print("\nCurrent EMS List:")
        for idx, ems in enumerate(self.ems_list):
            print(f"EMS {idx + 1}: (x_min={ems[0]}, y_min={ems[1]}, z_min={ems[2]}, "
                  f"x_max={ems[3]}, y_max={ems[4]}, z_max={ems[5]})")

class Box:
    """
    Class to represent a box with position and size.
    """
    def __init__(self, x: int, y: int, z: int, w: int, l: int, h: int):
        """
        Initialize the Box.

        :param x: X-coordinate (horizontal) of the box's bottom-left-front corner.
        :param y: Y-coordinate (vertical) of the box's bottom-left-front corner.
        :param z: Z-coordinate (height) of the box's bottom-left-front corner.
        :param w: Width of the box (along X-axis).
        :param l: Length of the box (along Y-axis).
        :param h: Height of the box (along Z-axis).
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.l = l
        self.h = h

    def as_tuple(self) -> tuple:
        """
        Return the box parameters as a tuple.

        :return: Tuple (x, y, z, w, l, h).
        """
        return (self.x, self.y, self.z, self.w, self.l, self.h)

def print_height_map(height_map: np.ndarray):
    """
    Print the height map in a readable format.

    :param height_map: 2D numpy array representing the height map.
    """
    print("\nCurrent Height Map (Z-values):")
    # Print Y from top to bottom to match the coordinate system
    for y in reversed(range(height_map.shape[1])):
        row = ""
        for x in range(height_map.shape[0]):
            row += f"{height_map[x][y]:2} "
        print(row)

def get_user_input():
    """
    Get box placement input from the user.

    :return: Tuple (x, y, z, w, l, h) if valid, else None.
    """
    try:
        print("\nEnter box placement details:")
        x, y, z = map(int, input("The corner where the box is placed (X Y Z): ").split())
        w, l, h = map(int, input("The size of the box (W L H): ").split())
        return (x, y, z, w, l, h)
    except ValueError:
        print("Invalid input. Please enter integer values.")
        return None

def main():
    print("=== 3D Bin Packing with EMS Management ===")

    # Step 1: Input bin size
    try:
        W, L, H = map(int, input("\nEnter the dimensions of the bin (W L H): ").split())
        if W <= 0 or L <= 0 or H <= 0:
            print("Bin dimensions must be positive integers.")
            return
    except ValueError:
        print("Invalid input. Please enter integer values for bin dimensions.")
        return

    bin_size = (W, L, H)
    height_map = np.zeros((W, L), dtype=int)

    # Initialize EMS Manager
    ems_manager = EMSManager(bin_size)

    # Initial state
    print_height_map(height_map)
    ems_manager.print_ems_list()

    while True:
        # Step 2: Get box placement from user
        box_input = get_user_input()
        if box_input is None:
            continue  # Invalid input, retry

        box = Box(*box_input)
        box_tuple = box.as_tuple()

        # Step 3: Validate box placement
        x, y, z, w, l, h = box_tuple

        # Check if box is within bin boundaries
        if (x < 0 or y < 0 or z < 0 or
            x + w > W or y + l > L or z + h > H):
            print("Error: Box placement is out of bin boundaries.")
            continue

        # Check if box overlaps with existing boxes based on height_map
        overlap = False
        for xi in range(x, x + w):
            for yi in range(y, y + l):
                if height_map[xi][yi] > z:
                    overlap = True
                    break
            if overlap:
                break
        if overlap:
            print("Error: Box placement overlaps with existing boxes.")
            continue

        # Check if the placement is within an EMS
        if ems_manager.find_containing_ems(box_tuple) is None:
            print("Error: No EMS contains the placed box.")
            continue

        # Step 4: Place the box
        # Update height_map
        for xi in range(x, x + w):
            for yi in range(y, y + l):
                height_map[xi][yi] = max(height_map[xi][yi], z + h)

        # Update EMS list
        ems_manager.update_ems_after_placement(box_tuple)

        # Step 5: Print updated height_map and EMS list
        print_height_map(height_map)
        ems_manager.print_ems_list()

        # Optional: Check if bin is full
        # This can be implemented based on specific criteria
        # For simplicity, we'll continue until the user decides to stop

        # Ask user if they want to continue
        cont = input("\nDo you want to place another box? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting the program.")
            break

if __name__ == "__main__":
    main()
