
import os
import sys
import numpy as np

sys.path.append(os.getcwd())

from env.ems import EMSManager

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
    for x in range(height_map.shape[0]):
        for y in range(height_map.shape[1]):
            print(height_map[x][y], end=" ")
        print()

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
