import os
import sys

sys.path.append(os.getcwd())

from env.env import BinPacking3DEnv

if __name__ == "__main__":
    # Define bin size and items
    bin_size = (5, 5, 5)
    items = [(2, 3, 1), (3, 3, 5), (2, 2, 3), (2, 2, 1)]  # Example items

    # Initialize environment
    env = BinPacking3DEnv(
        bin_size=bin_size,
        items=items,
        buffer_size=2,
        num_rotations=2
    )

    env.reset()

    print("Initial state:")
    env.render(verbose=True)

    # Define actions
    # Action format: (x, y, rotation, buffer_index)

    # Place first item of the buffer at (0, 0) with rotation 0
    action_1 = (0, 0, 0, 0)
    print(f"Performing action {action_1}...")
    print(f"Place the {env.buffer[0]} item at (0, 0) with rotation 0")
    print()
    env.step(action_1)
    env.render(verbose=True)

    # Place first item of the buffer at (2, 0) with rotation 1
    action_2 = (2, 0, 1, 0)
    print(f"Performing action {action_2}...")
    print(f"Place the {env.buffer[0]} item at (2, 0) with rotation 1")
    print()
    env.step(action_2)
    env.render(verbose=True)

    # Place second item of the buffer at (0, 3) with rotation 0
    action_3 = (0, 3, 0, 1)
    print(f"Performing action {action_3}...")
    print(f"Place the {env.buffer[1]} item at (0, 3) with rotation 0")
    print()
    env.step(action_3)
    env.render(verbose=True)

    # Place first item of the buffer at (0, 0) with rotation 0
    action_4 = (0, 0, 0, 0)
    print(f"Performing action {action_4}...")
    print(f"Place the {env.buffer[0]} item at (0, 0) with rotation 0")
    print()
    env.step(action_4)
    env.render(verbose=True)
    
    env.visualize()