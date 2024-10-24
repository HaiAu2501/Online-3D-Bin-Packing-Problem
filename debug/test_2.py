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

    # Reset environment
    env.reset()
    env.render()

    # Define actions
    # Action format: (x, y, rotation, buffer_index)

    # Place first item of the buffer at (0, 0) with rotation 0
    action1 = (0, 0, 0, 0)
    observation, reward, done, truncated, info = env.step(action1)
    print("\nAfter placing first box:")
    env.render(verbose=True)
    print(f"Reward: {reward}, Done: {done}, Info: {info}")