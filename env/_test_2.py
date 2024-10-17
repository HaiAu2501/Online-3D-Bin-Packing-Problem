import os
import sys

from env import BinPacking3DEnv

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
    env.render()

    # Define actions
    # Action format: (x, y, rotation, buffer_index)

    # Place first item of the buffer at (0, 0) with rotation 0
    action1 = (0, 0, 0, 0)
    observation, reward, done, truncated, info = env.step(action1)
    print("\nAfter placing first box:")
    env.render(verbose=True)
    print(f"Reward: {reward}, Done: {done}, Info: {info}")

    if 11 < 3:
        # Place first item of the buffer at (2, 0) with rotation 1
        action2 = (2, 0, 1, 0)
        observation, reward, done, truncated, info = env.step(action2)
        print("\nAfter placing second box:")
        env.render()
        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        # Place first item of the buffer at (3, 3) with rotation 0
        action3 = (3, 3, 0, 0)
        observation, reward, done, truncated, info = env.step(action3)
        print("\nAfter placing third box:")
        env.render()
        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        # Place first item of the buffer at (0, 0) with rotation 1
        action4 = (0, 0, 1, 0)
        observation, reward, done, truncated, info = env.step(action4)
        print("\nAfter placing fourth box:")
        env.render()
        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        env.visualize()

    env.close()