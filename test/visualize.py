from ems_generator import generate_ems_from_heightmap

# ems.py (tiếp tục)

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_ems(height_map: np.ndarray, ems_list: List[Tuple[int, int, int, int, int, int]], bin_size: Tuple[int, int, int]):
    """
    Visualize the height map and the generated EMS.

    :param height_map: 2D numpy array of shape (W, L), representing the current height at each (x, y).
    :param ems_list: List of EMS tuples.
    :param bin_size: A tuple (W, L, H), representing the size of the bin.
    """
    W, L, H = bin_size

    fig, ax = plt.subplots(figsize=(6,6))
    # Sử dụng origin='lower' để gốc tọa độ (0,0) nằm ở góc dưới cùng bên trái
    cax = ax.imshow(height_map.T, cmap='viridis', origin='lower', extent=[0, W, 0, L])

    # Thêm colorbar để xem giá trị chiều cao
    fig.colorbar(cax, ax=ax, label='Height')

    for ems in ems_list:
        x_min, y_min, z_min, x_max, y_max, z_max = ems
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min + width/2, y_min + height/2, f'EMS\nz:{z_min}-{z_max}', color='white', ha='center', va='center', fontsize=8)

    plt.title("Height Map with EMS")
    plt.xlabel("X-axis (Horizontal)")
    plt.ylabel("Y-axis (Vertical)")
    plt.show()

if __name__ == "__main__":
    height_map = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ])
    bin_size = (5, 5, 5)
    ems = generate_ems_from_heightmap(height_map, bin_size)
    print("Generated EMS:")
    for e in ems:
        print(e)
    
    # Visualize
    visualize_ems(height_map, ems, bin_size)
