import os
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Generator:
    def __init__(self, n_items: int, bin_size: List[int] = [100, 100, 100]):
        """
        Parameters:
        :param n_items: Number of items to generate for the bin
        :param bin_size: Size of the bin in 3 dimensions
        """
        self.n_items: int = n_items
        self.bin_size: List[int] = bin_size
        self.items: List[Tuple[List[int], List[int]]] = []

    def generate(self, seed: int) -> None:
        """
        Generate random items for the bin with a given seed and write them to a file.
        
        :param seed: Seed for random number generation
        """
        self.seed = seed  # Update seed for this generation run

        # Create a filename based on the new format
        self.filename: str = f'data/{self.bin_size[0]}_{self.bin_size[1]}_{self.bin_size[2]}_{self.n_items}_{self.seed}.dat'

        def generate_for_bin(bin_origin: List[int]) -> List[Tuple[List[int], List[int]]]:
            """
            Generate random items for a single bin by recursively splitting along the largest dimension.
            """
            items = [(bin_origin, self.bin_size[:])]
            bin_volume = self.bin_size[0] * self.bin_size[1] * self.bin_size[2]

            for _ in range(self.n_items - 1):
                (origin, item) = items.pop()
                
                # Choose the dimension with the largest size to split
                dimension: int = item.index(max(item))
                size: int = item[dimension]
                
                if size == 1:
                    items.append((origin, item))
                    continue
                
                # Randomly choose a cut point
                cut_point: int = random.randint(1, size - 1)
                
                # Create 2 new items after cutting
                new_item1: List[int] = item[:]
                new_item2: List[int] = item[:]
                new_item1[dimension] = cut_point
                new_item2[dimension] = size - cut_point
                
                # Create 2 new origins
                new_origin1: List[int] = origin[:]
                new_origin2: List[int] = origin[:]
                new_origin2[dimension] += cut_point
                
                # Add new items to the list
                items.append((new_origin1, new_item1))
                items.append((new_origin2, new_item2))
                items.sort(key=lambda x: x[1][0] * x[1][1] * x[1][2])

            return items

        if self.n_items < 10 or self.n_items > 1000:
            raise ValueError('Number of items must be between 10 and 1000')
        
        random.seed(self.seed)
        self.items = generate_for_bin([0, 0, 0])
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # Write data to file
        with open(self.filename, 'w') as file:
            file.write(f'{self.bin_size[0]} {self.bin_size[1]} {self.bin_size[2]}\n')
            for (_, item) in self.items:
                sample = random.sample(item, 3)
                file.write(f'{sample[0]} {sample[1]} {sample[2]}\n')

    def visualize(self) -> None:
        """
        Visualize the generated items in a 3D plot.
        """
        def plot_box(ax, x0: int, y0: int, z0: int, dx: int, dy: int, dz: int, color) -> None:
            vertices = [
                [x0, y0, z0], [x0 + dx, y0, z0], [x0 + dx, y0 + dy, z0], [x0, y0 + dy, z0],
                [x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz], [x0 + dx, y0 + dy, z0 + dz], [x0, y0 + dy, z0 + dz]
            ]
            
            faces = [
                [vertices[j] for j in [0, 1, 5, 4]],
                [vertices[j] for j in [7, 6, 2, 3]],
                [vertices[j] for j in [0, 3, 7, 4]],
                [vertices[j] for j in [1, 2, 6, 5]],
                [vertices[j] for j in [0, 1, 2, 3]],
                [vertices[j] for j in [4, 5, 6, 7]]
            ]
            
            ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=.3, edgecolors='k', alpha=.5, zsort='min'))

        if not self.items:
            raise ValueError('Items have not been generated yet')

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Create a color palette for items
        colors = plt.cm.viridis([i / len(self.items) for i in range(len(self.items))])

        for i, (origin, item) in enumerate(self.items):
            x0, y0, z0 = origin
            dx, dy, dz = item
            color = colors[i % len(colors)]
            plot_box(ax, x0, y0, z0, dx, dy, dz, color)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set limits for the axes
        ax.set_xlim([0, self.bin_size[0]])
        ax.set_ylim([0, self.bin_size[1]])
        ax.set_zlim([0, self.bin_size[2]])

        ax.title.set_text(f'3D Bin Packing Visualization')
        
        # Add a legend with information
        info_text = (
            f'Bin size: {self.bin_size}\n'
            f'Number of items: {self.n_items}\n'
        )
        plt.figtext(.8, .5, info_text, fontsize=8, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black'))

        plt.show()

    def delete(self) -> None:
        """
        Delete all files generated with the pattern {W}_{L}_{H}_{num_items}_{seed}.dat.
        """
        base_path = os.path.dirname(self.filename)
        pattern = f"{self.bin_size[0]}_{self.bin_size[1]}_{self.bin_size[2]}_{self.n_items}_*.dat"
        files_to_delete = [f for f in os.listdir(base_path) if f.endswith('.dat') and f.startswith(pattern.split('*')[0])]
        
        for file in files_to_delete:
            os.remove(os.path.join(base_path, file))
