import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_generator_dir = Path(__file__).parent.resolve()

class Generator:
    def __init__(self, n_items: int, bin_size: List[int] = [100, 100, 100]):
        """
        Parameters:
        :param n_items: Number of items to generate for the bin
        :param bin_size: Size of the bin in 3 dimensions
        """
        self.n_items: int = n_items
        self.bin_size: List[int] = bin_size
        # Dictionary to store items for each (seed, version)
        self.versions: Dict[Tuple[int, int], List[Tuple[List[int], List[int]]]] = {}
        self.directory: str = _generator_dir / 'data'
        os.makedirs(self.directory, exist_ok=True)

    def generate(self, seed: int, shuffle: bool = True, verbose: bool = False, detailed: bool = False) -> None:
        """
        Generate random items for the bin with a given seed and create 8 equivalent versions.
        Each version is saved to a separate file.

        :param seed: Seed for random number generation
        :param shuffle: Whether to shuffle items before processing (default: True)
        :param verbose: If True, write full item info (x, y, z, w, l, h); otherwise, only w, l, h (default: False)
        :param detailed: If True, print detailed item info using pandas (default: False)
        """
        def generate_for_bin(bin_origin: List[int]) -> List[Tuple[List[int], List[int]]]:
            """
            Generate random items for a single bin by recursively splitting along the largest dimension.
            """
            items = [(bin_origin, self.bin_size[:])]

            for _ in range(self.n_items - 1):
                if not items:
                    break  # Avoid popping from empty list
                origin, item = items.pop()

                # Choose the dimension with the largest size to split
                max_size = max(item)
                dimension: int = item.index(max_size)
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

        random.seed(seed)
        base_items = generate_for_bin([0, 0, 0])

        # Shuffle or sort items based on `shuffle` parameter
        if shuffle:
            random.shuffle(base_items)
        else:
            base_items.sort(key=lambda item: (item[0][2], item[0][1], item[0][0]))

        # Function to apply reflections based on version
        def apply_reflection(origin: List[int], size: List[int], version: int) -> Tuple[List[int], List[int]]:
            """
            Apply reflection to the item's origin based on the version number.

            :param origin: Original origin [x, y, z]
            :param size: Size [w, l, h]
            :param version: Version number (1-8)
            :return: Reflected origin and size
            """
            # Determine reflection flags using binary representation of version
            reflect_x = (version & 0b100) >> 2
            reflect_y = (version & 0b010) >> 1
            reflect_z = version & 0b001

            x, y, z = origin
            w, l, h = size

            if reflect_x:
                x_new = self.bin_size[0] - x - w
            else:
                x_new = x

            if reflect_y:
                y_new = self.bin_size[1] - y - l
            else:
                y_new = y

            if reflect_z:
                z_new = self.bin_size[2] - z - h
            else:
                z_new = z

            return [x_new, y_new, z_new], [w, l, h]

        # Iterate through versions 1 to 8
        for version in range(1, 9):
            reflected_items = []
            for origin, size in base_items:
                new_origin, new_size = apply_reflection(origin, size, version)
                reflected_items.append((new_origin, new_size))

            if not shuffle:
                reflected_items.sort(key=lambda item: (item[0][2], item[0][1], item[0][0]))

            self.versions[(seed, version)] = reflected_items

            # Define the filename for the current version
            version_filename = f'{self.bin_size[0]}_{self.bin_size[1]}_{self.bin_size[2]}_{self.n_items}_{seed}_{version}.dat'
            version_filepath = os.path.join(self.directory, version_filename)

            # Write data to the version file
            with open(version_filepath, 'w') as file:
                file.write(f'{self.bin_size[0]} {self.bin_size[1]} {self.bin_size[2]}\n')
                for (origin, size) in reflected_items:
                    x, y, z = origin
                    w, l, h = size
                    if verbose:
                        # Full info: x, y, z, w, l, h
                        file.write(f'{x} {y} {z} {w} {l} {h}\n')
                    else:
                        # Only w, l, h
                        file.write(f'{w} {l} {h}\n')

            if detailed:
                # Use pandas to display the data in a tabular format for each version
                data = {
                    'Item': list(range(1, len(reflected_items) + 1)),
                    'X': [item[0][0] for item in reflected_items],
                    'Y': [item[0][1] for item in reflected_items],
                    'Z': [item[0][2] for item in reflected_items],
                    'W': [item[1][0] for item in reflected_items],
                    'L': [item[1][1] for item in reflected_items],
                    'H': [item[1][2] for item in reflected_items],
                }
                df = pd.DataFrame(data)
                print(f'\nSeed {seed}, Version {version}: Generated {self.n_items} items for bin {self.bin_size}')
                print(df.to_string(index=False))

    def visualize(self, seed: int, version: int = 1) -> None:
        """
        Visualize the generated items for a specific seed and version in a 3D plot.
        If version = -1, visualize all 8 versions in a single figure with subplots.

        :param seed: Seed number used during generation
        :param version: Version number to visualize (1-8) or -1 to visualize all versions
        """
        if version != -1 and not (1 <= version <= 8):
            raise ValueError('Version must be between 1 and 8, or -1 to visualize all versions.')

        # Check if the seed exists
        seed_exists = any(key[0] == seed for key in self.versions.keys())
        if not seed_exists:
            raise ValueError(f'Seed {seed} has not been generated yet.')

        if version == -1:
            # Visualize all 8 versions in a single figure with subplots
            fig = plt.figure(figsize=(20, 10))
            for ver in range(1, 9):
                ax = fig.add_subplot(2, 4, ver, projection='3d')
                key = (seed, ver)
                if key not in self.versions:
                    ax.text(0.5, 0.5, 0.5, 'Not Generated', horizontalalignment='center', verticalalignment='center')
                    ax.set_title(f'Version {ver} (Not Generated)')
                    ax.set_xlim([0, self.bin_size[0]])
                    ax.set_ylim([0, self.bin_size[1]])
                    ax.set_zlim([0, self.bin_size[2]])
                    continue

                items = self.versions[key]
                # Create a color palette for items
                colors = plt.cm.viridis([i / len(items) for i in range(len(items))])

                for i, (origin, size) in enumerate(items):
                    x0, y0, z0 = origin
                    dx, dy, dz = size
                    color = colors[i % len(colors)]
                    # Define vertices of the box
                    vertices = [
                        [x0, y0, z0], [x0 + dx, y0, z0], [x0 + dx, y0 + dy, z0], [x0, y0 + dy, z0],
                        [x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz], [x0 + dx, y0 + dy, z0 + dz], [x0, y0 + dy, z0 + dz]
                    ]

                    # Define the six faces of the box
                    faces = [
                        [vertices[j] for j in [0, 1, 5, 4]],
                        [vertices[j] for j in [7, 6, 2, 3]],
                        [vertices[j] for j in [0, 3, 7, 4]],
                        [vertices[j] for j in [1, 2, 6, 5]],
                        [vertices[j] for j in [0, 1, 2, 3]],
                        [vertices[j] for j in [4, 5, 6, 7]]
                    ]

                    # Add the box to the plot
                    ax.add_collection3d(Poly3DCollection(
                        faces,
                        facecolors=color,
                        linewidths=.3,
                        edgecolors='k',
                        alpha=.5,
                        zsort='min'
                    ))

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # Set limits for the axes based on bin size
                ax.set_xlim([0, self.bin_size[0]])
                ax.set_ylim([0, self.bin_size[1]])
                ax.set_zlim([0, self.bin_size[2]])

                ax.set_title(f'Version {ver}')

            # Add a main title
            fig.suptitle(f'3D Bin Packing Visualization for Seed {seed} - All Versions', fontsize=16)

            # Add a legend with information
            info_text = (
                f'Bin size: {self.bin_size}\n'
                f'Number of items: {self.n_items}\n'
                f'Seed: {seed}\n'
                f'All 8 Versions'
            )
            plt.figtext(.5, 0.05, info_text, fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black'))

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.show()

        else:
            # Visualize a specific version
            key = (seed, version)
            if key not in self.versions:
                raise ValueError(f'Version {version} for seed {seed} has not been generated yet.')

            items = self.versions[key]

            if not items:
                raise ValueError('No items to visualize for the specified seed and version.')

            fig = plt.figure(figsize=(9, 5))
            ax = fig.add_subplot(111, projection='3d')

            # Create a color palette for items
            colors = plt.cm.viridis([i / len(items) for i in range(len(items))])

            for i, (origin, size) in enumerate(items):
                x0, y0, z0 = origin
                dx, dy, dz = size
                color = colors[i % len(colors)]
                # Define vertices of the box
                vertices = [
                    [x0, y0, z0], [x0 + dx, y0, z0], [x0 + dx, y0 + dy, z0], [x0, y0 + dy, z0],
                    [x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz], [x0 + dx, y0 + dy, z0 + dz], [x0, y0 + dy, z0 + dz]
                ]

                # Define the six faces of the box
                faces = [
                    [vertices[j] for j in [0, 1, 5, 4]],
                    [vertices[j] for j in [7, 6, 2, 3]],
                    [vertices[j] for j in [0, 3, 7, 4]],
                    [vertices[j] for j in [1, 2, 6, 5]],
                    [vertices[j] for j in [0, 1, 2, 3]],
                    [vertices[j] for j in [4, 5, 6, 7]]
                ]

                # Add the box to the plot
                ax.add_collection3d(Poly3DCollection(
                    faces,
                    facecolors=color,
                    linewidths=.3,
                    edgecolors='k',
                    alpha=.5,
                    zsort='min'
                ))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set limits for the axes based on bin size
            ax.set_xlim([0, self.bin_size[0]])
            ax.set_ylim([0, self.bin_size[1]])
            ax.set_zlim([0, self.bin_size[2]])

            ax.title.set_text(f'3D Bin Packing Visualization - Seed {seed}, Version {version}')

            # Add a legend with information
            info_text = (
                f'Bin size: {self.bin_size}\n'
                f'Number of items: {self.n_items}\n'
                f'Seed: {seed}\n'
                f'Version: {version}'
            )
            plt.figtext(.8, .5, info_text, fontsize=8, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black'))

            plt.show()

    def delete(self) -> None:
        """
        Delete all files generated by the generator across all seeds and versions.
        """
        pattern_prefix = f"{self.bin_size[0]}_{self.bin_size[1]}_{self.bin_size[2]}_{self.n_items}_"
        files_to_delete = [
            f for f in os.listdir(self.directory)
            if f.endswith('.dat') and f.startswith(pattern_prefix)
        ]

        for file in files_to_delete:
            file_path = os.path.join(self.directory, file)
            try:
                os.remove(file_path)
                print(f'Deleted file: {file_path}')
            except Exception as e:
                print(f'Error deleting file {file_path}: {e}')

        # Clear the in-memory versions
        self.versions.clear()
        print('All generated files have been deleted and in-memory data has been cleared.')
