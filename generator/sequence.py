from collections import deque
from typing import List, Tuple
import itertools

"""
Enhanced for Online 3D Bin Packing:
Find all sequences where items are removed based on the smallest z, then y, then x within the first k items.
The removal sequence must be sorted lexicographically by (z, y, x).
"""

def find_valid_sequences_zyx(items: List[Tuple[int, int, int, int, int, int]], k: int) -> List[List[Tuple[int, int, int, int, int, int]]]:
    """
    Finds all valid sequences of items where the removal order is sorted by (z, y, x).
    
    Args:
        items: List of items, each represented as a tuple (x, y, z, w, l, h).
        k: Size of the buffer.
    
    Returns:
        A list of valid sequences, where each sequence is a list of items.
    """
    n = len(items)
    total = n
    # Sort desired removals lexicographically by (z, y, x)
    desired_removals_sorted = sorted(items, key=lambda item: (item[2], item[1], item[0]))
    desired_removals = [ (item[2], item[1], item[0]) for item in desired_removals_sorted ]
    
    results = []
    used = [False] * n
    current_sequence = []
    
    def backtrack(sequence: List[int]):
        if len(sequence) == total:
            # After building the full sequence, check the removal process
            if simulate_removal([items[i] for i in sequence], desired_removals):
                # Convert indices back to items
                results.append([items[i] for i in sequence])
            return
        
        for i in range(n):
            if not used[i]:
                # Choose this item
                used[i] = True
                sequence.append(i)
                # Optional Pruning: Early validation can be added here
                backtrack(sequence)
                # Backtrack
                sequence.pop()
                used[i] = False
    
    def simulate_removal(seq: List[Tuple[int, int, int, int, int, int]], desired: List[Tuple[int, int, int]]) -> bool:
        """
        Simulates the removal process and verifies if the removal order matches the desired sequence.
        
        Args:
            seq: The sequence of items.
            desired: The desired removal sequence sorted by (z, y, x).
        
        Returns:
            True if the removal sequence matches the desired sequence, False otherwise.
        """
        seq = deque(seq)
        desired = deque(desired)
        
        while seq and desired:
            # Determine the window of the first k items
            window = list(seq)[:k] if len(seq) >= k else list(seq)
            if not window:
                break
            
            # Find the item to remove based on (z, y, x)
            # Sort the window lexicographically and pick the first item
            min_item = min(window, key=lambda item: (item[2], item[1], item[0]))
            min_index = window.index(min_item)
            
            # Compare with the expected removal
            expected_zyx = desired.popleft()
            actual_zyx = (min_item[2], min_item[1], min_item[0])
            
            if actual_zyx != expected_zyx:
                return False  # Mismatch in removal order
            
            # Remove the item from the sequence
            del seq[min_index]
        
        # After processing, ensure all desired removals are done
        return not desired and not seq
    
    # Start backtracking
    backtrack([])
    return results

def read_items_from_file(filename: str) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Reads items from a .dat file.
    
    Args:
        filename: Path to the .dat file.
    
    Returns:
        A list of items, each represented as a tuple (x, y, z, w, l, h).
    """
    items = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Assuming the first line is some metadata (e.g., bin size), skip it
        for line in lines[1:]:
            parts = list(map(int, line.strip().split()))
            if len(parts) != 6:
                continue  # Skip malformed lines
            items.append(tuple(parts))
    return items

if __name__ == "__main__":
    filename = 'generator/data/10_10_10_10_0_1.dat'
    items = read_items_from_file(filename)
    k = 2  # Example buffer size
    valid_sequences = find_valid_sequences_zyx(items, k)
    print(f"Number of valid sequences: {len(valid_sequences)}")
    # Optionally, print some or all valid sequences
    for idx, seq in enumerate(valid_sequences, 1):
        if idx > 5:
            break
        print(f"Sequence {idx}:")
        for item in seq:
            print(item)
        print("---")
