from dataclasses import dataclass
from typing import List
import random

@dataclass
class Box:
    x: int
    y: int
    z: int
    width: int
    height: int
    depth: int

def recursive_partition(box: Box, min_size: int) -> List[Box]:
    """
    Chia hộp hiện tại thành các hộp nhỏ hơn đệ quy với kích thước số nguyên.
    
    :param box: Hộp hiện tại cần chia.
    :param min_size: Kích thước tối thiểu của hộp nhỏ (số nguyên).
    :return: Danh sách các hộp nhỏ sau khi chia.
    """
    result = []
    
    def partition(current_box: Box):
        # Kiểm tra xem có thể chia tiếp hộp hiện tại hay không
        can_split_x = current_box.width >= 2 * min_size
        can_split_y = current_box.depth >= 2 * min_size
        can_split_z = current_box.height >= 2 * min_size
        
        # Tạo danh sách các trục có thể chia
        axes = []
        if can_split_x:
            axes.append('x')
        if can_split_y:
            axes.append('y')
        if can_split_z:
            axes.append('z')
        
        # Nếu không thể chia tiếp, thêm hộp vào kết quả
        if not axes:
            result.append(current_box)
            return
        
        # Chọn ngẫu nhiên một trục từ các trục có thể chia
        axis = random.choice(axes)
        
        # Chia hộp theo trục được chọn
        if axis == 'x':
            split = random.randint(min_size, current_box.width - min_size)
            box1 = Box(current_box.x, current_box.y, current_box.z,
                       split, current_box.height, current_box.depth)
            box2 = Box(current_box.x + split, current_box.y, current_box.z,
                       current_box.width - split, current_box.height, current_box.depth)
        elif axis == 'y':
            split = random.randint(min_size, current_box.depth - min_size)
            box1 = Box(current_box.x, current_box.y, current_box.z,
                       current_box.width, current_box.height, split)
            box2 = Box(current_box.x, current_box.y + split, current_box.z,
                       current_box.width, current_box.height, current_box.depth - split)
        else:  # axis == 'z'
            split = random.randint(min_size, current_box.height - min_size)
            box1 = Box(current_box.x, current_box.y, current_box.z,
                       current_box.width, split, current_box.depth)
            box2 = Box(current_box.x, current_box.y, current_box.z + split,
                       current_box.width, current_box.height - split, current_box.depth)
        
        # Đệ quy chia các hộp con
        partition(box1)
        partition(box2)
    
    partition(box)
    return result

def sort_boxes_by_z(boxes: List[Box]) -> List[Box]:
    return sorted(boxes, key=lambda box: box.z)

def check_overlap(box1: Box, box2: Box) -> bool:
    return not (box1.x + box1.width <= box2.x or
                box2.x + box2.width <= box1.x or
                box1.y + box1.depth <= box2.y or
                box2.y + box2.depth <= box1.y or
                box1.z + box1.height <= box2.z or
                box2.z + box2.height <= box1.z)

def verify_no_overlap(boxes: List[Box]) -> bool:
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if check_overlap(boxes[i], boxes[j]):
                print(f"Chồng lấp giữa Hộp {i + 1} và Hộp {j + 1}")
                return False
    return True

def main():
    random.seed(42)  # Đặt seed để kết quả lặp lại được (tùy chọn)
    
    container_size = (10, 10, 10)  # Kích thước hộp lớn (x, y, z)
    min_box_size = 3  # Kích thước tối thiểu của hộp nhỏ (số nguyên)
    initial_box = Box(0, 0, 0, *container_size)
    
    # Chia hộp lớn thành các hộp nhỏ
    boxes = recursive_partition(initial_box, min_box_size)
    sorted_boxes = sort_boxes_by_z(boxes)
    
    # Tính tổng thể tích các hộp nhỏ để xác minh
    total_volume = sum(box.width * box.height * box.depth for box in sorted_boxes)
    container_volume = container_size[0] * container_size[1] * container_size[2]
    
    print(f"Tổng số hộp nhỏ: {len(sorted_boxes)}")
    print(f"Tổng thể tích các hộp nhỏ: {total_volume}")
    print(f"Thể tích hộp lớn: {container_volume}\n")
    
    if total_volume != container_volume:
        print("Cảnh báo: Thể tích các hộp nhỏ không bằng thể tích hộp lớn!")
    else:
        print("Thể tích các hộp nhỏ bằng thể tích hộp lớn. Phủ kín thành công!\n")
    
    # Kiểm tra sự chồng lấp
    if verify_no_overlap(sorted_boxes):
        print("Không có sự chồng lấp giữa các hộp nhỏ.\n")
    else:
        print("Có sự chồng lấp giữa các hộp nhỏ!\n")
    
    # Hiển thị kết quả
    for idx, box in enumerate(sorted_boxes):
        print(f"Hộp {idx + 1}:")
        print(f"  Tọa độ góc dưới: ({box.x}, {box.y}, {box.z})")
        print(f"  Kích thước: {box.width} x {box.depth} x {box.height}\n")
        with open("boxes.dat", "a") as f:
            f.write(f"{box.width} {box.depth} {box.height}\n")

if __name__ == "__main__":
    main()
