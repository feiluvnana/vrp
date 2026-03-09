"""
Module này triển khai thuật toán Tìm kiếm Lân cận Lớn Thích nghi (Adaptive Large Neighborhood Search - ALNS).
ALNS là một khung thuật toán metaheuristic mạnh mẽ, hoạt động bằng cách liên tục 'phá hủy' và 'sửa chữa'
lời giải để tìm ra kết quả tối ưu hơn theo thời gian.
"""
from vrp_core import Request, Trip, VRPState, DATA, LW, ALPHA, GAMMA, print_solution
from greedy_solver import solve_greedy
import numpy as np
import random
import math
from alns import ALNS
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

# --- TOÁN TỬ PHÁ HỦY (DESTROY OPERATORS) ---
# Các toán tử này loại bỏ một phần của lời giải hiện tại.

def random_removal(state: VRPState, rnd_state):
    """
    Toán tử Phá hủy Ngẫu nhiên (Random Removal):
    Loại bỏ ngẫu nhiên một tỷ lệ đơn hàng (k%) khỏi các chuyến đi hiện tại.
    Mục đích: Giúp thuật toán khám phá các vùng không gian lời giải mới và thoát khỏi tối ưu cục bộ.
    """
    new_state = state.copy()
    all_served = [] # Danh sách các cặp (chuyến đi, đơn hàng) đang có trong lời giải
    
    # Duyệt qua tất cả các chuyến đi của xe tải và drone để thu thập đơn hàng đã phục vụ
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            all_served.extend([(t, r) for r in t.stops])
    
    if not all_served:
        return new_state
        
    # Tính số lượng đơn hàng cần loại bỏ (ví dụ: 30% tổng số đơn hàng)
    k = max(1, int(len(new_state.requests) * 0.3))
    num_to_remove = min(len(all_served), k)
    
    # Chọn ngẫu nhiên các chỉ số để loại bỏ
    to_remove_indices = rnd_state.choice(len(all_served), num_to_remove, replace=False)
    
    for idx in to_remove_indices:
        trip, req = all_served[idx]
        trip.stops.remove(req)   # Xóa khỏi chuyến đi
        new_state.unserved.add(req) # Đưa vào danh sách chưa phục vụ
    
    # Loại bỏ các chuyến đi bị trống sau khi xóa đơn hàng
    new_state.truck_trips = [[t for t in v if t.stops] for v in new_state.truck_trips]
    new_state.drone_trips = [[t for t in v if t.stops] for v in new_state.drone_trips]
    return new_state

def shaw_removal(state: VRPState, rnd_state):
    """
    Toán tử Phá hủy Shaw (Shaw Removal):
    Loại bỏ các đơn hàng có tính 'liên quan' cao với nhau (gần nhau về tọa độ hoặc thời gian).
    Mục đích: Gom nhóm các đơn hàng liên quan để toán tử sửa chữa có cơ hội sắp xếp lại chúng hiệu quả hơn.
    """
    new_state = state.copy()
    all_served_reqs = []
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            all_served_reqs.extend([r for r in t.stops])
            
    if not all_served_reqs: return new_state
    
    # Chọn ngẫu nhiên một đơn hàng làm 'trọng tâm' (pivot)
    pivot = rnd_state.choice(all_served_reqs)
    
    # Hàm tính toán độ liên quan (Relatedness): giá trị thấp nghĩa là càng liên quan
    def relatedness(r1, r2):
        dist = math.sqrt((r1.x-r2.x)**2 + (r1.y-r2.y)**2)
        time_diff = abs(r1.r_i - r2.r_i)
        return dist + 0.1 * time_diff # Kết hợp khoảng cách địa lý và thời gian hàng sẵn sàng
        
    # Sắp xếp các đơn hàng theo mức độ liên quan tới pivot
    all_served_reqs.sort(key=lambda r: relatedness(pivot, r))
    
    # Loại bỏ k% đơn hàng đứng đầu danh sách liên quan
    num_to_remove = max(1, int(len(new_state.requests) * 0.4))
    to_remove = all_served_reqs[:num_to_remove]
    
    for req in to_remove:
        for v_trips in new_state.truck_trips + new_state.drone_trips:
            for t in v_trips:
                if req in t.stops:
                    t.stops.remove(req)
                    new_state.unserved.add(req)
                    
    # Vệ sinh lại các chuyến đi trống
    new_state.truck_trips = [[t for t in v if t.stops] for v in new_state.truck_trips]
    new_state.drone_trips = [[t for t in v if t.stops] for v in new_state.drone_trips]
    return new_state

# --- TOÁN TỬ SỬA CHỮA (REPAIR OPERATORS) ---
# Các toán tử này tái thiết lập lời giải bằng cách chèn lại các đơn hàng đã bị loại bỏ.

def greedy_repair(state: VRPState, rnd_state):
    """
    Chèn lại các đơn hàng chưa phục vụ bằng logic tham lam cải tiến.
    """
    # Sử dụng hàm solve_greedy đã được định nghĩa trong greedy_solver.py
    return solve_greedy(state)

# --- VÒNG LẶP CHÍNH CỦA BỘ GIẢI ALNS ---

def solve_alns(iterations=1000):
    """
    Hàm thực thi chính của thuật toán ALNS.
    Quy trình:
    1. Tạo lời giải ban đầu bằng Heuristic Tham lam.
    2. Trong mỗi vòng lặp:
       - Chọn ngẫu nhiên một toán tử Phá hủy và một toán tử Sửa chữa (dựa trên trọng số thích nghi).
       - Phá hủy một phần lời giải cũ và xây lại lời giải mới.
       - Quyết định có chấp nhận lời giải mới hay không (Hill Climbing / Simulated Annealing).
    3. Cập nhật điểm số cho các toán tử để ưu tiên các toán tử mang lại kết quả tốt.
    """
    # Bước 1: Khởi tạo trạng thái và tạo lời giải mồi bằng Greedy
    initial_state = VRPState(DATA, LW, ALPHA, GAMMA)
    initial_sol = solve_greedy(initial_state)
    
    print("Kết quả Tham lam ban đầu (Gốc để ALNS bắt đầu tối ưu):")
    print_solution(initial_sol)
    
    # Khởi tạo khung ALNS với hạt giống ngẫu nhiên cố định để kết quả có thể tái lập
    alns = ALNS(np.random.RandomState(42))
    
    # Đăng ký các chiến lược phá hủy và sửa chữa
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(shaw_removal)
    alns.add_repair_operator(greedy_repair)
    
    # Cấu hình RouletteWheel: Cách thuật toán 'học' để chọn toán tử tốt
    # Trọng số điểm cho 4 trường hợp: [Tìm thấy Best mới, Tìm thấy lời giải tốt hơn hiện tại, Chấp nhận lời giải tệ hơn, Lời giải bị từ chối]
    select = RouletteWheel([50, 20, 5, 2], 0.8, 2, 1) 
    
    # Chiến lược chấp nhận: 
    # HillClimbing: Chỉ chấp nhận nếu lời giải mới TỐT HƠN lời giải hiện tại.
    accept = SimulatedAnnealing(5000, 1, 0.9997)
    
    # Điều kiện dừng: Sau một số lượng vòng lặp nhất định
    stop = MaxIterations(iterations)
    
    print(f"Đang thực hiện tối ưu hóa ALNS qua {iterations} vòng lặp...")
    result = alns.iterate(initial_sol, select, accept, stop)
    
    # Trả về lời giải tốt nhất tìm được trong toàn bộ quá trình tìm kiếm
    return result.best_state

if __name__ == "__main__":
    best_state = solve_alns(iterations=20000)
    print("\nHoàn tất tối ưu hóa ALNS!")
    print_solution(best_state)
