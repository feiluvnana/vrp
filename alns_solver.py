from vrp_core import Request, Trip, VRPState, DATA, LW, ALPHA, GAMMA
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
    Loại bỏ ngẫu nhiên k% đơn hàng khỏi lời giải hiện tại.
    Giúp tạo sự ngẫu nhiên để thoát khỏi tối ưu cục bộ.
    """
    new_state = state.copy()
    all_served = []
    # Thu thập tất cả các chuyến đi đã có đơn hàng
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            all_served.extend([(t, r) for r in t.stops])
    
    if not all_served:
        return new_state
        
    k = max(1, int(len(new_state.requests) * 0.3))
    num_to_remove = min(len(all_served), k)
    to_remove = rnd_state.choice(len(all_served), num_to_remove, replace=False)
    
    for idx in to_remove:
        trip, req = all_served[idx]
        trip.stops.remove(req)
        new_state.unserved.add(req)
    
    # Làm sạch các chuyến đi có thể bị trống
    new_state.truck_trips = [[t for t in v if t.stops] for v in new_state.truck_trips]
    new_state.drone_trips = [[t for t in v if t.stops] for v in new_state.drone_trips]
    return new_state

def shaw_removal(state: VRPState, rnd_state):
    """
    Shaw Removal: Loại bỏ các đơn hàng 'liên quan' (gần nhau về không gian/thời gian).
    Cho phép toán tử sửa chữa sắp xếp lại một khu vực lân cận của lịch trình.
    """
    new_state = state.copy()
    all_reqs = []
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            all_reqs.extend([r for r in t.stops])
            
    if not all_reqs: return new_state
        
    pivot = rnd_state.choice(all_reqs)
    
    # Hàm tính độ liên quan: kết hợp khoảng cách và chênh lệch thời gian có sẵn
    def relatedness(r1, r2):
        dist = math.sqrt((r1.x-r2.x)**2 + (r1.y-r2.y)**2)
        time_diff = abs(r1.r_i - r2.r_i)
        return dist + 0.1 * time_diff # Chi phí kết hợp có trọng số
        
    all_reqs.sort(key=lambda r: relatedness(pivot, r))
    k = max(1, int(len(new_state.requests) * 0.4))
    to_remove = all_reqs[:k]
    
    for req in to_remove:
        for v_trips in new_state.truck_trips + new_state.drone_trips:
            for t in v_trips:
                if req in t.stops:
                    t.stops.remove(req)
                    new_state.unserved.add(req)
                    
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

# --- TRÌNH IN KẾT QUẢ ---

def print_solution(state: VRPState):
    """In tóm tắt lời giải VRP."""
    print("\n" + "="*50)
    print("    TÓM TẮT LỜI GIẢI VRP")
    print("="*50)
    print(f"Tổng đơn hàng  : {len(state.requests)}")
    print(f"Đã phục vụ     : {len(state.requests) - len(state.unserved)}")
    print(f"Chi phí mục tiêu: {state.objective():.2f}")
    
    print("\nCHUYẾN ĐI DRONE:")
    for d, trips in enumerate(state.drone_trips):
        for idx, t in enumerate(trips):
            print(f"  D{d} T{idx}: Điểm {[s.id for s in t.stops]} | Q.Đường {t.total_dist:.0f}m | Về kho {t.return_time:.0f}s")
            
    print("\nCHUYẾN ĐI XE TẢI:")
    for t_idx, trips in enumerate(state.truck_trips):
        for idx, t in enumerate(trips):
            print(f"  VT{t_idx} T{idx}: Điểm {[s.id for s in t.stops]} | Q.Đường {t.total_dist:.0f}m | Về kho {t.return_time:.0f}s")
    print("="*50 + "\n")

# --- VÒNG LẶP CHÍNH CỦA BỘ GIẢI ALNS ---

def solve_alns(iterations=1000):
    """
    Vòng lặp Tìm kiếm Lân cận Lớn Thích nghi (ALNS).
    1. Bắt đầu với một lời giải tham lam ban đầu.
    2. Lặp lại quá trình phá hủy (loại bỏ) và sửa chữa (chèn lại) các phần của lời giải.
    3. Thích nghi cách chọn toán tử dựa trên hiệu quả trong quá khứ.
    """
    initial_state = VRPState(DATA, LW, ALPHA, GAMMA)
    initial_sol = solve_greedy(initial_state)
    
    print("Kết quả Tham lam ban đầu:")
    print_solution(initial_sol)
    
    alns = ALNS(np.random.RandomState(42))
    
    # Đăng ký các thành phần logic
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(shaw_removal)
    alns.add_repair_operator(greedy_repair)
    
    # Thiết lập thuật toán chọn và chấp nhận
    # RouletteWheel: gán xác suất cho các toán tử dựa trên điểm số
    select = RouletteWheel([50, 20, 5, 2], 0.8, 2, 1) # Điểm cho [Tối ưu mới, Tốt hơn, Được chấp nhận, Tệ hơn]
    
    # Chấp nhận: HillClimbing (chỉ chấp nhận tốt hơn) hoặc SimulatedAnnealing
    accept = HillClimbing()
    
    stop = MaxIterations(iterations)
    
    print(f"Đang chạy ALNS trong {iterations} vòng lặp...")
    result = alns.iterate(initial_sol, select, accept, stop)
    
    return result.best_state

if __name__ == "__main__":
    best_state = solve_alns(iterations=500)
    print("\nHoàn tất tối ưu hóa ALNS!")
    print_solution(best_state)
