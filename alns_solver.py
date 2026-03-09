"""
Module này triển khai thuật toán Tìm kiếm Lân cận Lớn Thích nghi (Adaptive Large Neighborhood Search - ALNS).

ALNS là một khung thuật toán metaheuristic (siêu phỏng đoán) mạnh mẽ. 
Nguyên lý cốt lõi là 'PHÁ HỦY' (Destroy) một phần lời giải hiện tại và 'SỬA CHỮA' (Repair) 
nó để tìm ra các trạng thái tốt hơn, giúp thoát khỏi các hố tối ưu cục bộ.
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

# =============================================================================
# --- PHẦN 1: CÁC TOÁN TỬ PHÁ HỦY (DESTROY OPERATORS) ---
# Nhiệm vụ: Chọn ra một tập hợp các đơn hàng và gỡ chúng khỏi lịch trình hiện tại.
# Việc này tạo ra không gian để ta có thể sắp xếp lại các đơn hàng đó hiệu quả hơn.
# =============================================================================

def random_removal(state: VRPState, rnd_state):
    """
    Toán tử Phá hủy Ngẫu nhiên (Random Removal):
    - Chiến thuật: Chọn 'k' đơn hàng bất kỳ và gỡ chúng ra.
    - Ý nghĩa: Tạo tính ngẫu nhiên cao, giúp thuật toán khám phá những vùng không gian 
      mới mà các thuật toán có tính quy luật khó chạm tới.
    """
    new_state = state.copy()
    all_served = [] # Danh sách các cặp (chuyến đi, đơn hàng)
    
    # 1. Thu thập toàn bộ các đơn hàng đang nằm trong các chuyến đi
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            all_served.extend([(t, r) for r in t.stops])
    
    if not all_served:
        return new_state
        
    # 2. Quyết định số lượng đơn hàng cần xóa (thường từ 10% - 30% tổng số đơn)
    # Với bài toán lớn (100 đơn), xóa khoảng 30 đơn (30%) giúp thay đổi cấu trúc lời giải mạnh mẽ hơn.
    k = max(1, int(len(new_state.requests) * 0.3))
    num_to_remove = min(len(all_served), k)
    
    # 3. Chọn ngẫu nhiên các chỉ số đơn hàng để loại bỏ
    to_remove_indices = rnd_state.choice(len(all_served), num_to_remove, replace=False)
    
    for idx in to_remove_indices:
        trip, req = all_served[idx]
        if req in trip.stops:
            trip.stops.remove(req)   # Gỡ khỏi chuyến đi
            new_state.unserved.add(req) # Đưa vào danh sách 'chờ xử lý' (unserved)
    
    # 4. 'Vệ sinh' các chuyến đi trống (chuyến đi không còn đơn hàng nào)
    new_state.truck_trips = [[t for t in v if t.stops] for v in new_state.truck_trips]
    new_state.drone_trips = [[t for t in v if t.stops] for v in new_state.drone_trips]
    return new_state

def shaw_removal(state: VRPState, rnd_state):
    """
    Toán tử Phá hủy Shaw (Shaw Removal / Similarity Removal):
    - Chiến thuật: Xóa các đơn hàng có tính 'liên quan' cao (ở gần nhau địa lý hoặc thời gian).
    - Ý nghĩa: Nếu các đơn hàng ở gần nhau đang bị chia cho nhiều xe khác nhau, 
      việc xóa cả cụm này sẽ giúp thuật toán Repair có cơ hội gom chúng vào 1 chuyến đi duy nhất. 
      Đây là cách hiệu quả nhất để giảm số lượng chuyến đi.
    """
    new_state = state.copy()
    all_served_reqs = []
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            all_served_reqs.extend([r for r in t.stops])
            
    if not all_served_reqs: return new_state
    
    # 1. Chọn một đơn hàng 'mồi' (pivot) ngẫu nhiên
    pivot = rnd_state.choice(all_served_reqs)
    
    # 2. Tính toán độ liên quan (Relatedness)
    # r1 và r2 càng gần nhau (vị trí + thời gian sẵn sàng) thì giá trị này càng thấp.
    def relatedness(r1, r2):
        dist = math.sqrt((r1.x-r2.x)**2 + (r1.y-r2.y)**2)
        time_diff = abs(r1.r_i - r2.r_i)
        # Tỷ lệ 1.0 (vị trí) và 0.1 (thời gian) giúp cân bằng giữa không gian và thời gian.
        return dist + 0.1 * time_diff 
        
    # 3. Sắp xếp toàn bộ đơn hàng theo mức độ liên quan tới pivot
    all_served_reqs.sort(key=lambda r: relatedness(pivot, r))
    
    # 4. Chọn top k đơn hàng liên quan nhất để xóa
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

# =============================================================================
# --- PHẦN 2: CÁC TOÁN TỬ SỬA CHỮA (REPAIR OPERATORS) ---
# Nhiệm vụ: Tái thiết lập lời giải bằng cách chèn lại các đơn hàng đang bị bỏ trống.
# =============================================================================

def greedy_repair(state: VRPState, rnd_state):
    """
    Toán tử Sửa chữa Tham lam (Greedy Repair):
    - Chiến thuật: Sử dụng trực tiếp thuật toán Tham lam Cải tiến (Improved Greedy).
    - Ý nghĩa: Tìm vị trí chèn tốt nhất (tăng chi phí ít nhất) cho từng đơn hàng đang chờ.
    """
    return solve_greedy(state)

# =============================================================================
# --- PHẦN 3: ĐIỀU PHỐI ALNS ---
# =============================================================================

def solve_alns(iterations=10000):
    """
    Hàm thực thi chính của thuật toán ALNS.
    Quy trình:
    1. Tạo lời giải ban đầu bằng Greedy.
    2. Trong mỗi vòng lặp:
       - Chọn ngẫu nhiên 1 cặp toán tử (Destroy & Repair).
       - Tính toán lời giải mới.
       - Quyết định chấp nhận/từ chối lời giải đó dựa trên cơ chế Simulated Annealing.
       - Cập nhật 'điểm thưởng' cho các toán từ mang lại kết quả tốt.
    """
    # Bước 1: Khởi tạo trạng thái và tạo 'lời giải mồi' bằng Greedy
    initial_state = VRPState(DATA, LW, ALPHA, GAMMA)
    initial_sol = solve_greedy(initial_state)
    
    # In kết quả ban đầu để so sánh mức độ cải thiện
    print("--- KẾT QUẢ THAM LAM BAN ĐẦU (GỐC) ---")
    print_solution(initial_sol, name="INITIAL_GREEDY")
    
    # Bước 2: Cấu hình khung ALNS
    alns = ALNS(np.random.RandomState(42))
    
    # Đăng ký các toán tử phá hủy và sửa chữa đã viết ở trên
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(shaw_removal)
    alns.add_repair_operator(greedy_repair)
    
    # CƠ CHẾ LỰA CHỌN (Select): Roulette Wheel
    # Trình tự điểm thưởng: [Best mới (50đ), Better mới (20đ), Chấp nhận được (5đ), Bị từ chối (2đ)]
    # decay=0.8: Thuật toán sẽ học từ lịch sử, nhưng ưu tiên 80% kinh nghiệm gần nhất.
    select = RouletteWheel([50, 20, 5, 2], 0.8, 2, 1) 
    
    # CƠ CHẾ CHẤP NHẬN (Accept): Simulated Annealing (Luyện kim)
    # Triết lý: Chấp nhận cả những lời giải TỆ HƠN ở giai đoạn đầu để thoát khỏi bẫy cục bộ.
    # 5000: Nhiệt độ bắt đầu (Dễ dãi). Nên để tương đương với giá trị của một lỗi lớn (ở đây GAMMA=10000).
    # 1: Nhiệt độ kết thúc (Khắt khe). Chỉ chấp nhận lời giải tốt nhất.
    # 0.9997: Hạ nhiệt rất chậm qua 10.000 - 20.000 vòng lặp để tìm kiếm kỹ nhất.
    accept = SimulatedAnnealing(5000, 1, 0.9997)
    
    # Điều kiện dừng
    stop = MaxIterations(iterations)
    
    print(f"\nĐang thực thi tối ưu hóa ALNS qua {iterations} vòng lặp. Vui lòng đợi...")
    result = alns.iterate(initial_sol, select, accept, stop)
    
    # Trả về lời giải tốt nhất tìm được trong toàn bộ quá trình
    return result.best_state

if __name__ == "__main__":
    # Thực hiện tối ưu với 10.000 vòng lặp (con số lý tưởng cho bài toán 100 đơn)
    best_state = solve_alns(iterations=2000)
    
    print("\n" + "="*50)
    print("      HOÀN TẤT TỐI ƯU HÓA ALNS!")
    print("="*50)
    print_solution(best_state, name="ALNS_FINAL")
