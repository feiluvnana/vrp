import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple

# =============================================================================
# CẤU TRÚC DỮ LIỆU CỐT LÕI CHO BÀI TOÁN VRP (VEHICLE ROUTING PROBLEM)
# =============================================================================
# Hệ thống hỗ trợ bài toán Truck và Drone hoạt động song song, có ràng buộc:
# - Cửa sổ thời gian (Time Window)
# - Dung tải (Capacity)
# - Bán kính bay của Drone (Battery/Range)
# - Thời gian chờ tối đa (Waiting Time/Freshness)
# =============================================================================

def load_data(file_path):
    """Tải dữ liệu bài toán từ file JSON."""
    try:
        with open(file_path, "r") as f: return json.load(f)
    except FileNotFoundError:
        # Dữ liệu mặc định nếu không tìm thấy file
        return {
            "requests": [], 
            "truck_vel": 15.0, "drone_vel": 30.0, 
            "truck_cap": 400.0, "drone_cap": 2.27, 
            "drone_lim": 700.0, 
            "truck_num": 1, "drone_num": 1
        }

DATA = load_data("data.json")

# Hằng số hệ thống
ALPHA = 1.0     # Trọng số cho tổng quãng đường trong hàm mục tiêu
GAMMA = 10000.0 # Hình phạt (penalty) cho mỗi đơn hàng không được phục vụ
LW = 3600.0    # Giới hạn Waiting Time (ví dụ: 1 giờ = 3600s)

@dataclass(frozen=True)
class Request:
    # Use __slots__ to reduce memory footprint for thousands of instances
    __slots__ = ['id', 'x', 'y', 'demand', 'can_drone', 'request_time', 'earliest_start', 'latest_end']
    id: int
    x: float
    y: float
    demand: float
    can_drone: bool
    request_time: float      # Time when the request becomes available
    earliest_start: float    # Delivery time window start
    latest_end: float        # Delivery time window end

def build_dist_matrix(requests):
    """Precompute distance matrix to avoid repeated sqrt calculations (O(N^2))."""
    # Index 0 is the Depot, indices 1 to N are requests
    nodes = [(0.0, 0.0)] + [(r.x, r.y) for r in requests]
    n = len(nodes)
    matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = math.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
            matrix[i][j] = matrix[j][i] = d
    return matrix

@dataclass
class Trip:
    """
    Thực thể Hành trình (Trip/Tour/Sortie).
    Đây là đơn vị nguyên tử của lời giải, biểu diễn một chuỗi khách hàng được 
    phục vụ bởi một phương tiện trong một lần rời Depot.
    """
    vehicle_type: str       # 'truck' hoặc 'drone'
    vehicle_index: int      # Chỉ số phương tiện (0, 1, 2...)
    stops: List[Request] = field(default_factory=list)
    depart_time: float = 0.0 # Thời gian rời Depot
    
    # Các thuộc tính được tính toán sau (Calculated properties)
    return_time: float = 0.0
    total_dist: float = 0.0
    total_load: float = 0.0
    service_times: Dict[int, float] = field(default_factory=dict) # RequestID -> Service Start Time
    post_service_wait: Dict[int, float] = field(default_factory=dict)

    def eval(self, dist_matrix, velocity, capacity, distance_limit, max_wait_limit) -> bool:
        """
        Hàm thẩm định (Evaluation): Kiểm tra tính khả thi và tính toán các thông số của hành trình.
        Ràng buộc được kiểm tra theo thứ tự: Request tồn tại -> Dung tải -> Quãng đường (Pin) -> Time Window.
        """
        # 1. Kiểm tra thời điểm xuất hiện của các request: Xe không thể đi trước khi đơn hàng xuất hiện.
        if self.stops and self.depart_time < max(req.request_time for req in self.stops):
            return False

        prev_node_idx = 0 # Bắt đầu từ Depot (index 0 trong ma trận khoảng cách)
        current_time, dist_acc, load_acc = self.depart_time, 0.0, 0.0
        service_times = {}

        for req in self.stops:
            # 2. Kiểm tra dung tải (Capacity)
            load_acc += req.demand
            if load_acc > capacity: return False
            
            # Tính toán logistics
            curr_node_idx = req.id + 1
            leg = dist_matrix[prev_node_idx][curr_node_idx]
            dist_acc += leg
            
            # 3. Kiểm tra giới hạn quãng đường (Distance Limit - quan trọng với Drone/Battery)
            if dist_acc > distance_limit: return False
            
            # 4. Kiểm tra Cửa sổ thời gian (Time Window)
            arrival_time = current_time + leg / velocity
            if arrival_time > req.latest_end: # Đến muộn hơn cửa sổ cho phép
                return False
            
            # Thời điểm bắt đầu phục vụ (S_i) = max(Thời điểm đến, Thời điểm mở cửa)
            pickup_time = max(arrival_time, req.earliest_start)
            service_times[req.id] = pickup_time
            
            prev_node_idx = curr_node_idx
            current_time = pickup_time

        # Tính toán chân quay về Depot (Leg back)
        leg_back = dist_matrix[prev_node_idx][0]
        dist_acc += leg_back
        if dist_acc > distance_limit: return False
        
        q_depot = current_time + leg_back / velocity
        
        # 5. Kiểm tra thời hạn phải về Depot (Horizon constraint - nếu có)
        # Trong bài toán này, return_time được lưu lại để kiểm tra với tour tiếp theo.
        
        # 6. Kiểm tra thời gian chờ tối đa (Waiting Time/Freshness)
        # Thời gian từ lúc lấy hàng ($p_i$) đến khi hàng về kho ($q_i$) <= LW
        post_service_wait = {}
        for rid, p_i in service_times.items():
            wait_time = q_depot - p_i
            if wait_time > max_wait_limit: return False
            post_service_wait[rid] = wait_time
        
        # Cập nhật các thông số sau khi thẩm định thành công
        self.return_time, self.total_dist = q_depot, dist_acc
        self.total_load, self.service_times = load_acc, service_times
        self.post_service_wait = post_service_wait
        return True

    def __repr__(self):
        """Hiển thị tóm tắt hành trình (phục vụ debug/đọc code)."""
        stop_ids = [r.id for r in self.stops]
        return f"Trip({self.vehicle_type}#{self.vehicle_index}: {stop_ids}, Dep:{self.depart_time:.1f}, Ret:{self.return_time:.1f})"

class VRPState:
    """
    Biểu diễn lời giải hoàn chỉnh (Solution Representation).
    Quản lý tập hợp các hành trình của tất cả phương tiện và các request chưa được phục vụ.
    """
    def __init__(self, data, max_wait_time, dist_weight, penalty_weight):
        self.data = data
        self.max_wait_time = max_wait_time
        self.dist_weight = dist_weight      # ALPHA
        self.penalty_weight = penalty_weight # GAMMA
        
        # Biểu diễn dữ liệu đầu vào dưới dạng đối tượng
        self.requests = [Request(i, r[0], r[1], r[2], bool(r[3]), r[4], r[5], r[6]) for i, r in enumerate(data["requests"])]
        self.dist_matrix = build_dist_matrix(self.requests)
        
        # --- Solution Encoding ---
        # Danh sách các Trips cho mỗi phương tiện. Ví dụ: truck_trips[0] = [Trip1, Trip2...]
        self.truck_trips = [[] for _ in range(data["truck_num"])]
        self.drone_trips = [[] for _ in range(data["drone_num"])]
        
        # Theo dõi các khách hàng chưa được phục vụ (để tính penalty)
        self.unserved = set(self.requests)

    def objective(self) -> float:
        """
        Tính hàm mục tiêu: Tổng chi phí = sum(Khoảng cách) * ALPHA + len(Unserved) * GAMMA.
        """
        total_dist = 0.0
        for v_trips in self.truck_trips + self.drone_trips:
            for trip in v_trips:
                total_dist += trip.total_dist
        
        return self.dist_weight * total_dist + self.penalty_weight * len(self.unserved)

    def validate_global(self) -> bool:
        """
        Kế hoạch kiểm tra toàn cục: Đảm bảo các hành trình của cùng một xe không bị chồng chéo thời gian.
        """
        for v_trips in self.truck_trips + self.drone_trips:
            for i in range(1, len(v_trips)):
                if v_trips[i].depart_time < v_trips[i-1].return_time:
                    return False
        return True

    def copy(self):
        """
        Tạo bản sao sâu (Deep copy) để phục vụ các thuật toán tìm kiếm (Heuristics/ALNS).
        """
        import copy
        new_state = copy.copy(self)
        
        def clone_trips(v_trips_list):
            new_v_list = []
            for trips in v_trips_list:
                new_t_list = []
                for t in trips:
                    nt = Trip(t.vehicle_type, t.vehicle_index, list(t.stops), t.depart_time)
                    nt.return_time = t.return_time
                    nt.total_dist = t.total_dist
                    nt.total_load = t.total_load
                    nt.service_times = t.service_times.copy()
                    nt.post_service_wait = t.post_service_wait.copy()
                    new_t_list.append(nt)
                new_v_list.append(new_t_list)
            return new_v_list

        new_state.truck_trips = clone_trips(self.truck_trips)
        new_state.drone_trips = clone_trips(self.drone_trips)
        new_state.unserved = self.unserved.copy()
        return new_state

def print_solution(state: VRPState, name="SOL"):
    print(f"\n=== {name} | Objective: {state.objective():.2f} ===")
    for i, v in enumerate(state.truck_trips):
        for trip in v: print(f"Truck {i}: Dep {trip.depart_time:.1f} | Ret {trip.return_time:.1f} | Stops {[r.id for r in trip.stops]}")
    for i, v in enumerate(state.drone_trips):
        for trip in v: print(f"Drone {i}: Dep {trip.depart_time:.1f} | Ret {trip.return_time:.1f} | Stops {[r.id for r in trip.stops]}")
    if state.unserved: print(f"Unserved: {[r.id for r in state.unserved]}")
