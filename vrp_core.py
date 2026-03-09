import math
import json
import os
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

# --- CẤU HÌNH DỮ LIỆU VÀ THAM SỐ BÀI TOÁN ---
# Load dữ liệu từ file data.json
def load_data(file_path="data.json"):
    # Lấy đường dẫn tuyệt đối để tránh lỗi khi chạy từ thư mục khác
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, file_path)
    
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Fallback dữ liệu mẫu nếu không tìm thấy file
        print(f"CẢNH BÁO: Không tìm thấy {file_path}, sử dụng dữ liệu mặc định.")
        return {"requests": [], "truck_vel": 15.0, "drone_vel": 30.0, "truck_cap": 400.0, "drone_cap": 2.27, "drone_lim": 700.0, "truck_num": 1, "drone_num": 1}

DATA = load_data("data.json")

# Tham số tối ưu cho bài toán VRP
ALPHA = 1.0       # Hệ số chi phí trên mỗi đơn vị quãng đường di chuyển
GAMMA = 10000.0    # Hình phạt (penalty) cho mỗi đơn hàng không được phục vụ
LW = 3600.0       # Thời gian chờ tối đa (giây) của hàng hóa kể từ khi lấy từ kho
DEPOT = [0.0, 0.0] # Tọa độ vị trí kho trung tâm

@dataclass(frozen=True)
class Request:
    """
    Đại diện cho một yêu cầu giao hàng của khách hàng.
    
    Thuộc tính:
        id: Chỉ số/ID gốc của đơn hàng.
        x, y: Tọa độ của khách hàng.
        demand: Khối lượng hàng (kg).
        can_drone: Boolean cho biết drone có thể giao đơn này không.
        r_i: Thời điểm hàng có sẵn tại kho (Release time).
        e_i: Thời điểm bắt đầu cửa sổ thời gian giao hàng.
        l_i: Thời điểm kết thúc cửa sổ thời gian giao hàng.
    """
    id: int
    x: float
    y: float
    demand: float
    can_drone: bool
    r_i: float
    e_i: float
    l_i: float

@dataclass
class Trip:
    """
    Đại diện cho một chuyến đi duy nhất từ kho, ghé qua một hoặc nhiều điểm và quay về.
    """
    vtype: str  # 'truck' hoặc 'drone'
    vidx: int   # Chỉ số phương tiện
    stops: List[Request] = field(default_factory=list)
    depart_time: float = 0.0
    return_time: float = 0.0
    total_dist: float = 0.0
    total_load: float = 0.0
    pis: Dict[int, float] = field(default_factory=dict) # Thời gian đến/lấy hàng tại mỗi điểm
    wis: Dict[int, float] = field(default_factory=dict) # Thời gian chờ (từ lúc lấy hàng đến lúc xe về kho)

    def eval(self, start_pos, vel, cap, dist_lim, Lw) -> bool:
        """
        Giai đoạn 'mô phỏng' chuyến đi để kiểm tra xem nó có hợp lệ hay không.
        Sử dụng logic của bài toán VRP với xe tải và drone:
        - start_pos: Tọa độ bắt đầu (thường là DEPOT [0,0]).
        - vel: Vận tốc phương tiện.
        - cap: Tải trọng tối đa.
        - dist_lim: Giới hạn quãng đường pin (drone).
        - Lw: Thời gian tồn kho tối đa cho phép.
        """
        pos = start_pos
        t = self.depart_time
        dist_acc = 0.0
        load_acc = 0.0
        pis = {} # Lưu lại thời gian bắt đầu xử lý từng đơn hàng

        for req in self.stops:
            # 1. KIỂM TRA TẢI TRỌNG (Capacity constraint)
            load_acc += req.demand
            if load_acc > cap:
                return False
            
            # 2. KIỂM TRA QUÃNG ĐƯỜNG (Distance/Battery constraint)
            # Tính khoảng cách Euclidean từ vị trí hiện tại đến khách hàng tiếp theo
            leg = math.sqrt((pos[0] - req.x)**2 + (pos[1] - req.y)**2)
            dist_acc += leg
            if dist_acc > dist_lim:
                return False
            
            # 3. KIỂM TRA CỬA SỔ THỜI GIAN (Time Window constraint)
            # Thời điểm phương tiện đến vị trí khách hàng
            arrival_time = t + leg / vel
            if arrival_time > req.l_i:
                return False
            
            # Thời điểm lấy hàng/phục vụ thực tế (phải nằm trong cửa sổ [e_i, l_i])
            pickup_time = max(arrival_time, req.e_i)
            pis[req.id] = pickup_time
            
            # Cập nhật vị trí và thời gian hiện tại của phương tiện
            pos = [req.x, req.y]
            t = pickup_time

        # Quay về vị trí kho (DEPOT)
        leg_back = math.sqrt((pos[0] - 0.0)**2 + (pos[1] - 0.0)**2)
        dist_acc += leg_back
        # Kiểm tra giới hạn quãng đường bao gồm cả lượt về
        if dist_acc > dist_lim:
            return False
        
        # Thời điểm phương tiện quay về đến kho
        q_depot = t + leg_back / vel
        
        # 4. KIỂM TRA HÀI LÒNG KHÁCH HÀNG (Waiting time constraint LW)
        # Hàng không được ở trên xe quá lâu kể từ khi sẵn sàng ở kho cho đến khi xe về
        wis = {}
        for rid, p_i in pis.items():
            W_i = q_depot - p_i
            if W_i > Lw:
                return False
            wis[rid] = W_i
        
        # Nếu đi qua tất cả các bước mà không gặp False, chuyến đi là hợp lệ
        # Cập nhật thông số thực tế cho đối tượng Trip
        self.return_time = q_depot
        self.total_dist = dist_acc
        self.total_load = load_acc
        self.pis = pis
        self.wis = wis
        return True

class VRPState:
    """
    Đại diện cho trạng thái hiện tại của lời giải bài toán VRP.
    """
    def __init__(self, data, lw, alpha, gamma):
        self.data = data
        self.lw = lw
        self.alpha = alpha
        self.gamma = gamma
        
        self.requests = [
            Request(
                id=i,
                x=r[0], y=r[1],
                demand=r[2],
                can_drone=bool(r[3]),
                r_i=r[4], e_i=r[5], l_i=r[6]
            ) for i, r in enumerate(data["requests"])
        ]
        
        self.truck_trips: List[List[Trip]] = [[] for _ in range(data["truck_num"])]
        self.drone_trips: List[List[Trip]] = [[] for _ in range(data["drone_num"])]
        self.unserved: Set[Request] = set(self.requests)

    def objective(self) -> float:
        """
        Tính toán giá trị hàm mục tiêu (Objective function) - Tổng chi phí cần tối thiểu hóa.
        Chi phí = Alpha * (Tổng quãng đường) + Gamma * (Số đơn hàng bị bỏ sót)
        """
        dist_truck = sum(t.total_dist for v_trips in self.truck_trips for t in v_trips)
        dist_drone = sum(t.total_dist for v_trips in self.drone_trips for t in v_trips)
        # alpha * tổng quãng đường + gamma * số đơn hàng không được phục vụ
        return self.alpha * (dist_truck + dist_drone) + self.gamma * len(self.unserved)

    def copy(self):
        """
        Tạo một bản sao sâu (deep copy) của trạng thái.
        """
        import copy
        return copy.deepcopy(self)

def print_solution(state: VRPState, name="SOLVER"):
    """
    In kết quả lời giải dưới dạng ngắn gọn (JSON-like), tương thích với định dạng của file result.jsonc.
    
    Định dạng này giúp dễ dàng so sánh kết quả giữa các thuật toán (Greedy, ALNS) và dữ liệu mẫu (Benchmark).
    Mỗi dòng in ra là một đối tượng JSON đại diện cho:
    1. "ROUTE": Lộ trình chi tiết của một phương tiện (xe tải hoặc drone).
       - vehicle: Chỉ số của phương tiện.
       - route: Một dictionary với key là thời gian (giây) và value là ID của điểm dừng (0 là kho).
       - dropped: Danh sách các ID đơn hàng không được phục vụ.
    2. "full_result": Tóm tắt tổng thể lời giải.
       - result: [Tổng chi phí mục tiêu, Số đơn hàng bị bỏ sót].
    """
    dropped = sorted([r.id for r in state.unserved])
    
    # --- PHẦN 1: IN LỘ TRÌNH CHI TIẾT CỦA TỪNG PHƯƠNG TIỆN ---
    v_idx = 0
    
    # Xử lý lộ trình cho các Xe tải (Truck)
    for i, trips in enumerate(state.truck_trips):
        route_data = {}
        for t in trips:
            # Lưu vết các điểm dừng trong chuyến đi
            for s in t.stops:
                # Sử dụng thời gian hoàn thành chuyến làm mốc thời gian (key)
                route_data[str(int(t.return_time))] = s.id
            # Điểm dừng cuối cùng luôn là kho (ID = 0)
            route_data[str(int(t.return_time) + 1)] = 0 
        
        # In dưới dạng JSON một dòng
        print(json.dumps({"__": "ROUTE", "_": "route_log", "vehicle": v_idx, "route": route_data, "dropped": dropped}))
        v_idx += 1
        
    # Xử lý lộ trình cho các Drone
    for i, trips in enumerate(state.drone_trips):
        route_data = {}
        for t in trips:
            for s in t.stops:
                route_data[str(int(t.return_time))] = s.id
            route_data[str(int(t.return_time) + 1)] = 0
            
        print(json.dumps({"__": "ROUTE", "_": "route_log", "vehicle": v_idx, "route": route_data, "dropped": dropped}))
        v_idx += 1

    # --- PHẦN 2: IN TÓM TẮT KẾT QUẢ CUỐI CÙNG ---
    obj_val = state.objective()
    num_dropped = len(state.unserved)
    # Hiển thị kết quả tổng hợp bao gồm chi phí (đã làm tròn) và số lượng đơn bị bỏ sót
    print(json.dumps({"__": name, "_": "full_result", "result": [round(obj_val, 2), num_dropped]}, ensure_ascii=False))
