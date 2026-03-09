import math
import json
import os
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

# --- CẤU HÌNH DỮ LIỆU VÀ THAM SỐ BÀI TOÁN ---
# Load dữ liệu từ file data.json
def load_data(file_path="data.json"):
    """
    Đọc dữ liệu bài toán từ file JSON.
    Nội dung bao gồm: thông tin đơn hàng, số lượng xe, vận tốc và tải trọng phương tiện.
    """
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, file_path)
    
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Fallback dữ liệu mẫu nếu không tìm thấy file
        print(f"CẢNH BÁO: Không tìm thấy {file_path}, sử dụng dữ liệu mặc định.")
        return {"requests": [], "truck_vel": 15.0, "drone_vel": 30.0, "truck_cap": 400.0, "drone_cap": 2.27, "drone_lim": 700.0, "truck_num": 1, "drone_num": 1}

# Khởi tạo dữ liệu toàn cục
DATA = load_data("data.json")

# Tham số tối ưu hóa cho hàm mục tiêu
ALPHA = 1.0       # Trọng số chi phí di chuyển (tổng quãng đường)
GAMMA = 10000.0    # Hình phạt (penalty) cực lớn cho mỗi đơn hàng bị bỏ sót
LW = 3600.0       # Giới hạn thời gian chờ tối đa của hàng hóa (giây)
DEPOT = [0.0, 0.0] # Tọa độ của Kho trung tâm

@dataclass(frozen=True)
class Request:
    """
    Đại diện cho một đơn hàng của khách hàng.
    """
    id: int         # ID duy nhất của đơn hàng
    x: float        # Tọa độ X
    y: float        # Tọa độ Y
    demand: float   # Khối lượng hàng hóa (kg)
    can_drone: bool # Cho biết drone có thể giao đơn này không (do kích thước hoặc vùng cấm)
    r_i: float      # Thời điểm hàng có sẵn tại kho (Release time)
    e_i: float      # Thời điểm bắt đầu cửa sổ thời gian (Earliest time)
    l_i: float      # Thời điểm kết thúc cửa sổ thời gian (Latest time)

@dataclass
class Trip:
    """
    Đại diện cho một lộ trình khép kín: Kho -> khách hàng 1 -> ... -> khách hàng N -> Kho.
    """
    vtype: str  # Loại phương tiện: 'truck' (xe tải) hoặc 'drone'
    vidx: int   # Chỉ số của phương tiện trong đội xe
    stops: List[Request] = field(default_factory=list) # Danh sách các điểm dừng (đơn hàng)
    depart_time: float = 0.0 # Thời gian xe bắt đầu rời kho
    return_time: float = 0.0 # Thời gian xe quay về tới kho
    total_dist: float = 0.0  # Tổng quãng đường di chuyển của chuyến đi
    total_load: float = 0.0  # Tổng khối lượng hàng hóa chuyên chở trên chuyến đi
    pis: Dict[int, float] = field(default_factory=dict) # Thời điểm thực tế phục vụ từng đơn hàng
    wis: Dict[int, float] = field(default_factory=dict) # Thời gian chờ đợi của từng mặt hàng trên xe

    def eval(self, start_pos, vel, cap, dist_lim, Lw) -> bool:
        """
        Kiểm tra tính hợp lệ của chuyến đi dựa trên các ràng buộc vật lý và thời gian.
        Trả về True nếu chuyến đi thỏa mãn tất cả điều kiện, ngược lại False.
        """
        pos = start_pos
        t = self.depart_time
        dist_acc = 0.0
        load_acc = 0.0
        pis = {}

        for req in self.stops:
            # 1. Kiểm tra tải trọng tối đa
            load_acc += req.demand
            if load_acc > cap:
                return False
            
            # 2. Kiểm tra giới hạn quãng đường (đặc biệt quan trọng với drone - Pin)
            leg = math.sqrt((pos[0] - req.x)**2 + (pos[1] - req.y)**2)
            dist_acc += leg
            if dist_acc > dist_lim:
                return False
            
            # 3. Kiểm tra cửa sổ thời gian giao hàng (Time Window)
            arrival_time = t + leg / vel # Thời điểm đến vị trí khách hàng
            if arrival_time > req.l_i: # Nếu đến muộn hơn giới hạn kết thúc
                return False
            
            # Thời điểm phục vụ thực tế: phải chờ nếu đến sớm hơn e_i
            pickup_time = max(arrival_time, req.e_i)
            pis[req.id] = pickup_time
            
            # Cập nhật trạng thái phương tiện
            pos = [req.x, req.y]
            t = pickup_time

        # Quay về kho
        leg_back = math.sqrt((pos[0] - 0.0)**2 + (pos[1] - 0.0)**2)
        dist_acc += leg_back
        if dist_acc > dist_lim:
            return False
        
        q_depot = t + leg_back / vel # Thời điểm xe về kho
        
        # 4. Kiểm tra giới hạn thời gian tồn kho (Waiting time LW)
        # Thời gian từ lúc lấy hàng đến lúc xe về kho không được quá LW
        wis = {}
        for rid, p_i in pis.items():
            W_i = q_depot - p_i
            if W_i > Lw:
                return False
            wis[rid] = W_i
        
        # Lưu kết quả tính toán nếu chuyến đi hợp lệ
        self.return_time = q_depot
        self.total_dist = dist_acc
        self.total_load = load_acc
        self.pis = pis
        self.wis = wis
        return True

class VRPState:
    """
    Quản lý trạng thái của một lời giải (toàn bộ lịch trình các xe).
    """
    def __init__(self, data, lw, alpha, gamma):
        self.data = data
        self.lw = lw
        self.alpha = alpha
        self.gamma = gamma
        
        # Chuyển đổi dữ liệu thô sang đối tượng Request
        self.requests = [
            Request(
                id=i,
                x=r[0], y=r[1],
                demand=r[2],
                can_drone=bool(r[3]),
                r_i=r[4], e_i=r[5], l_i=r[6]
            ) for i, r in enumerate(data["requests"])
        ]
        
        # Cấu trúc lịch trình: List của các danh sách Trip cho từng phương tiện
        self.truck_trips: List[List[Trip]] = [[] for _ in range(data["truck_num"])]
        self.drone_trips: List[List[Trip]] = [[] for _ in range(data["drone_num"])]
        self.unserved: Set[Request] = set(self.requests) # Tập hợp các đơn hàng chưa được phục vụ

    def objective(self) -> float:
        """
        Tính toán giá trị hàm mục tiêu (Objective Value).
        Bao gồm tổng quãng đường di chuyển và hình phạt cho các đơn hàng chưa phục vụ.
        Mục tiêu của bài toán là tìm lời giải có giá trị này nhỏ nhất.
        """
        dist_truck = sum(t.total_dist for v_trips in self.truck_trips for t in v_trips)
        dist_drone = sum(t.total_dist for v_trips in self.drone_trips for t in v_trips)
        return self.alpha * (dist_truck + dist_drone) + self.gamma * len(self.unserved)

    def copy(self):
        """
        Tạo một bản sao sâu của trạng thái lời giải để phục vụ ALNS.
        """
        import copy
        return copy.deepcopy(self)

def print_solution(state: VRPState, name="SOLVER"):
    """
    Xuất kết quả lời giải ra terminal theo định dạng JSON.
    Hỗ trợ theo dõi lộ trình của từng xe và đánh giá hiệu quả thuật toán.
    """
    dropped = sorted([r.id for r in state.unserved])
    v_idx = 0
    
    # In lộ trình xe tải
    for i, trips in enumerate(state.truck_trips):
        route_data = {}
        for t in trips:
            for s in t.stops:
                route_data[str(int(t.return_time))] = s.id
            route_data[str(int(t.return_time) + 1)] = 0 
        print(json.dumps({"__": "ROUTE", "_": "route_log", "vehicle": v_idx, "route": route_data, "dropped": dropped}))
        v_idx += 1
        
    # In lộ trình drone
    for i, trips in enumerate(state.drone_trips):
        route_data = {}
        for t in trips:
            for s in t.stops:
                route_data[str(int(t.return_time))] = s.id
            route_data[str(int(t.return_time) + 1)] = 0
        print(json.dumps({"__": "ROUTE", "_": "route_log", "vehicle": v_idx, "route": route_data, "dropped": dropped}))
        v_idx += 1

    # Tóm tắt kết quả cuối cùng
    obj_val = state.objective()
    num_dropped = len(state.unserved)
    print(json.dumps({"__": name, "_": "full_result", "result": [round(obj_val, 2), num_dropped]}, ensure_ascii=False))
