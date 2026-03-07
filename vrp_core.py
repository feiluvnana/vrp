import math
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

# Dữ liệu bài toán mẫu
DATA = {
    "requests": [
        [ 3738.682118759545,   3176.3620564508888,  0.9255958108248765, 1, 1358.7472330159917, 1242.4073038422487, 1602.4073038422487],
        [-7332.379353375232,  -4188.522793367846,   0.7478413173478692, 1,  605.3789606058994,  652.9093445397065, 1012.9093445397065],
        [ 5777.475570492262,  -7986.273425724185,   0.9634110342579557, 1, 1102.7598350528779, 1237.7513484295284, 1597.7513484295284],
        [ 6996.942373406358,   4418.535163659321,   0.9908166528639936, 1,  722.3235983813729,  969.8366216562558, 1329.8366216562558],
        [ 4400.906550671561,   2432.3349996395264, 10.776638065660618,  0, 1149.8366216562558, 1178.7472330159917, 1538.7472330159917],
        [-1169.2698685472992,  3497.237946688962,  27.380202204619298,  0,    0.0,               55.679027962466364,  415.67902796246636],
        [ 6710.724138902582,  -3348.9561871222304,  0.2506526123531701, 1,    0.0,               59.670309697854066,  419.67030969785407],
        [ -216.72058856320808,-3938.133905039083,   0.8925640200120823, 1,  479.34061939570813, 425.37896060589935,  785.3789606058994 ],
        [-3536.845602701436,   6049.129314355786,   1.0952132256820593, 1,  235.67902796246636, 278.16018765158515,  638.1601876515851 ],
        [  586.8106635735363,  6329.938959833223,   0.5507433758448146, 1,  458.16018765158515, 542.3235983813729,   902.3235983813729 ],
    ],
    "truck_vel": 15.6464,
    "drone_vel": 31.2928,
    "truck_cap": 400.0,
    "drone_cap": 2.27,
    "drone_lim": 700.0,
    "truck_num": 1,
    "drone_num": 1,
}

# Tham số tối ưu cho lời giải VRP
ALPHA = 1.0       # Chi phí trên mỗi mét di chuyển (cho cả xe tải và drone)
GAMMA = 5000.0    # Hình phạt nặng cho việc từ chối đơn hàng (để ưu tiên phục vụ 100%)
LW = 3600.0       # Thời gian chờ tối đa (s) hàng hóa trên xe
DEPOT = [0.0, 0.0] # Vị trí kho trung tâm

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
        Mô phỏng chuyến đi để kiểm tra tính khả thi so với các ràng buộc.
        Cập nhật các chỉ số nội bộ: return_time, total_dist, total_load, pis, wis.
        
        Các ràng buộc được kiểm tra:
        1. Tải trọng: tổng khối lượng <= tải trọng phương tiện.
        2. Khoảng cách: tổng quãng đường <= giới hạn (chỉ dành cho drone).
        3. Cửa sổ thời gian: thời gian đến điểm dừng <= thời gian giao hàng muộn nhất (l_i).
        4. Sự hài lòng của khách hàng: return_time - pickup_time <= Lw.
        """
        pos = start_pos
        t = self.depart_time
        dist_acc = 0.0
        load_acc = 0.0
        pis = {}

        for req in self.stops:
            # 1. Kiểm tra tải trọng
            load_acc += req.demand
            if load_acc > cap:
                return False
            
            # 2. Kiểm tra khoảng cách
            leg = math.sqrt((pos[0] - req.x)**2 + (pos[1] - req.y)**2)
            dist_acc += leg
            if dist_acc > dist_lim:
                return False
            
            # 3. Kiểm tra cửa sổ thời gian
            arrival_time = t + leg / vel
            if arrival_time > req.l_i:
                return False
            
            # Thời gian lấy hàng thực tế là max(thời gian đến, thời gian sớm nhất)
            pickup_time = max(arrival_time, req.e_i)
            pis[req.id] = pickup_time
            pos = [req.x, req.y]
            t = pickup_time

        # Quay về kho
        leg_back = math.sqrt((pos[0] - 0.0)**2 + (pos[1] - 0.0)**2)
        dist_acc += leg_back
        if dist_acc > dist_lim:
            return False
        
        q_depot = t + leg_back / vel
        
        # 4. Kiểm tra thời gian chờ Lw
        wis = {}
        for rid, p_i in pis.items():
            W_i = q_depot - p_i
            if W_i > Lw:
                return False
            wis[rid] = W_i
        
        # Tất cả các bước kiểm tra đều đạt
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
        Tính toán giá trị hàm mục tiêu (tổng chi phí).
        """
        dist_truck = sum(t.total_dist for v_trips in self.truck_trips for t in v_trips)
        dist_drone = sum(t.total_dist for v_trips in self.drone_trips for t in v_trips)
        return self.alpha * (dist_truck + dist_drone) + self.gamma * len(self.unserved)

    def copy(self):
        """
        Tạo một bản sao sâu (deep copy) của trạng thái.
        """
        import copy
        return copy.deepcopy(self)
