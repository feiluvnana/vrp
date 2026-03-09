"""
Module này triển khai thuật toán Tham lam cải tiến (Improved Greedy) để giải bài toán VRP.
Thuật toán tìm cách chèn từng đơn hàng vào vị trí tối ưu nhất (tăng chi phí ít nhất) 
trong lịch trình hiện có của các phương tiện (xe tải và drone).
"""
# Nhập các hằng số và lớp cơ sở từ module vrp_core
from vrp_core import Request, DATA, LW, ALPHA, GAMMA, VRPState, Trip, print_solution
import random
import math

def get_insertion_candidates(state: VRPState, req: Request):
    """
    HÀM TÌM KIẾM VỊ TRÍ CHÈN (Get Insertion Candidates):
    Đây là "trái tim" của thuật toán tham lam và các toán tử sửa chữa.
    Nhiệm vụ: Duyệt qua TẤT CẢ các phương tiện và TẤT CẢ các vị trí có thể để chèn đơn hàng 'req'.
    
    Đầu vào:
    - state: Trạng thái hiện tại của lời giải (lịch trình của xe tải và drone).
    - req: Đơn hàng cụ thể đang cần tìm chỗ chèn.
    
    Trả về: Một danh sách các "ứng viên" (phương án chèn khả thi), mỗi ứng viên là một tuple:
    (chi phí tăng thêm, loại xe, chỉ số xe, chỉ số chuyến, vị trí trong chuyến, thời gian bắt đầu).
    """
    import random
    
    # --- BƯỚC 1: LẤY CÁC THÔNG SỐ VẬT LÝ TỪ DỮ LIỆU ---
    t_vel = state.data["truck_vel"]       # Vận tốc Xe tải (m/s)
    d_vel = state.data["drone_vel"]       # Vận tốc Drone (m/s)
    t_cap = state.data["truck_cap"]       # Tải trọng tối đa Xe tải (kg)
    d_cap = state.data["drone_cap"]       # Tải trọng tối đa Drone (kg)
    # Giới hạn quãng đường bay (Pin) của drone. Chú ý: data["drone_lim"] là đơn vị thời gian (s),
    # nên ta nhân với vận tốc để ra mét.
    d_lim = state.data["drone_lim"] * d_vel 
    
    candidates = [] # Danh sách lưu các kịch bản chèn hợp lệ (candidate list)

    # =========================================================================
    # --- KỊCH BẢN 1: THỬ CHÈN VÀO CÁC CHUYẾN ĐI (TRIP) ĐÃ CÓ SẴN TRONG LỊCH ---
    # =========================================================================
    
    # --- 1.1 Thử chèn vào lộ trình của Drone ---
    # Chỉ thử nếu đơn hàng này cho phép giao bằng drone (can_drone) và khối lượng (demand) nhỏ hơn tải trọng drone.
    if req.can_drone and req.demand <= d_cap:
        drones = list(range(state.data["drone_num"]))
        random.shuffle(drones) # TRỘN NGẪU NHIÊN: Để tránh việc chỉ nhồi đơn vào một.
        for d in drones:
            # Duyệt qua từng chuyến đi (trip) mà drone 'd' đang đảm nhiệm
            for trip_idx, trip in enumerate(state.drone_trips[d]):
                
                # BIẾN THỜI GIAN QUAN TRỌNG:
                # next_start: Thời điểm chuyến đi tiếp theo bắt đầu (Xe không được về kho muộn hơn lúc này).
                # prev_return: Thời điểm chuyến đi trước đó kết thúc (Xe không được đi chuyến này sớm hơn lúc này).
                next_start = state.drone_trips[d][trip_idx+1].depart_time if trip_idx + 1 < len(state.drone_trips[d]) else float('inf')
                prev_return = state.drone_trips[d][trip_idx-1].return_time if trip_idx > 0 else 0.0
                
                # Thử chèn 'req' vào mọi vị trí (pos) bên trong danh sách stops của chuyến đi này
                for pos in range(len(trip.stops) + 1):
                    temp_stops = trip.stops[:pos] + [req] + trip.stops[pos:]
                    
                    # temp_depart: Thời gian khởi hành sớm nhất cho chuyến đi mới này.
                    # Phải thỏa mãn: 1. Xe đã về từ chuyến trước (prev_return)
                    #               2. Hàng hóa cho tất cả các đơn trong chuyến đã có sẵn tại kho (req.r_i).
                    temp_depart = max(prev_return, max(r.r_i for r in temp_stops))
                    
                    # Tạo một chuyến đi tạm thời (temp_trip) để "mô phỏng" việc chèn
                    temp_trip = Trip(vtype='drone', vidx=d, stops=temp_stops, depart_time=temp_depart)
                    
                    # HÀM EVAL: Thực hiện kiểm tra 4 ràng buộc cứng (Tải trọng, Pin, Time Window, LW)
                    if temp_trip.eval([0,0], d_vel, d_cap, d_lim, state.lw):
                        # Nếu chèn xong mà vẫn về kho kịp trước khi phải đi chuyến tiếp theo (next_start)
                        if temp_trip.return_time <= next_start:
                            # delta_dist: Chi phí tăng thêm (quãng đường mới - quãng đường cũ)
                            delta_dist = temp_trip.total_dist - trip.total_dist
                            candidates.append((delta_dist, 'drone', d, trip_idx, pos, temp_depart))

    # --- 1.2 Thử chèn vào lộ trình của Xe tải ---
    if req.demand <= t_cap:
        trucks = list(range(state.data["truck_num"]))
        random.shuffle(trucks) # Shuffle để sử dụng đều các xe tải
        for t in trucks:
            for trip_idx, trip in enumerate(state.truck_trips[t]):
                # Logic tương tự như drone nhưng Xe tải không bị giới hạn Pin (dist_lim = inf)
                next_start = state.truck_trips[t][trip_idx+1].depart_time if trip_idx + 1 < len(state.truck_trips[t]) else float('inf')
                prev_return = state.truck_trips[t][trip_idx-1].return_time if trip_idx > 0 else 0.0
                
                for pos in range(len(trip.stops) + 1):
                    temp_stops = trip.stops[:pos] + [req] + trip.stops[pos:]
                    temp_depart = max(prev_return, max(r.r_i for r in temp_stops))
                    temp_trip = Trip(vtype='truck', vidx=t, stops=temp_stops, depart_time=temp_depart)
                    
                    if temp_trip.eval([0,0], t_vel, t_cap, float('inf'), state.lw):
                        if temp_trip.return_time <= next_start:
                            delta_dist = temp_trip.total_dist - trip.total_dist
                            candidates.append((delta_dist, 'truck', t, trip_idx, pos, temp_depart))

    # =========================================================================
    # --- KỊCH BẢN 2: THỬ TẠO MỘT CHUYẾN ĐI (TRIP) MỚI HOÀN TOÀN ---
    # =========================================================================
    
    # --- 2.1 Tạo chuyến mới cho Drone ---
    if req.can_drone and req.demand <= d_cap:
        drones = list(range(state.data["drone_num"]))
        random.shuffle(drones)
        for d in drones:
            # Một chuyến mới có thể đứng trước chuyến 1, giữa chuyến 1-2, hoặc sau chuyến cuối cùng.
            # Vì vậy ta duyệt trip_idx từ 0 đến len(trips)
            for trip_idx in range(len(state.drone_trips[d]) + 1):
                prev_return = state.drone_trips[d][trip_idx-1].return_time if trip_idx > 0 else 0.0
                next_start = state.drone_trips[d][trip_idx].depart_time if trip_idx < len(state.drone_trips[d]) else float('inf')
                
                # Chuyến mới khởi hành sớm nhất có thể
                new_trip = Trip(vtype='drone', vidx=d, stops=[req], depart_time=max(prev_return, req.r_i))
                
                if new_trip.eval([0,0], d_vel, d_cap, d_lim, state.lw):
                    if new_trip.return_time <= next_start:
                        # trip_idx = -1: Ký hiệu đây là một chuyến đi mới (Special Marker)
                        # pos: Chỉ số vị trí của CHUYẾN ĐI này trong danh sách trips của phương tiện
                        candidates.append((new_trip.total_dist, 'drone', d, -1, trip_idx, new_trip.depart_time))

    # --- 2.2 Tạo chuyến mới cho Xe tải ---
    if req.demand <= t_cap:
        trucks = list(range(state.data["truck_num"]))
        random.shuffle(trucks)
        for t in trucks:
            for trip_idx in range(len(state.truck_trips[t]) + 1):
                prev_return = state.truck_trips[t][trip_idx-1].return_time if trip_idx > 0 else 0.0
                next_start = state.truck_trips[t][trip_idx].depart_time if trip_idx < len(state.truck_trips[t]) else float('inf')
                
                new_trip = Trip(vtype='truck', vidx=t, stops=[req], depart_time=max(prev_return, req.r_i))
                if new_trip.eval([0,0], t_vel, t_cap, float('inf'), state.lw):
                    if new_trip.return_time <= next_start:
                        # trip_idx = -2: Ký hiệu đây là một chuyến đi mới của Xe tải
                        candidates.append((new_trip.total_dist, 'truck', t, -2, trip_idx, new_trip.depart_time))

    # --- BƯỚC CUỐI: LỌC VÀ SẮP XẾP PHƯƠNG ÁN TỐT NHẤT ---
    # Mục tiêu: Chọn phương án nào làm tăng tổng quãng đường ít nhất.
    def sort_key(cand):
        # cost: Chi phí tăng thêm (delta_dist hoặc total_dist của chuyến mới)
        cost, vtype, vidx, trip_idx, pos, depart = cand
        return cost

    candidates.sort(key=sort_key)
    return candidates


def solve_greedy(state: VRPState):
    """
    Thuật toán Tham lam cải tiến (Improved Greedy):
    Xử lý tất cả các đơn hàng chưa được phục vụ, tìm vị trí 'rẻ' nhất để chèn chúng vào.
    
    Quy trình:
    1. Sắp xếp đơn hàng theo thời điểm hàng sẵn sàng tại kho (Release time).
    2. Với mỗi đơn hàng, gọi get_insertion_candidates để tìm nơi chèn tốt nhất.
    3. Thực hiện chèn vào kịch bản có chi phí thấp nhất.
    """
    # Ưu tiên đơn hàng có deadline sớm (l_i) và ở xa depot
    def difficulty_score(r):
        dist = math.sqrt(r.x**2 + r.y**2)
        return r.l_i - (dist / 15.0) # Kết hợp deadline và thời gian di chuyển

    sorted_reqs = sorted(list(state.unserved), key=difficulty_score)
    
    # Lấy thông số phương diện
    t_vel, d_vel = state.data["truck_vel"], state.data["drone_vel"]
    t_cap, d_cap = state.data["truck_cap"], state.data["drone_cap"]
    d_lim = state.data["drone_lim"] * d_vel
    
    for req in sorted_reqs:
        # Tìm các phương án chèn khả thi
        candidates = get_insertion_candidates(state, req)
        
        if candidates:
            # Chọn phương án đầu tiên (có delta_dist thấp nhất vì đã được sort trong get_insertion_candidates)
            cost, vtype, vidx, trip_idx, pos, depart = candidates[0]
            
            if trip_idx < 0: # Trường hợp tạo chuyến đi mới
                new_trip = Trip(vtype=vtype, vidx=vidx, stops=[req], depart_time=depart)
                vel = d_vel if vtype == 'drone' else t_vel
                cap = d_cap if vtype == 'drone' else t_cap
                dist_lim = d_lim if vtype == 'drone' else float('inf')
                
                # Cập nhật thông số chuyến đi
                new_trip.eval([0,0], vel, cap, dist_lim, state.lw)
                if vtype == 'drone':
                    state.drone_trips[vidx].insert(pos, new_trip)
                else:
                    state.truck_trips[vidx].insert(pos, new_trip)
            else: # Trường hợp chèn vào chuyến đi đang có sẵn
                trip = (state.drone_trips[vidx][trip_idx] if vtype == 'drone' else state.truck_trips[vidx][trip_idx])
                trip.stops.insert(pos, req)
                trip.depart_time = depart
                vel = d_vel if vtype == 'drone' else t_vel
                cap = d_cap if vtype == 'drone' else t_cap
                dist_lim = d_lim if vtype == 'drone' else float('inf')
                
                # Tính toán lại chi phí và thời gian của chuyến đi sau khi chèn thêm điểm
                trip.eval([0,0], vel, cap, dist_lim, state.lw)
            
            # Xóa khỏi danh sách chờ
            state.unserved.remove(req)

    return state

if __name__ == "__main__":
    # 1. Khởi tạo trạng thái ban đầu với dữ liệu (DATA, LW, ALPHA, GAMMA từ vrp_core)
    state = VRPState(DATA, LW, ALPHA, GAMMA)
    
    # 2. Thực thi thuật toán tham lam để lập lịch trình
    print("Đang chạy thuật toán Tham lam cải tiến...")
    result_state = solve_greedy(state)
    
    # 3. Hiển thị kết quả thu được ra màn hình
    print_solution(result_state)
