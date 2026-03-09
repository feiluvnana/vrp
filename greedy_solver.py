"""
Module này triển khai thuật toán Tham lam cải tiến (Improved Greedy) để giải bài toán VRP.
Thuật toán tìm cách chèn từng đơn hàng vào vị trí tối ưu nhất (tăng chi phí ít nhất) 
trong lịch trình hiện có của các phương tiện (xe tải và drone).
"""
# Nhập các hằng số và lớp cơ sở từ module vrp_core
from vrp_core import DATA, LW, ALPHA, GAMMA, VRPState, Trip, print_solution

def solve_greedy(state: VRPState):
    """
    Chiến lược Tham lam cải tiến:
    1. Sắp xếp tất cả đơn hàng theo thời điểm có sẵn (release time r_i).
    2. Với mỗi đơn hàng, thử TẤT CẢ các vị trí chèn khả thi trong TẤT CẢ
       các chuyến đi hiện có của loại phương tiện tương ứng.
    3. Đồng thời thử tạo một chuyến đi MỚI tại bất kỳ thời điểm nào trong lịch trình.
    4. Chọn phương án làm tăng tổng quãng đường hệ thống (chi phí) ít nhất.
    
    Phương pháp này mạnh hơn nhiều so với tham lam đơn giản ban đầu (chỉ thử thêm vào cuối).
    """
    # Sắp xếp để xử lý tuần tự
    sorted_reqs = sorted(list(state.unserved), key=lambda r: r.r_i)
    
    t_vel = state.data["truck_vel"]
    d_vel = state.data["drone_vel"]
    t_cap = state.data["truck_cap"]
    d_cap = state.data["drone_cap"]
    d_lim = state.data["drone_lim"] * d_vel
    
    # Duyệt qua từng đơn hàng đã được sắp xếp
    for req in sorted_reqs:
        assigned = False
        # Danh sách lưu các phương án chèn khả thi: (chi phí tăng thêm, loại xe, chỉ số xe, chỉ số chuyến, vị trí chèn, thời gian khởi hành mới)
        candidates = [] 

        # 1. THỬ CHÈN VÀO TẤT CẢ CÁC CHUYẾN ĐI HIỆN CÓ
        
        # --- Thử với Drone ---
        if req.can_drone and req.demand <= d_cap:
            for d in range(state.data["drone_num"]):
                for trip_idx, trip in enumerate(state.drone_trips[d]):
                    # Xác định thời điểm sớm nhất có thể đi (sau khi chuyến trước về) 
                    # và thời điểm muộn nhất phải về (trước khi chuyến sau bắt đầu)
                    next_start = state.drone_trips[d][trip_idx+1].depart_time if trip_idx + 1 < len(state.drone_trips[d]) else float('inf')
                    prev_return = state.drone_trips[d][trip_idx-1].return_time if trip_idx > 0 else 0.0
                    
                    # Thử chèn vào mọi vị trí pos trong danh sách các điểm dừng của chuyến trip
                    for pos in range(len(trip.stops) + 1):
                        temp_stops = trip.stops[:pos] + [req] + trip.stops[pos:]
                        # Thời gian khởi hành mới: phải sau khi tất cả hàng trong chuyến sẵn sàng và sau khi xe về kho
                        temp_depart = max(prev_return, max(r.r_i for r in temp_stops))
                        temp_trip = Trip(vtype='drone', vidx=d, stops=temp_stops, depart_time=temp_depart)
                        
                        # Kiểm tra xem chuyến đi mới có vi phạm ràng buộc (tải trọng, pin, khoảng cách) không
                        if temp_trip.eval([0,0], d_vel, d_cap, d_lim, state.lw):
                            # Kiểm tra xem có về kịp để thực hiện chuyến tiếp theo đã lên lịch không
                            if temp_trip.return_time <= next_start:
                                delta_dist = temp_trip.total_dist - trip.total_dist
                                candidates.append((delta_dist, 'drone', d, trip_idx, pos, temp_depart))

        # --- Thử với Xe tải ---
        if req.demand <= t_cap:
            for t in range(state.data["truck_num"]):
                for trip_idx, trip in enumerate(state.truck_trips[t]):
                    next_start = state.truck_trips[t][trip_idx+1].depart_time if trip_idx + 1 < len(state.truck_trips[t]) else float('inf')
                    prev_return = state.truck_trips[t][trip_idx-1].return_time if trip_idx > 0 else 0.0
                    
                    for pos in range(len(trip.stops) + 1):
                        temp_stops = trip.stops[:pos] + [req] + trip.stops[pos:]
                        temp_depart = max(prev_return, max(r.r_i for r in temp_stops))
                        temp_trip = Trip(vtype='truck', vidx=t, stops=temp_stops, depart_time=temp_depart)
                        
                        # Xe tải không bị giới hạn quãng đường pin (truyền float('inf') vào dist_lim)
                        if temp_trip.eval([0,0], t_vel, t_cap, float('inf'), state.lw):
                            if temp_trip.return_time <= next_start:
                                delta_dist = temp_trip.total_dist - trip.total_dist
                                candidates.append((delta_dist, 'truck', t, trip_idx, pos, temp_depart))

        # 2. THỬ TẠO MỘT CHUYẾN ĐI MỚI (CHỈ CHỨA DUY NHẤT ĐƠN HÀNG NÀY)
        
        # --- Drone mới ---
        if req.can_drone and req.demand <= d_cap:
            for d in range(state.data["drone_num"]):
                # Có thể chèn chuyến mới vào trước, giữa hoặc sau các chuyến hiện có của drone d
                for trip_idx in range(len(state.drone_trips[d]) + 1):
                    prev_return = state.drone_trips[d][trip_idx-1].return_time if trip_idx > 0 else 0.0
                    next_start = state.drone_trips[d][trip_idx].depart_time if trip_idx < len(state.drone_trips[d]) else float('inf')
                    
                    new_trip = Trip(vtype='drone', vidx=d, stops=[req], depart_time=max(prev_return, req.r_i))
                    if new_trip.eval([0,0], d_vel, d_cap, d_lim, state.lw):
                        if new_trip.return_time <= next_start:
                            # Với chuyến mới, toàn bộ tổng quãng đường của nó chính là "chi phí tăng thêm"
                            candidates.append((new_trip.total_dist, 'drone', d, -1, trip_idx, new_trip.depart_time))

        # --- Xe tải mới ---
        if req.demand <= t_cap:
            for t in range(state.data["truck_num"]):
                for trip_idx in range(len(state.truck_trips[t]) + 1):
                    prev_return = state.truck_trips[t][trip_idx-1].return_time if trip_idx > 0 else 0.0
                    next_start = state.truck_trips[t][trip_idx].depart_time if trip_idx < len(state.truck_trips[t]) else float('inf')
                    
                    new_trip = Trip(vtype='truck', vidx=t, stops=[req], depart_time=max(prev_return, req.r_i))
                    if new_trip.eval([0,0], t_vel, t_cap, float('inf'), state.lw):
                        if new_trip.return_time <= next_start:
                            candidates.append((new_trip.total_dist, 'truck', t, -2, trip_idx, new_trip.depart_time))

        # 3. CHỌN PHƯƠNG ÁN TỐT NHẤT TRONG CÁC ỨNG VIÊN
        if candidates:
            # Sắp xếp các ứng viên để chọn phương án làm tăng tổng quãng đường ít nhất
            candidates.sort(key=lambda x: x[0])
            cost, vtype, vidx, trip_idx, pos, depart = candidates[0]
            
            if trip_idx < 0: # Trường hợp tạo chuyến mới (trip_idx là -1 hoặc -2)
                new_trip = Trip(vtype=vtype, vidx=vidx, stops=[req], depart_time=depart)
                vel = d_vel if vtype == 'drone' else t_vel
                cap = d_cap if vtype == 'drone' else t_cap
                dist_lim = d_lim if vtype == 'drone' else float('inf')
                # Tính toán lại các thông số chính xác của chuyến đi mới
                new_trip.eval([0,0], vel, cap, dist_lim, state.lw)
                if vtype == 'drone':
                    state.drone_trips[vidx].insert(pos, new_trip)
                else:
                    state.truck_trips[vidx].insert(pos, new_trip)
            else: # Trường hợp chèn vào chuyến hiện có
                trip = (state.drone_trips[vidx][trip_idx] if vtype == 'drone' else state.truck_trips[vidx][trip_idx])
                trip.stops.insert(pos, req)
                trip.depart_time = depart
                vel = d_vel if vtype == 'drone' else t_vel
                cap = d_cap if vtype == 'drone' else t_cap
                dist_lim = d_lim if vtype == 'drone' else float('inf')
                # Cập nhật lại quãng đường và thời gian về của chuyến sau khi chèn thêm điểm
                trip.eval([0,0], vel, cap, dist_lim, state.lw)
            
            # Đánh dấu đơn hàng đã được phục vụ
            state.unserved.remove(req)
            assigned = True

    return state

if __name__ == "__main__":
    # 1. Khởi tạo trạng thái ban đầu với dữ liệu (DATA, LW, ALPHA, GAMMA từ vrp_core)
    state = VRPState(DATA, LW, ALPHA, GAMMA)
    
    # 2. Thực thi thuật toán tham lam để lập lịch trình
    print("Đang chạy thuật toán Tham lam cải tiến...")
    result_state = solve_greedy(state)
    
    # 3. Hiển thị kết quả thu được ra màn hình
    print_solution(result_state)
