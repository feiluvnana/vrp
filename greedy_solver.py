from vrp_core import DATA, LW, ALPHA, GAMMA, VRPState, Trip

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
    
    for req in sorted_reqs:
        assigned = False
        candidates = [] # (cost, vtype, vidx, trip_idx, pos, new_depart)

        # 1. Thử chèn vào TẤT CẢ các chuyến hiện có
        # Drone
        if req.can_drone and req.demand <= d_cap:
            for d in range(state.data["drone_num"]):
                for trip_idx, trip in enumerate(state.drone_trips[d]):
                    # Đảm bảo chuyến hiện tại về kịp để chuyến tiếp theo bắt đầu
                    next_start = state.drone_trips[d][trip_idx+1].depart_time if trip_idx + 1 < len(state.drone_trips[d]) else float('inf')
                    prev_return = state.drone_trips[d][trip_idx-1].return_time if trip_idx > 0 else 0.0
                    
                    for pos in range(len(trip.stops) + 1):
                        temp_stops = trip.stops[:pos] + [req] + trip.stops[pos:]
                        temp_depart = max(prev_return, max(r.r_i for r in temp_stops))
                        temp_trip = Trip(vtype='drone', vidx=d, stops=temp_stops, depart_time=temp_depart)
                        if temp_trip.eval([0,0], d_vel, d_cap, d_lim, state.lw):
                            if temp_trip.return_time <= next_start:
                                delta_dist = temp_trip.total_dist - trip.total_dist
                                candidates.append((delta_dist, 'drone', d, trip_idx, pos, temp_depart))

        # Xe tải
        if req.demand <= t_cap:
            for t in range(state.data["truck_num"]):
                for trip_idx, trip in enumerate(state.truck_trips[t]):
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

        # 2. Thử tạo chuyến mới
        # Drone mới
        if req.can_drone and req.demand <= d_cap:
            for d in range(state.data["drone_num"]):
                # Chèn chuyến mới vào giữa các chuyến hiện có
                for trip_idx in range(len(state.drone_trips[d]) + 1):
                    prev_return = state.drone_trips[d][trip_idx-1].return_time if trip_idx > 0 else 0.0
                    next_start = state.drone_trips[d][trip_idx].depart_time if trip_idx < len(state.drone_trips[d]) else float('inf')
                    
                    new_trip = Trip(vtype='drone', vidx=d, stops=[req], depart_time=max(prev_return, req.r_i))
                    if new_trip.eval([0,0], d_vel, d_cap, d_lim, state.lw):
                        if new_trip.return_time <= next_start:
                            candidates.append((new_trip.total_dist, 'drone', d, -1, trip_idx, new_trip.depart_time))

        # Xe tải mới
        if req.demand <= t_cap:
            for t in range(state.data["truck_num"]):
                for trip_idx in range(len(state.truck_trips[t]) + 1):
                    prev_return = state.truck_trips[t][trip_idx-1].return_time if trip_idx > 0 else 0.0
                    next_start = state.truck_trips[t][trip_idx].depart_time if trip_idx < len(state.truck_trips[t]) else float('inf')
                    
                    new_trip = Trip(vtype='truck', vidx=t, stops=[req], depart_time=max(prev_return, req.r_i))
                    if new_trip.eval([0,0], t_vel, t_cap, float('inf'), state.lw):
                        if new_trip.return_time <= next_start:
                            candidates.append((new_trip.total_dist, 'truck', t, -2, trip_idx, new_trip.depart_time))

        if candidates:
            # Ưu tiên các phương án có chi phí tăng thêm thấp nhất
            candidates.sort(key=lambda x: x[0])
            cost, vtype, vidx, trip_idx, pos, depart = candidates[0]
            
            if trip_idx < 0: # Chuyến mới
                new_trip = Trip(vtype=vtype, vidx=vidx, stops=[req], depart_time=depart)
                vel = d_vel if vtype == 'drone' else t_vel
                cap = d_cap if vtype == 'drone' else t_cap
                dist_lim = d_lim if vtype == 'drone' else float('inf')
                new_trip.eval([0,0], vel, cap, dist_lim, state.lw)
                if vtype == 'drone':
                    state.drone_trips[vidx].insert(pos, new_trip)
                else:
                    state.truck_trips[vidx].insert(pos, new_trip)
            else: # Chuyến hiện có
                trip = (state.drone_trips[vidx][trip_idx] if vtype == 'drone' else state.truck_trips[vidx][trip_idx])
                trip.stops.insert(pos, req)
                trip.depart_time = depart
                vel = d_vel if vtype == 'drone' else t_vel
                cap = d_cap if vtype == 'drone' else t_cap
                dist_lim = d_lim if vtype == 'drone' else float('inf')
                trip.eval([0,0], vel, cap, dist_lim, state.lw)
            
            state.unserved.remove(req)
            assigned = True

    return state

def print_solution(state: VRPState):
    """In tóm tắt kết quả lời giải VRP."""
    print("\n" + "="*50)
    print("    TÓM TẮT KẾT QUẢ THAM LAM (GREEDY)")
    print("="*50)
    print(f"Tổng số đơn hàng: {len(state.requests)}")
    print(f"Đã phục vụ     : {len(state.requests) - len(state.unserved)}")
    print(f"Chi phí mục tiêu: {state.objective():.2f}")
    
    print("\nCHUYẾN ĐI DRONE:")
    for d, trips in enumerate(state.drone_trips):
        for idx, t in enumerate(trips):
            print(f"  Drone {d} Chuyến {idx}: Điểm {[s.id for s in t.stops]} | Quãng đường {t.total_dist:.0f}m | Về kho lúc {t.return_time:.0f}s")
            
    print("\nCHUYẾN ĐI XE TẢI:")
    for t_idx, trips in enumerate(state.truck_trips):
        for idx, t in enumerate(trips):
            print(f"  Xe tải {t_idx} Chuyến {idx}: Điểm {[s.id for s in t.stops]} | Quãng đường {t.total_dist:.0f}m | Về kho lúc {t.return_time:.0f}s")
    print("="*50 + "\n")

if __name__ == "__main__":
    # 1. Khởi tạo trạng thái với dữ liệu bài toán
    state = VRPState(DATA, LW, ALPHA, GAMMA)
    
    # 2. Chạy thuật toán tham lam cải tiến
    print("Đang chạy thuật toán Tham lam cải tiến...")
    result_state = solve_greedy(state)
    
    # 3. Xuất kết quả cuối cùng
    print_solution(result_state)
