from vrp_core import Request, Trip, VRPState, DATA, LW, ALPHA, GAMMA, print_solution
import math

# =============================================================================
# CÁC THÔNG SỐ CẢI TIẾN THEO GREEDY.PDF (SLACK-BASED GREEDY)
# =============================================================================
# BETA: Trọng số điều chỉnh sự cân bằng giữa quãng đường và độ linh hoạt.
# Nếu BETA cao, thuật toán sẽ hy sinh quãng đường để đổi lấy một lịch trình "lỏng" hơn.
BETA = 15.0      

# KAPPA & DELTA: Xác định "vùng an toàn" cho Slack.
# KAPPA = 0.1 nghĩa là chúng ta muốn giữ 10% độ rộng cửa sổ thời gian làm biên an toàn.
KAPPA = 0.1      
DELTA_MIN = 180.0 
DELTA_MAX = 720.0 

# LAMBDA (Smoothing Factor): Kiểm soát tốc độ thích ứng của hệ thống với sự thay đổi của dữ liệu.
# LAMBDA nhỏ (0.05) giúp ngưỡng an toàn delta không bị nhảy quá mạnh khi gặp các request bất thường.
LAMBDA = 0.05    

# RHO: Ngưỡng kinh tế. Nếu chi phí phục vụ một khách (sau khi đã cộng phạt slack) 
# lớn hơn RHO, ta sẽ để khách đó unserved để tối ưu hàm mục tiêu tổng thể.
RHO = 10000.0    

def get_insertion_candidates(state: VRPState, req: Request, delta: float):
    """
    Tìm tất cả các vị trí chèn khả thi cho một request và tính toán chi phí được cải tiến (Improved Cost).
    Theo Bước 2.2 và 2.3 trong greedy.pdf.
    """
    truck_conf = (state.data["truck_vel"], state.data["truck_cap"], float('inf'))
    drone_conf = (state.data["drone_vel"], state.data["drone_cap"], state.data["drone_lim"] * state.data["drone_vel"])
    
    candidates = []
    
    # Bước 2.1: Xác định các lựa chọn hợp lệ theo loại request (Truck only vs Flex)
    for v_type in (['drone', 'truck'] if req.can_drone else ['truck']):
        vel, cap, d_lim = drone_conf if v_type == 'drone' else truck_conf
        if req.demand > cap: continue
        
        v_list = state.drone_trips if v_type == 'drone' else state.truck_trips
        for v_idx, trips in enumerate(v_list):
            
            # Lựa chọn 1: Chèn vào một hành trình (trip/tour) ĐÃ CÓ
            for t_idx, trip in enumerate(trips):
                # Ràng buộc thời gian bận: không được chồng lấn với chuyến tiếp theo
                next_start = trips[t_idx+1].depart_time if t_idx+1 < len(trips) else float('inf')
                prev_return = trips[t_idx-1].return_time if t_idx > 0 else 0.0
                
                for i_pos in range(len(trip.stops) + 1):
                    # Sinh tập ứng viên chèn (Candidate insertions)
                    stops = trip.stops[:i_pos] + [req] + trip.stops[i_pos:]
                    dep = max(prev_return, max(r.request_time for r in stops))
                    t_trip = Trip(v_type, v_idx, stops, dep)
                    
                    # Bước 2.3: Kiểm tra feasibility (khả thi)
                    # Phương thức eval() kiểm tra: dung lượng, pin/bán kính, cửa sổ thời gian, waiting time.
                    if t_trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time) and t_trip.return_time <= next_start:
                        
                        # Bước 2.4: Tính Incremental Cost cải tiến với Slack Penalty
                        # dJ = delta_distance + penalty(slack)
                        delta_dist = t_trip.total_dist - trip.total_dist
                        
                        # Tính Min-slack của tour sau khi chèn: slack_i = l_i - S_i
                        min_slack = min(r.latest_end - t_trip.service_times[r.id] for r in t_trip.stops)
                        
                        # Risk score (penalty): beta * max(0, delta - SlackMin)
                        slack_penalty = BETA * max(0, delta - min_slack)
                        
                        improved_cost = delta_dist + slack_penalty
                        candidates.append((improved_cost, v_type, v_idx, t_idx, i_pos, dep))

            # Lựa chọn 2: Tạo một hành trình MỚI (New tour/sortie)
            for t_idx in range(len(trips) + 1):
                prev_return = trips[t_idx-1].return_time if t_idx > 0 else 0.0
                next_start = trips[t_idx].depart_time if t_idx < len(trips) else float('inf')
                
                new_t = Trip(v_type, v_idx, [req], max(prev_return, req.request_time))
                if new_t.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time) and new_t.return_time <= next_start:
                    
                    # Tính slack cho hành trình mới (chỉ có 1 khách)
                    slack_r = req.latest_end - new_t.service_times[req.id]
                    slack_penalty = BETA * max(0, delta - slack_r)
                    
                    improved_cost = new_t.total_dist + slack_penalty
                    candidates.append((improved_cost, v_type, v_idx, -1 - t_idx, t_idx, new_t.depart_time))

    # Sắp xếp theo chi phí chèn cải tiến tốt nhất
    return sorted(candidates, key=lambda x: x[0])

def solve_greedy(state: VRPState):
    """
    Giải bài toán VRP bằng thuật toán Greedy cải tiến (Slack-based Insertion).
    Tích hợp EWMA để cập nhật ngưỡng an toàn slack động.
    """
    # Sắp xếp request theo độ ưu tiên (earliest start/latest end vs distance to depot)
    sorted_reqs = sorted(list(state.unserved), key=lambda r: r.latest_end - (math.sqrt(r.x**2 + r.y**2) / 15.0))
    
    m_avg_tw = 0.0 # Biến trạng thái m: ước lượng trung bình trượt (EWMA) của độ rộng time window
    
    for req in sorted_reqs:
        # Cập nhật m theo EWMA (Bước 2.4 trong greedy.pdf)
        w_r = req.latest_end - req.earliest_start
        if m_avg_tw == 0:
            m_avg_tw = w_r # Khởi tạo bằng request đầu tiên
        else:
            m_avg_tw = (1 - LAMBDA) * m_avg_tw + LAMBDA * w_r
        
        # Tính ngưỡng an toàn delta(t) = min(delta_max, max(delta_min, kappa * m))
        delta_t = min(DELTA_MAX, max(DELTA_MIN, KAPPA * m_avg_tw))
        
        # Tìm các ứng viên chèn với delta_t hiện tại
        cands = get_insertion_candidates(state, req, delta_t)
        
        if not cands:
            # (a) Bắt buộc reject request: không có phương án khả thi
            continue
        
        # Lấy ứng viên tốt nhất (chi phí thấp nhất)
        best_cost, v_type, v_idx, t_idx, i_pos, dep = cands[0]
        
        # (b) Tự chọn reject (Optional): Nếu chi phí tốt nhất vượt ngưỡng rho
        if best_cost > RHO:
            continue
            
        v_trips = state.drone_trips[v_idx] if v_type == 'drone' else state.truck_trips[v_idx]
        conf = state.data
        vel, cap, d_lim = (conf["drone_vel"], conf["drone_cap"], conf["drone_lim"]*conf["drone_vel"]) if v_type == 'drone' else (conf["truck_vel"], conf["truck_cap"], float('inf'))

        if t_idx < 0:
            # Tạo hành trình mới
            new_t = Trip(v_type, v_idx, [req], dep)
            new_t.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time)
            v_trips.insert(i_pos, new_t)
        else:
            # Chèn vào hành trình hiện có
            trip = v_trips[t_idx]
            trip.stops.insert(i_pos, req)
            trip.depart_time = dep
            trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time)
        
        state.unserved.remove(req)
        
    return state

if __name__ == "__main__":
    result = solve_greedy(VRPState(DATA, LW, ALPHA, GAMMA))
    print_solution(result)
