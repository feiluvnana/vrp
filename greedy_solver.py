from vrp_core import Request, Trip, VRPState, DATA, LW, ALPHA, GAMMA, print_solution
import math

def get_insertion_candidates(state: VRPState, req: Request):
    """Exhaustively find all valid insertion points for a request."""
    # Pre-cache vehicle configurations to avoid dictionary overhead
    truck_conf = (state.data["truck_vel"], state.data["truck_cap"], float('inf'))
    drone_conf = (state.data["drone_vel"], state.data["drone_cap"], state.data["drone_lim"] * state.data["drone_vel"])
    
    candidates = []
    
    # Iterate through allowed vehicle types for this specific request
    for v_type in (['drone', 'truck'] if req.can_drone else ['truck']):
        vel, cap, d_lim = drone_conf if v_type == 'drone' else truck_conf
        if req.demand > cap: continue
        
        v_list = state.drone_trips if v_type == 'drone' else state.truck_trips
        for v_idx, trips in enumerate(v_list):
            
            # OPTION 1: Insert into an EXISTING trip
            for t_idx, trip in enumerate(trips):
                # Busy-period constraints: cannot overlap with next trip or finish before previous one
                next_start = trips[t_idx+1].depart_time if t_idx+1 < len(trips) else float('inf')
                prev_return = trips[t_idx-1].return_time if t_idx > 0 else 0.0
                
                for i_pos in range(len(trip.stops) + 1):
                    stops = trip.stops[:i_pos] + [req] + trip.stops[i_pos:]
                    # Vehicle waits for both warehouse return and request existence
                    dep = max(prev_return, max(r.request_time for r in stops))
                    t_trip = Trip(v_type, v_idx, stops, dep)
                    
                    if t_trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time) and t_trip.return_time <= next_start:
                        candidates.append((t_trip.total_dist - trip.total_dist, v_type, v_idx, t_idx, i_pos, dep))

            # OPTION 2: Create a BRAND NEW trip between existing trips
            for t_idx in range(len(trips) + 1):
                prev_return = trips[t_idx-1].return_time if t_idx > 0 else 0.0
                next_start = trips[t_idx].depart_time if t_idx < len(trips) else float('inf')
                
                new_t = Trip(v_type, v_idx, [req], max(prev_return, req.request_time))
                if new_t.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time) and new_t.return_time <= next_start:
                    # trip_idx < 0 signals a NEW trip creation to the solver
                    candidates.append((new_t.total_dist, v_type, v_idx, -1 - t_idx, t_idx, new_t.depart_time))

    # Sort by cheapest distance increase
    return sorted(candidates, key=lambda x: x[0])

def solve_greedy(state: VRPState):
    """Initial solution generation using Cheapest Insertion Heuristic."""
    # Urgency sorting: distance to depot (math.sqrt part) weighed against latest deadlines
    sorted_reqs = sorted(list(state.unserved), key=lambda r: r.latest_end - (math.sqrt(r.x**2 + r.y**2) / 15.0))
    
    for req in sorted_reqs:
        cands = get_insertion_candidates(state, req)
        if not cands: continue
        
        cost, v_type, v_idx, t_idx, i_pos, dep = cands[0]
        v_trips = state.drone_trips[v_idx] if v_type == 'drone' else state.truck_trips[v_idx]
        conf = state.data
        vel, cap, d_lim = (conf["drone_vel"], conf["drone_cap"], conf["drone_lim"]*conf["drone_vel"]) if v_type == 'drone' else (conf["truck_vel"], conf["truck_cap"], float('inf'))

        if t_idx < 0:
            new_t = Trip(v_type, v_idx, [req], dep)
            new_t.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time)
            v_trips.insert(i_pos, new_t)
        else:
            trip = v_trips[t_idx]
            trip.stops.insert(i_pos, req)
            trip.depart_time = dep
            trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time)
        
        state.unserved.remove(req)
    return state

if __name__ == "__main__":
    result = solve_greedy(VRPState(DATA, LW, ALPHA, GAMMA))
    print_solution(result)
