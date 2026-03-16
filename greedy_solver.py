from vrp_core import Request, DATA, LW, ALPHA, GAMMA, VRPState, Trip, print_solution
import random
import math

def get_insertion_candidates(state: VRPState, req: Request):
    """Find all valid insertion candidates for the given request."""
    t_vel = state.data["truck_vel"]
    d_vel = state.data["drone_vel"]
    t_cap = state.data["truck_cap"]
    d_cap = state.data["drone_cap"]
    d_lim = state.data["drone_lim"] * d_vel 
    
    candidates = []

    # 1. Insert into existing trips
    if req.can_drone and req.demand <= d_cap:
        drones = list(range(state.data["drone_num"]))
        random.shuffle(drones)
        for d in drones:
            for trip_idx, trip in enumerate(state.drone_trips[d]):
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

    if req.demand <= t_cap:
        trucks = list(range(state.data["truck_num"]))
        random.shuffle(trucks)
        for t in trucks:
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

    # 2. Create new trips
    if req.can_drone and req.demand <= d_cap:
        drones = list(range(state.data["drone_num"]))
        random.shuffle(drones)
        for d in drones:
            for trip_idx in range(len(state.drone_trips[d]) + 1):
                prev_return = state.drone_trips[d][trip_idx-1].return_time if trip_idx > 0 else 0.0
                next_start = state.drone_trips[d][trip_idx].depart_time if trip_idx < len(state.drone_trips[d]) else float('inf')
                
                new_trip = Trip(vtype='drone', vidx=d, stops=[req], depart_time=max(prev_return, req.r_i))
                
                if new_trip.eval([0,0], d_vel, d_cap, d_lim, state.lw):
                    if new_trip.return_time <= next_start:
                        candidates.append((new_trip.total_dist, 'drone', d, -1, trip_idx, new_trip.depart_time))

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
                        candidates.append((new_trip.total_dist, 'truck', t, -2, trip_idx, new_trip.depart_time))

    def sort_key(cand):
        return cand[0] # Sort by cost

    candidates.sort(key=sort_key)
    return candidates

def solve_greedy(state: VRPState):
    """Improved greedy solver to insert requests sequentially at cheapest positions."""
    def difficulty_score(r):
        dist = math.sqrt(r.x**2 + r.y**2)
        return r.l_i - (dist / 15.0)

    sorted_reqs = sorted(list(state.unserved), key=difficulty_score)
    
    t_vel, d_vel = state.data["truck_vel"], state.data["drone_vel"]
    t_cap, d_cap = state.data["truck_cap"], state.data["drone_cap"]
    d_lim = state.data["drone_lim"] * d_vel
    
    for req in sorted_reqs:
        candidates = get_insertion_candidates(state, req)
        if candidates:
            cost, vtype, vidx, trip_idx, pos, depart = candidates[0]
            
            if trip_idx < 0:
                new_trip = Trip(vtype=vtype, vidx=vidx, stops=[req], depart_time=depart)
                vel = d_vel if vtype == 'drone' else t_vel
                cap = d_cap if vtype == 'drone' else t_cap
                dist_lim = d_lim if vtype == 'drone' else float('inf')
                
                new_trip.eval([0,0], vel, cap, dist_lim, state.lw)
                if vtype == 'drone':
                    state.drone_trips[vidx].insert(pos, new_trip)
                else:
                    state.truck_trips[vidx].insert(pos, new_trip)
            else:
                trip = (state.drone_trips[vidx][trip_idx] if vtype == 'drone' else state.truck_trips[vidx][trip_idx])
                trip.stops.insert(pos, req)
                trip.depart_time = depart
                vel = d_vel if vtype == 'drone' else t_vel
                cap = d_cap if vtype == 'drone' else t_cap
                dist_lim = d_lim if vtype == 'drone' else float('inf')
                
                trip.eval([0,0], vel, cap, dist_lim, state.lw)
            
            state.unserved.remove(req)

    return state

if __name__ == "__main__":
    state = VRPState(DATA, LW, ALPHA, GAMMA)
    print("Running greedy solver...")
    result_state = solve_greedy(state)
    print_solution(result_state)
