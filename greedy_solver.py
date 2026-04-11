from vrp_core import (Request, Trip, VRPState, DATA, MAX_WAITING_TIME, ALPHA,
                       print_solution)
import math

TRUCK_COST_WEIGHT = 1.0 
DRONE_COST_WEIGHT = 1.0
SLACK_PENALTY_WEIGHT = 15.0 
SLACK_MARGIN_RATIO = 0.1 
MIN_SAFETY_MARGIN = 180.0
MAX_SAFETY_MARGIN = 720.0 
EWMA_UPDATE_RATE = 0.05 
MAX_REJECTION_COST = float("inf") 

def get_insertion_candidates(state: VRPState, req: Request, safety_margin: float):
    truck_cfg = (state.data["truck_vel"], state.data["truck_cap"], float("inf"))
    drone_cfg = (state.data["drone_vel"], state.data["drone_cap"],
                 state.data["drone_lim"] * state.data["drone_vel"])
    candidates = []
    v_types = ["drone", "truck"] if req.can_drone else ["truck"]

    for v_type in v_types:
        vel, cap, d_lim = drone_cfg if v_type == "drone" else truck_cfg
        if req.demand > cap:
            continue
            
        cost_weight = DRONE_COST_WEIGHT if v_type == "drone" else TRUCK_COST_WEIGHT
        v_list = state.drone_trips if v_type == "drone" else state.truck_trips

        for v_idx, trips in enumerate(v_list):
            # --- Insert into existing trips ---
            for t_idx, trip in enumerate(trips):
                prev_ret = trips[t_idx - 1].return_time if t_idx > 0 else 0.0
                next_dep = trips[t_idx + 1].depart_time if t_idx + 1 < len(trips) else float("inf")
                for pos in range(len(trip.stops) + 1):
                    new_stops = trip.stops[:pos] + [req] + trip.stops[pos:]
                    dep = max(prev_ret, max(r.request_time for r in new_stops))
                    trial = Trip(v_type, v_idx, new_stops, dep)

                    if not trial.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time):
                        continue
                    if trial.return_time > next_dep:
                        continue

                    delta_dist = cost_weight * (trial.total_dist - trip.total_dist)
                    min_slack = min(
                        r.time_window_end - trial.service_times[r.id]
                        for r in trial.stops
                    )
                    slack_penalty = SLACK_PENALTY_WEIGHT * max(0.0, safety_margin - min_slack)
                    cost = delta_dist + slack_penalty
                    candidates.append((cost, v_type, v_idx, t_idx, pos, dep))

            # --- Create new trips ---
            for t_pos in range(len(trips) + 1):
                prev_ret = trips[t_pos - 1].return_time if t_pos > 0 else 0.0
                next_dep = trips[t_pos].depart_time if t_pos < len(trips) else float("inf")
                new_trip = Trip(v_type, v_idx, [req], max(prev_ret, req.request_time))

                if not new_trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time):
                    continue
                if new_trip.return_time > next_dep:
                    continue

                slack_r = req.time_window_end - new_trip.service_times[req.id]
                slack_penalty = SLACK_PENALTY_WEIGHT * max(0.0, safety_margin - slack_r)
                cost = cost_weight * new_trip.total_dist + slack_penalty
                candidates.append((cost, v_type, v_idx, -(1 + t_pos), t_pos, new_trip.depart_time))

    candidates.sort(key=lambda c: c[0])
    return candidates

def solve_greedy(state: VRPState) -> VRPState:
    # --- Sort requests ---
    sorted_reqs = sorted(
        state.unserved,
        key=lambda r: r.time_window_end - math.sqrt(r.x**2 + r.y**2) / 15.0,
    )
    avg_window_width = 0.0 

    for req in sorted_reqs:
        # --- Update EWMA ---
        window_width = req.time_window_end - req.time_window_start
        if avg_window_width == 0.0:
            avg_window_width = window_width 
        else:
            avg_window_width = (1 - EWMA_UPDATE_RATE) * avg_window_width + EWMA_UPDATE_RATE * window_width
            
        current_safety_margin = min(MAX_SAFETY_MARGIN, max(MIN_SAFETY_MARGIN, SLACK_MARGIN_RATIO * avg_window_width))

        # --- Generate candidates ---
        cands = get_insertion_candidates(state, req, current_safety_margin)
        if not cands:
            continue

        best_cost, v_type, v_idx, t_idx, pos, dep = cands[0]
        if best_cost > MAX_REJECTION_COST:
            continue

        # --- Execute insertion ---
        v_trips = state.drone_trips[v_idx] if v_type == "drone" else state.truck_trips[v_idx]
        cfg = state.data
        vel, cap, d_lim = (
            (cfg["drone_vel"], cfg["drone_cap"], cfg["drone_lim"] * cfg["drone_vel"])
            if v_type == "drone"
            else (cfg["truck_vel"], cfg["truck_cap"], float("inf"))
        )

        if t_idx < 0:
            new_trip = Trip(v_type, v_idx, [req], dep)
            new_trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time)
            v_trips.insert(pos, new_trip)
        else:
            trip = v_trips[t_idx]
            trip.stops.insert(pos, req)
            trip.depart_time = dep
            trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time)
            
        state.unserved.remove(req)

    return state

if __name__ == "__main__":
    result = solve_greedy(VRPState(DATA, MAX_WAITING_TIME, ALPHA, 500000.0))
    print_solution(result, "GREEDY SOLUTION")
