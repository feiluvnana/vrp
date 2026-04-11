from vrp_core import (Request, Trip, VRPState, DATA, MAX_WAITING_TIME, ALPHA,
                       print_solution)
from greedy_solver import solve_greedy, get_insertion_candidates, MIN_SAFETY_MARGIN
import numpy as np
import random
import math
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

ALPHA = 1.0
GAMMA = 500000.0

def _clean(state: VRPState) -> VRPState:
    state.truck_trips = [[t for t in v if t.stops] for v in state.truck_trips]
    state.drone_trips = [[t for t in v if t.stops] for v in state.drone_trips]
    return state

def _get_served(state: VRPState):
    return [
        (trip, req)
        for v_trips in state.truck_trips + state.drone_trips
        for trip in v_trips
        for req in trip.stops
    ]

def _remove_request(state: VRPState, trip: Trip, req: Request):
    if req in trip.stops:
        trip.stops.remove(req)
        state.unserved.add(req)

def random_removal(state: VRPState, rng) -> VRPState:
    s = state.copy()
    served = _get_served(s)
    if not served:
        return s
    k = min(len(served), max(1, int(len(s.requests) * 0.3)))
    for idx in rng.choice(len(served), k, replace=False):
        _remove_request(s, *served[idx])
    return _clean(s)

def shaw_removal(state: VRPState, rng) -> VRPState:
    s = state.copy()
    served = _get_served(s)
    if not served:
        return s

    pivot = served[rng.randint(len(served))][1]
    max_dist = max(math.sqrt(r.x**2 + r.y**2) for _, r in served) or 1.0
    max_demand = max(r.demand for _, r in served) or 1.0

    scored = []
    for trip, req in served:
        spatial = math.sqrt((pivot.x - req.x)**2 + (pivot.y - req.y)**2) / max_dist
        temporal = abs(pivot.request_time - req.request_time) / 3600.0
        demand = abs(pivot.demand - req.demand) / max_demand
        score = 0.6 * spatial + 0.2 * temporal + 0.2 * demand
        scored.append((score, trip, req))

    scored.sort(key=lambda x: x[0])
    k = max(1, int(len(s.requests) * 0.4))
    for _ in range(min(k, len(scored))):
        idx = int(rng.random()**2 * len(scored))
        _, trip, req = scored.pop(idx)
        _remove_request(s, trip, req)
    return _clean(s)

def worst_removal(state: VRPState, rng) -> VRPState:
    s = state.copy()
    cfg = s.data
    costs = []

    for v_trips in s.truck_trips + s.drone_trips:
        for trip in v_trips:
            vel, cap, d_lim = (
                (cfg["drone_vel"], cfg["drone_cap"], cfg["drone_lim"] * cfg["drone_vel"])
                if trip.vehicle_type == "drone"
                else (cfg["truck_vel"], cfg["truck_cap"], float("inf"))
            )
            for i, req in enumerate(trip.stops):
                other_stops = trip.stops[:i] + trip.stops[i + 1:]
                if not other_stops:
                    savings = trip.total_dist
                else:
                    tmp = Trip(trip.vehicle_type, trip.vehicle_index,
                               other_stops, trip.depart_time)
                    if tmp.eval(s.dist_matrix, vel, cap, d_lim, s.max_wait_time):
                        savings = trip.total_dist - tmp.total_dist
                    else:
                        savings = 0.0
                costs.append((savings, trip, req))

    if not costs:
        return s

    costs.sort(key=lambda x: x[0], reverse=True)
    k = min(len(costs), max(1, int(len(s.requests) * 0.2)))
    for _ in range(k):
        idx = int(rng.random()**3 * len(costs))
        _, trip, req = costs.pop(idx)
        _remove_request(s, trip, req)
    return _clean(s)

def string_removal(state: VRPState, rng) -> VRPState:
    s = state.copy()
    trips = [t for v in s.truck_trips + s.drone_trips for t in v if t.stops]
    if not trips:
        return s

    trip = trips[rng.randint(len(trips))]
    n = len(trip.stops)
    size = rng.randint(1, min(n, 4) + 1)
    start = rng.randint(0, n - size + 1)

    for req in list(trip.stops[start:start + size]):
        _remove_request(s, trip, req)
    return _clean(s)

def greedy_repair(state: VRPState, rng) -> VRPState:
    return solve_greedy(state)

def regret_repair(state: VRPState, rng) -> VRPState:
    s = state.copy()
    cfg = s.data

    while s.unserved:
        best_regret = -1.0
        best_req = None
        best_cand = None

        for req in list(s.unserved):
            cands = get_insertion_candidates(s, req, safety_margin=MIN_SAFETY_MARGIN)
            if not cands:
                continue

            cost_1st = cands[0][0]
            cost_3rd = cands[2][0] if len(cands) > 2 else (
                cands[1][0] if len(cands) > 1 else GAMMA
            )
            regret = cost_3rd - cost_1st

            if regret > best_regret:
                best_regret = regret
                best_req = req
                best_cand = cands[0]

        if best_req is None:
            break

        cost, v_type, v_idx, t_idx, pos, dep = best_cand
        vel, cap, d_lim = (
            (cfg["drone_vel"], cfg["drone_cap"], cfg["drone_lim"] * cfg["drone_vel"])
            if v_type == "drone"
            else (cfg["truck_vel"], cfg["truck_cap"], float("inf"))
        )
        v_trips = s.drone_trips[v_idx] if v_type == "drone" else s.truck_trips[v_idx]

        if t_idx < 0:
            trip = Trip(v_type, v_idx, [best_req], dep)
            trip.eval(s.dist_matrix, vel, cap, d_lim, s.max_wait_time)
            v_trips.insert(pos, trip)
        else:
            trip = v_trips[t_idx]
            trip.stops.insert(pos, best_req)
            trip.depart_time = dep
            trip.eval(s.dist_matrix, vel, cap, d_lim, s.max_wait_time)

        s.unserved.remove(best_req)
    return s

def solve_alns(iterations=1000) -> VRPState:
    rng = np.random.RandomState(42)
    alns = ALNS(rng)

    for op in [random_removal, shaw_removal, worst_removal, string_removal]:
        alns.add_destroy_operator(op)
    for op in [greedy_repair, regret_repair]:
        alns.add_repair_operator(op)

    init = solve_greedy(VRPState(DATA, MAX_WAITING_TIME, ALPHA, GAMMA))
    init_obj = init.objective()

    select = RouletteWheel([50, 20, 5, 2], 0.8, 4, 1)
    start_temp = max(1.0, 0.3 * init_obj)
    accept = SimulatedAnnealing(start_temp, 1, 0.9997)
    stop = MaxIterations(iterations)

    result = alns.iterate(init, select, accept, stop)
    return result.best_state

if __name__ == "__main__":
    best = solve_alns(1000)
    print_solution(best, "ALNS OPTIMIZED SOLUTION")
