from vrp_core import Request, Trip, VRPState, DATA, LW, ALPHA, GAMMA, print_solution
from greedy_solver import solve_greedy, get_insertion_candidates
import numpy as np
import random
import math
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

def clean_state(state):
    """Purge empty trips from the solution to keep the search space clean."""
    state.truck_trips = [[t for t in v if t.stops] for v in state.truck_trips]
    state.drone_trips = [[t for t in v if t.stops] for v in state.drone_trips]
    return state

def random_removal(state: VRPState, random_state):
    """Destruction: Removes ~30% of served requests at random."""
    new_state = state.copy()
    all_served = [(t, r) for v in new_state.truck_trips + new_state.drone_trips for t in v for r in t.stops]
    if not all_served: return new_state
    
    k = min(len(all_served), max(1, int(len(new_state.requests) * 0.3)))
    for idx in random_state.choice(len(all_served), k, replace=False):
        trip, req = all_served[idx]
        trip.stops.remove(req)
        new_state.unserved.add(req)
    return clean_state(new_state)

def shaw_removal(state: VRPState, random_state):
    """Destruction: Removes orders that are 'similar' (clustered by space and time availability)."""
    new_state = state.copy()
    all_reqs = [r for v in new_state.truck_trips + new_state.drone_trips for t in v for r in t.stops]
    if not all_reqs: return new_state
    
    # Select a random anchor and find its neighbors
    pivot = random_state.choice(all_reqs)
    all_reqs.sort(key=lambda r: math.sqrt((pivot.x-r.x)**2 + (pivot.y-r.y)**2) + 0.1 * abs(pivot.request_time - r.request_time))
    
    k = max(1, int(len(new_state.requests) * 0.4))
    for req in all_reqs[:k]:
        for v in new_state.truck_trips + new_state.drone_trips:
            for t in v:
                if req in t.stops:
                    t.stops.remove(req)
                    new_state.unserved.add(req)
    return clean_state(new_state)

def worst_removal(state: VRPState, random_state):
    """Destruction: Removes orders that cause the highest distance increase in their current trips."""
    new_state = state.copy()
    costs, conf = [], new_state.data
    
    for v in new_state.truck_trips + new_state.drone_trips:
        for t in v:
            vel, cap, d_lim = (conf["drone_vel"], conf["drone_cap"], conf["drone_lim"]*conf["drone_vel"]) if t.vehicle_type == 'drone' else (conf["truck_vel"], conf["truck_cap"], float('inf'))
            for i, req in enumerate(t.stops):
                temp_stops = t.stops[:i] + t.stops[i+1:]
                if not temp_stops: loss = t.total_dist
                else:
                    tmp = Trip(t.vehicle_type, t.vehicle_index, temp_stops, t.depart_time)
                    loss = t.total_dist - tmp.total_dist if tmp.eval(new_state.dist_matrix, vel, cap, d_lim, state.max_wait_time) else 0
                costs.append({'t': t, 'r': req, 'cost': loss})
    
    if not costs: return new_state
    costs.sort(key=lambda x: x['cost'], reverse=True)
    k = min(len(costs), max(1, int(len(new_state.requests) * 0.2)))
    
    # Use random^3 to bias selection towards 'worst' items while keeping some diversity
    for _ in range(k):
        item = costs.pop(int(random_state.random()**3 * len(costs)))
        if item['r'] in item['t'].stops:
            item['t'].stops.remove(item['r'])
            new_state.unserved.add(item['r'])
    return clean_state(new_state)

def string_removal(state: VRPState, random_state):
    """Destruction: Removes a contiguous sequence of stops from a single trip."""
    new_state = state.copy()
    trips = [t for v in new_state.truck_trips + new_state.drone_trips for t in v if t.stops]
    if not trips: return new_state
    
    trip = random_state.choice(trips)
    size = random_state.randint(1, min(len(trip.stops), 4) + 1)
    start = random_state.randint(0, len(trip.stops) - size + 1)
    
    for req in list(trip.stops[start:start+size]):
        trip.stops.remove(req)
        new_state.unserved.add(req)
    return clean_state(new_state)

def greedy_repair(state: VRPState, random_state):
    """Repair: Simply re-insert removed requests using the greedy heuristic."""
    return solve_greedy(state)

def regret_repair(state: VRPState, random_state):
    """Repair: Inserts the request with the highest 'regret' (Best_cost - Second_best_cost)."""
    new_state = state.copy()
    conf = new_state.data
    
    while new_state.unserved:
        best_regret, best_req, best_cand = -1.0, None, None
        for req in list(new_state.unserved):
            cands = get_insertion_candidates(new_state, req)
            if not cands: continue
            # Regret measures the loss if we DON'T pick the best insertion now
            regret = (cands[1][0] if len(cands) > 1 else GAMMA) - cands[0][0]
            if regret > best_regret:
                best_regret, best_req, best_cand = regret, req, cands[0]
        
        if not best_req: break
        cost, v_type, v_idx, t_idx, pos, dep = best_cand
        vel, cap, d_lim = (conf["drone_vel"], conf["drone_cap"], conf["drone_lim"]*conf["drone_vel"]) if v_type == 'drone' else (conf["truck_vel"], conf["truck_cap"], float('inf'))
        v_trips = new_state.drone_trips[v_idx] if v_type == 'drone' else new_state.truck_trips[v_idx]
        
        if t_idx < 0:
            trip = Trip(v_type, v_idx, [best_req], dep)
            trip.eval(new_state.dist_matrix, vel, cap, d_lim, new_state.max_wait_time)
            v_trips.insert(pos, trip)
        else:
            trip = v_trips[t_idx]
            trip.stops.insert(pos, best_req)
            trip.depart_time = dep
            trip.eval(new_state.dist_matrix, vel, cap, d_lim, new_state.max_wait_time)
        new_state.unserved.remove(best_req)
    return new_state

def solve_alns(iterations=1000):
    """Main Optimization Cycle: Destroy -> Repair -> Accept/Reject."""
    alns = ALNS(np.random.RandomState(42))
    for op in [random_removal, shaw_removal, worst_removal, string_removal]: alns.add_destroy_operator(op)
    for op in [greedy_repair, regret_repair]: alns.add_repair_operator(op)
    
    # Warm-start with Greedy
    init = solve_greedy(VRPState(DATA, LW, ALPHA, GAMMA))
    
    # Configure Adaptive Weights and Simulated Annealing
    result = alns.iterate(init, RouletteWheel([50, 20, 5, 2], 0.8, 4, 1), 
                          SimulatedAnnealing(5000, 1, 0.9997), MaxIterations(iterations))
    return result.best_state

if __name__ == "__main__":
    print_solution(solve_alns(1000), name="ALNS_FINAL")
