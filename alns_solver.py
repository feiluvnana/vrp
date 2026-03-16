from vrp_core import Request, Trip, VRPState, DATA, LW, ALPHA, GAMMA, print_solution
from greedy_solver import solve_greedy, get_insertion_candidates
import numpy as np
import random
import math
from alns import ALNS
from alns.accept import HillClimbing, SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

def random_removal(state: VRPState, rnd_state):
    """Remove a random subset of requests."""
    new_state = state.copy()
    all_served = []
    
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            all_served.extend([(t, r) for r in t.stops])
    
    if not all_served:
        return new_state
        
    k = max(1, int(len(new_state.requests) * 0.3))
    num_to_remove = min(len(all_served), k)
    
    to_remove_indices = rnd_state.choice(len(all_served), num_to_remove, replace=False)
    
    for idx in to_remove_indices:
        trip, req = all_served[idx]
        if req in trip.stops:
            trip.stops.remove(req)
            new_state.unserved.add(req)
    
    new_state.truck_trips = [[t for t in v if t.stops] for v in new_state.truck_trips]
    new_state.drone_trips = [[t for t in v if t.stops] for v in new_state.drone_trips]
    return new_state

def shaw_removal(state: VRPState, rnd_state):
    """Remove similar (related) requests based on distance and time."""
    new_state = state.copy()
    all_served_reqs = []
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            all_served_reqs.extend([r for r in t.stops])
            
    if not all_served_reqs: return new_state
    
    pivot = rnd_state.choice(all_served_reqs)
    
    def relatedness(r1, r2):
        dist = math.sqrt((r1.x-r2.x)**2 + (r1.y-r2.y)**2)
        time_diff = abs(r1.r_i - r2.r_i)
        return dist + 0.1 * time_diff 
        
    all_served_reqs.sort(key=lambda r: relatedness(pivot, r))
    
    num_to_remove = max(1, int(len(new_state.requests) * 0.4))
    to_remove = all_served_reqs[:num_to_remove]
    
    for req in to_remove:
        for v_trips in new_state.truck_trips + new_state.drone_trips:
            for t in v_trips:
                if req in t.stops:
                    t.stops.remove(req)
                    new_state.unserved.add(req)
                    
    new_state.truck_trips = [[t for t in v if t.stops] for v in new_state.truck_trips]
    new_state.drone_trips = [[t for t in v if t.stops] for v in new_state.drone_trips]
    return new_state

def worst_removal(state: VRPState, rnd_state):
    """Remove requests with the highest cost contribution."""
    new_state = state.copy()
    all_served = [] 
    
    for v_trips in new_state.truck_trips + new_state.drone_trips:
        for t in v_trips:
            base_dist = t.total_dist
            v_vel = state.data["drone_vel"] if t.vtype == 'drone' else state.data["truck_vel"]
            v_cap = state.data["drone_cap"] if t.vtype == 'drone' else state.data["truck_cap"]
            v_lim = (state.data["drone_lim"] * v_vel) if t.vtype == 'drone' else float('inf')
            
            for i, req in enumerate(t.stops):
                temp_stops = t.stops[:i] + t.stops[i+1:]
                if not temp_stops:
                    contribution = base_dist
                else:
                    temp_trip = Trip(vtype=t.vtype, vidx=t.vidx, stops=temp_stops, depart_time=t.depart_time)
                    if temp_trip.eval([0,0], v_vel, v_cap, v_lim, state.lw):
                        contribution = base_dist - temp_trip.total_dist
                    else:
                        contribution = 0
                all_served.append({'trip': t, 'req': req, 'cost': contribution})
    
    if not all_served: return new_state
    
    all_served.sort(key=lambda x: x['cost'], reverse=True)
    k = max(1, int(len(new_state.requests) * 0.2)) 
    num_to_remove = min(len(all_served), k)
    
    for _ in range(num_to_remove):
        idx = int(rnd_state.random()**3 * len(all_served))
        item = all_served.pop(idx)
        if item['req'] in item['trip'].stops:
            item['trip'].stops.remove(item['req'])
            new_state.unserved.add(item['req'])
            
    new_state.truck_trips = [[t for t in v if t.stops] for v in new_state.truck_trips]
    new_state.drone_trips = [[t for t in v if t.stops] for v in new_state.drone_trips]
    return new_state

def string_removal(state: VRPState, rnd_state):
    """Remove a sequence of consecutive requests from a random trip."""
    new_state = state.copy()
    all_trips = [t for v_trips in new_state.truck_trips + new_state.drone_trips for t in v_trips if t.stops]
    if not all_trips: return new_state
    
    trip = rnd_state.choice(all_trips)
    max_string = min(len(trip.stops), 4)
    size = rnd_state.randint(1, max_string + 1)
    start = rnd_state.randint(0, len(trip.stops) - size + 1)
    
    to_remove = trip.stops[start:start+size]
    for req in list(to_remove):
        trip.stops.remove(req)
        new_state.unserved.add(req)

    new_state.truck_trips = [[t for t in v if t.stops] for v in new_state.truck_trips]
    new_state.drone_trips = [[t for t in v if t.stops] for v in new_state.drone_trips]
    return new_state

def greedy_repair(state: VRPState, rnd_state):
    """Repair method using greedy insertion."""
    return solve_greedy(state)

def regret_repair(state: VRPState, rnd_state):
    """Repair method prioritizing requests with high regret cost."""
    new_state = state.copy()
    
    while new_state.unserved:
        best_regret = -1.0
        best_req = None
        best_cand = None
        
        for req in list(new_state.unserved):
            candidates = get_insertion_candidates(new_state, req)
            if not candidates: continue
            
            c1 = candidates[0][0]
            c2 = candidates[1][0] if len(candidates) > 1 else GAMMA 
            regret = c2 - c1
            
            if regret > best_regret:
                best_regret = regret
                best_req = req
                best_cand = candidates[0]
        
        if not best_req: break
        
        cost, vtype, vidx, trip_idx, pos, depart = best_cand
        d_vel, t_vel = new_state.data["drone_vel"], new_state.data["truck_vel"]
        d_cap, t_cap = new_state.data["drone_cap"], new_state.data["truck_cap"]
        d_lim = new_state.data["drone_lim"] * d_vel
        
        if trip_idx < 0:
            new_trip = Trip(vtype=vtype, vidx=vidx, stops=[best_req], depart_time=depart)
            vel = d_vel if vtype == 'drone' else t_vel
            cap = d_cap if vtype == 'drone' else t_cap
            dist_lim = d_lim if vtype == 'drone' else float('inf')
            new_trip.eval([0,0], vel, cap, dist_lim, new_state.lw)
            if vtype == 'drone': new_state.drone_trips[vidx].insert(pos, new_trip)
            else: new_state.truck_trips[vidx].insert(pos, new_trip)
        else:
            trip = (new_state.drone_trips[vidx][trip_idx] if vtype == 'drone' else new_state.truck_trips[vidx][trip_idx])
            trip.stops.insert(pos, best_req)
            trip.depart_time = depart
            vel = d_vel if vtype == 'drone' else t_vel
            cap = d_cap if vtype == 'drone' else t_cap
            dist_lim = d_lim if vtype == 'drone' else float('inf')
            trip.eval([0,0], vel, cap, dist_lim, new_state.lw)
            
        new_state.unserved.remove(best_req)
        
    return new_state

def solve_alns(iterations=10000):
    """Main execution of the ALNS algorithm."""
    initial_state = VRPState(DATA, LW, ALPHA, GAMMA)
    initial_sol = solve_greedy(initial_state)
    
    print("--- INITIAL GREEDY RESULT ---")
    print_solution(initial_sol, name="INITIAL_GREEDY")
    
    alns = ALNS(np.random.RandomState(42))
    
    alns.add_destroy_operator(random_removal)
    alns.add_destroy_operator(shaw_removal)
    alns.add_destroy_operator(worst_removal)
    alns.add_destroy_operator(string_removal)
    
    alns.add_repair_operator(greedy_repair)
    alns.add_repair_operator(regret_repair)
    
    select = RouletteWheel([50, 20, 5, 2], 0.8, 2, 1) 
    accept = SimulatedAnnealing(5000, 1, 0.9997)
    stop = MaxIterations(iterations)
    
    print(f"\nRunning ALNS for {iterations} iterations. Please wait...")
    result = alns.iterate(initial_sol, select, accept, stop)
    
    return result.best_state

if __name__ == "__main__":
    best_state = solve_alns(iterations=1000)
    print("\n" + "="*50)
    print("      ALNS OPTIMIZATION COMPLETE!")
    print("="*50)
    print_solution(best_state, name="ALNS_FINAL")
