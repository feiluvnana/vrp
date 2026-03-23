from vrp_core import (Request, Trip, VRPState, DATA, LW, ALPHA, GAMMA,
                       print_solution)
from greedy_solver import solve_greedy, get_insertion_candidates, DELTA_MIN
import numpy as np
import random
import math
from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

# =============================================================================
# ALNS — Adaptive Large Neighborhood Search
# =============================================================================
# ALNS improves an initial solution (from Greedy) by repeatedly:
#   1. DESTROY: remove some requests from the current solution → unserved pool
#   2. REPAIR:  re-insert unserved requests into (hopefully better) positions
#   3. ACCEPT:  decide whether to keep the new solution (via Simulated Annealing)
#
# "Adaptive" means the algorithm tracks which destroy/repair operators
# perform well and increases their selection probability over time
# (via Roulette Wheel selection with score-based weight updates).
# =============================================================================


def _clean(state: VRPState) -> VRPState:
    """Remove empty trips (no stops) to keep the solution clean."""
    state.truck_trips = [[t for t in v if t.stops] for v in state.truck_trips]
    state.drone_trips = [[t for t in v if t.stops] for v in state.drone_trips]
    return state


def _get_served(state: VRPState):
    """Collect all (trip, request) pairs currently in the solution."""
    return [
        (trip, req)
        for v_trips in state.truck_trips + state.drone_trips
        for trip in v_trips
        for req in trip.stops
    ]


def _remove_request(state: VRPState, trip: Trip, req: Request):
    """Remove a request from its trip and add it back to the unserved pool."""
    if req in trip.stops:
        trip.stops.remove(req)
        state.unserved.add(req)


# =============================================================================
# DESTROY OPERATORS
# =============================================================================
# Each operator removes a subset of requests, creating "room" for the repair
# operator to find better placements.  Different strategies create different
# kinds of disruption, which is key to escaping local optima.
# =============================================================================

def random_removal(state: VRPState, rng) -> VRPState:
    """
    Remove ~30% of served requests, chosen uniformly at random.
    This is the simplest destroy operator — it creates broad, unpatterned gaps
    that force the repair operator to rebuild from scratch in those areas.
    """
    s = state.copy()
    served = _get_served(s)
    if not served:
        return s

    # Remove up to 30% of all requests
    k = min(len(served), max(1, int(len(s.requests) * 0.3)))
    for idx in rng.choice(len(served), k, replace=False):
        _remove_request(s, *served[idx])

    return _clean(s)


def shaw_removal(state: VRPState, rng) -> VRPState:
    """
    Shaw (Relatedness) Removal: remove requests that are "similar" to a random pivot.

    Similarity is measured by:
      - Spatial distance  (close customers are related)
      - Temporal distance (similar request times are related)
      - Demand similarity (similar cargo sizes are related)

    Why? Removing a cluster of related requests opens up a coherent region
    for the repair operator to re-optimize, rather than scattered gaps.

    Uses randomized selection: instead of taking the top-k most similar,
    we use index = floor(random^φ * n) which biases toward similar requests
    but adds randomization (φ > 1 concentrates more toward the most similar).
    """
    s = state.copy()
    served = _get_served(s)
    if not served:
        return s

    # Pick a random pivot request
    pivot = served[rng.randint(len(served))][1]

    # Score all served requests by relatedness to pivot
    # Lower score = more similar = more likely to be removed
    max_dist = max(math.sqrt(r.x**2 + r.y**2) for _, r in served) or 1.0
    max_demand = max(r.demand for _, r in served) or 1.0

    scored = []
    for trip, req in served:
        spatial = math.sqrt((pivot.x - req.x)**2 + (pivot.y - req.y)**2) / max_dist
        temporal = abs(pivot.request_time - req.request_time) / 3600.0
        demand = abs(pivot.demand - req.demand) / max_demand
        # Weighted combination (spatial dominates, temporal and demand secondary)
        score = 0.6 * spatial + 0.2 * temporal + 0.2 * demand
        scored.append((score, trip, req))

    scored.sort(key=lambda x: x[0])

    # Remove up to 40% using biased random selection (φ=2 exponent)
    k = max(1, int(len(s.requests) * 0.4))
    for _ in range(min(k, len(scored))):
        idx = int(rng.random()**2 * len(scored))
        _, trip, req = scored.pop(idx)
        _remove_request(s, trip, req)

    return _clean(s)


def worst_removal(state: VRPState, rng) -> VRPState:
    """
    Remove requests that contribute the most to the total distance.

    For each request, we compute "how much distance would be saved if we
    removed it from its trip."  Requests with high marginal cost are
    candidates for removal — they're likely poorly placed.

    Uses biased random selection (exponent φ=3): strongly favors removing
    the worst-placed requests, but occasionally picks less-terrible ones
    to maintain diversity.
    """
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
                # Simulate trip without this request
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

    # Sort descending by savings (worst-placed = highest savings if removed)
    costs.sort(key=lambda x: x[0], reverse=True)

    k = min(len(costs), max(1, int(len(s.requests) * 0.2)))
    for _ in range(k):
        # φ=3 exponent: strongly biases toward the worst items
        idx = int(rng.random()**3 * len(costs))
        _, trip, req = costs.pop(idx)
        _remove_request(s, trip, req)

    return _clean(s)


def string_removal(state: VRPState, rng) -> VRPState:
    """
    Remove a consecutive substring of stops from a single trip.

    This preserves the "before" and "after" structure of the trip while
    opening a gap in the middle.  Useful for rethinking a portion of a route
    without disrupting the entire plan.
    """
    s = state.copy()
    trips = [t for v in s.truck_trips + s.drone_trips for t in v if t.stops]
    if not trips:
        return s

    trip = trips[rng.randint(len(trips))]
    n = len(trip.stops)
    size = rng.randint(1, min(n, 4) + 1)         # remove 1–4 consecutive stops
    start = rng.randint(0, n - size + 1)

    for req in list(trip.stops[start:start + size]):
        _remove_request(s, trip, req)

    return _clean(s)


# =============================================================================
# REPAIR OPERATORS
# =============================================================================
# After destroying part of the solution, repair operators re-insert unserved
# requests.  Different strategies produce different solution structures.
# =============================================================================

def greedy_repair(state: VRPState, rng) -> VRPState:
    """
    Repair by running the standard greedy insertion heuristic.
    Each unserved request is inserted at its cheapest feasible position,
    one at a time, in urgency order.
    """
    return solve_greedy(state)


def regret_repair(state: VRPState, rng) -> VRPState:
    """
    Regret-3 Repair: a more look-ahead repair strategy.

    Instead of greedily picking the cheapest insertion globally, we ask:
    "Which unserved request would we regret the most if we DON'T insert
     it right now?"

    Regret is measured as:
      regret = cost_of_3rd_best_insertion - cost_of_best_insertion

    A request with high regret has one clearly superior insertion position
    that might disappear if we wait — so we should insert it NOW.

    This is less myopic than pure greedy: it prioritizes "fragile" requests
    (those with few good options) over "flexible" ones (many good options).
    """
    s = state.copy()
    cfg = s.data

    while s.unserved:
        best_regret = -1.0
        best_req = None
        best_cand = None

        for req in list(s.unserved):
            cands = get_insertion_candidates(s, req, delta=DELTA_MIN)
            if not cands:
                continue

            # Regret-3: difference between 3rd-best and best insertion cost
            # If fewer than 3 candidates, use a large penalty as the missing cost
            # (this means requests with very few options get high priority)
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
            break  # no more feasible insertions

        # Execute the best insertion (same logic as greedy solver)
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
    """
    Main ALNS loop.

    1. Build initial solution with greedy heuristic
    2. Register destroy/repair operators
    3. Run ALNS with Simulated Annealing acceptance and Roulette Wheel selection
    4. Return the best solution found

    SA temperature is calibrated from the initial objective so it scales
    properly with different instance sizes (not a hardcoded constant).
    """
    rng = np.random.RandomState(42)
    alns = ALNS(rng)

    # Register operators
    for op in [random_removal, shaw_removal, worst_removal, string_removal]:
        alns.add_destroy_operator(op)
    for op in [greedy_repair, regret_repair]:
        alns.add_repair_operator(op)

    # Build initial greedy solution
    init = solve_greedy(VRPState(DATA, LW, ALPHA, GAMMA))
    init_obj = init.objective()

    # --- Roulette Wheel selection ---
    # Scores: [global_best, local_best, accepted, rejected]
    # The algorithm increases the weight of operators that produce these outcomes.
    # decay=0.8 means weights slowly converge; higher = more memory of past performance
    select = RouletteWheel([50, 20, 5, 2], 0.8, 4, 1)

    # --- Simulated Annealing acceptance ---
    # Start temperature: ~30% of initial objective (accepts 30% worse solutions initially)
    # End temperature: 1 (essentially greedy near the end)
    # Step: 0.9997 cooling rate → slow cooling preserves exploration
    start_temp = max(1.0, 0.3 * init_obj)
    accept = SimulatedAnnealing(start_temp, 1, 0.9997)

    stop = MaxIterations(iterations)

    result = alns.iterate(init, select, accept, stop)
    return result.best_state


if __name__ == "__main__":
    best = solve_alns(1000)
    print_solution(best, "ALNS OPTIMIZED SOLUTION")
