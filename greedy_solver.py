from vrp_core import (Request, Trip, VRPState, DATA, LW, ALPHA, GAMMA,
                       PI_1, PI_2, print_solution)
import math

# =============================================================================
# GREEDY SOLVER — Slack-Based Insertion Heuristic (from PDF pages 3–8)
# =============================================================================
#
# High-level idea:
#   Process requests one by one (sorted by urgency).
#   For each request, try inserting it at every feasible position across
#   all vehicles/trips.  Score each insertion using an "improved cost"
#   that balances distance increase vs. schedule flexibility (slack).
#   Pick the cheapest insertion, or reject the request if too expensive.
#
# Key formula (PDF page 5):
#   ΔJ'(c) = ΔJ(c) + β · max{0, δ(t) - SlackMin(τ')}
#   where:
#     ΔJ(c)       = π₁·Δd_truck  or  π₂·Δd_drone   (raw distance increase)
#     SlackMin(τ') = min over all stops of (l_i - S_i)  (tightest remaining buffer)
#     δ(t)         = dynamic safety threshold via EWMA   (what slack we'd like to keep)
#     β            = penalty weight for tight schedules
#
# Parameters tuning guidance (PDF pages 7–8):
#   β  ≈ cost of "1 km" — so slack shortage is penalized in the same units as distance
#   κ  ∈ [0.05, 0.15] — 10% of avg time-window width becomes the safety margin
#   λ  ∈ (0, 1] — EWMA smoothing (small = stable, large = reactive)
#   ρ  = rejection threshold; if best insertion cost > ρ, reject the request
#   δ_min, δ_max = floor/ceiling for the dynamic safety threshold
# =============================================================================

# --- Tunable parameters ---
# BETA (β): Converts "slack units" (seconds) into "cost units" (meters).
#           If β = 15.0, a 1-second violation of the safety threshold 'costs' the
#           same as driving 15 extra meters. (15m/s = truck speed, so 1 sec = 15m)
BETA = 15.0

# KAPPA (κ): Determines the target safety margin. If κ = 0.1, the algorithm tries
#            to maintain 10% of the customer's average time window as idle slack time.
KAPPA = 0.1

# DELTA Bounds (δ_min, δ_max): Floor and ceiling for the safety threshold (in seconds).
#                              Prevents the moving average from becoming too wild.
DELTA_MIN = 180.0  # At least 3 minutes of slack always required.
DELTA_MAX = 720.0  # Never demand more than 12 minutes of slack.

# LAMBDA (λ): EWMA smoothing factor. How fast does the algorithm adapt to changing
#             time windows? λ = 0.05 means old requests are weighted 95% and the
#             newest request only affects 5% of the average.
LAMBDA = 0.05

# RHO (ρ): Local GREEDY Optional Rejection Threshold.
#          Unlike GAMMA (which punishes the FINAL objective score), RHO is used
#          during greedy insertion. If the cheapest insertion for a single order
#          costs MORE than RHO, the greedy algorithm will skip it immediately.
#          Best Value: Should be > max possible round-trip distance (e.g. 100,000+),
#          or `float("inf")` to let greedy fit everyone it can and let ALNS optimize it.
RHO = float("inf")


def get_insertion_candidates(state: VRPState, req: Request, delta: float):
    """
    Generate all feasible insertion candidates for `req` into the current solution.

    For every vehicle and every position within every trip (plus creating new trips),
    we check feasibility and compute the improved cost ΔJ'(c).

    Returns a sorted list of candidates (cheapest first).
    Each candidate is a tuple:
      (cost, vehicle_type, vehicle_idx, trip_idx, insert_pos, depart_time)

    Convention: trip_idx < 0 means "create a new trip".
      Specifically, trip_idx = -(1 + position) where position is where
      the new trip should be inserted into the vehicle's trip list.
    """
    # Vehicle configurations: (velocity, capacity, max_distance)
    # Trucks have no distance limit; drones are limited by battery range
    truck_cfg = (state.data["truck_vel"], state.data["truck_cap"], float("inf"))
    drone_cfg = (state.data["drone_vel"], state.data["drone_cap"],
                 state.data["drone_lim"] * state.data["drone_vel"])

    candidates = []

    # Try drone first (if eligible), then truck.  This ordering doesn't
    # affect correctness since we pick the best candidate at the end.
    v_types = ["drone", "truck"] if req.can_drone else ["truck"]

    for v_type in v_types:
        vel, cap, d_lim = drone_cfg if v_type == "drone" else truck_cfg

        # Skip immediately if request demand exceeds vehicle capacity
        if req.demand > cap:
            continue

        # Choose the cost weight based on vehicle type (PDF: π₁ for truck, π₂ for drone)
        pi = PI_2 if v_type == "drone" else PI_1

        v_list = state.drone_trips if v_type == "drone" else state.truck_trips

        for v_idx, trips in enumerate(v_list):

            # === Option A: Insert into an EXISTING trip ===
            for t_idx, trip in enumerate(trips):
                # Time boundaries to prevent trip overlaps on the same vehicle
                prev_ret = trips[t_idx - 1].return_time if t_idx > 0 else 0.0
                next_dep = trips[t_idx + 1].depart_time if t_idx + 1 < len(trips) else float("inf")

                # Try every insertion position within this trip
                for pos in range(len(trip.stops) + 1):
                    new_stops = trip.stops[:pos] + [req] + trip.stops[pos:]

                    # Departure = max(previous trip return, latest request_time in cargo)
                    dep = max(prev_ret, max(r.request_time for r in new_stops))
                    trial = Trip(v_type, v_idx, new_stops, dep)

                    # Step 2.3 — Feasibility check (constraints a–e)
                    if not trial.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time):
                        continue
                    if trial.return_time > next_dep:
                        continue  # would overlap with the next trip

                    # Step 2.4 — Compute improved cost ΔJ'(c)
                    # ΔJ = π · (new_dist - old_dist) — raw distance increase
                    delta_dist = pi * (trial.total_dist - trip.total_dist)

                    # SlackMin = min slack across all stops after insertion
                    # slack_i = l_i - S_i  (how much time buffer remains)
                    min_slack = min(
                        r.latest_end - trial.service_times[r.id]
                        for r in trial.stops
                    )

                    # Slack penalty: penalize if min slack drops below safety threshold δ
                    slack_pen = BETA * max(0.0, delta - min_slack)

                    cost = delta_dist + slack_pen
                    candidates.append((cost, v_type, v_idx, t_idx, pos, dep))

            # === Option B: Create a BRAND NEW trip ===
            for t_pos in range(len(trips) + 1):
                prev_ret = trips[t_pos - 1].return_time if t_pos > 0 else 0.0
                next_dep = trips[t_pos].depart_time if t_pos < len(trips) else float("inf")

                new_trip = Trip(v_type, v_idx, [req], max(prev_ret, req.request_time))

                if not new_trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time):
                    continue
                if new_trip.return_time > next_dep:
                    continue

                # For a single-stop trip: slack = l_i - S_i
                slack_r = req.latest_end - new_trip.service_times[req.id]
                slack_pen = BETA * max(0.0, delta - slack_r)
                cost = pi * new_trip.total_dist + slack_pen

                # Negative trip_idx signals "new trip"; t_pos = where to insert it
                candidates.append((cost, v_type, v_idx, -(1 + t_pos), t_pos, new_trip.depart_time))

    # Sort by cost ascending — the solver picks candidates[0]
    candidates.sort(key=lambda c: c[0])
    return candidates


def solve_greedy(state: VRPState) -> VRPState:
    """
    Slack-Based Greedy Insertion Heuristic (PDF Algorithm: GreedyAssign).

    Steps (matching PDF page 4):
      2.1  Sort requests by urgency (earliest deadline first, with distance tie-break)
      2.2  For each request, generate all feasible insertion candidates
      2.3  Check feasibility of each candidate
      2.4  Score each candidate using ΔJ' = ΔJ + β·max{0, δ - SlackMin}
      2.5  Pick the cheapest; reject if cost > ρ (optional rejection)

    The EWMA (Exponentially Weighted Moving Average) from PDF pages 7-8
    dynamically adapts the safety threshold δ(t) based on recent requests'
    time-window widths. This helps because:
      - If recent requests have wide windows → δ grows → we demand more slack
      - If recent requests have tight windows → δ shrinks → we accept tighter fits
    """
    # Step 2.1 — Sort by urgency: earliest deadline first, then closer to depot
    # This ensures tight-deadline requests get served first (common OR heuristic)
    sorted_reqs = sorted(
        state.unserved,
        key=lambda r: r.latest_end - math.sqrt(r.x**2 + r.y**2) / 15.0,
    )

    # Initialize EWMA state for the dynamic safety threshold δ(t)
    # m tracks the "typical" time-window width across recent requests
    m = 0.0  # will be initialized to first request's window width

    for req in sorted_reqs:
        # --- EWMA update (PDF page 7) ---
        # w_r = l_r - e_r = time-window width of this request
        w_r = req.latest_end - req.earliest_start

        if m == 0.0:
            m = w_r  # initialize with first request
        else:
            # m ← (1-λ)·m + λ·w_r  — blend old average with new observation
            # Small λ=0.05 means only 5% weight on new data → smooth, stable
            m = (1 - LAMBDA) * m + LAMBDA * w_r

        # --- Dynamic safety threshold (PDF page 8) ---
        # δ(t) = clamp(κ·m, δ_min, δ_max)
        # κ=0.1 means we want slack ≥ 10% of the typical time-window width
        delta_t = min(DELTA_MAX, max(DELTA_MIN, KAPPA * m))

        # Step 2.2 + 2.3 + 2.4 — Generate & score candidates
        cands = get_insertion_candidates(state, req, delta_t)

        # Step 2.5 — Selection with optional rejection
        if not cands:
            # (a) Mandatory rejection: no feasible insertion exists at all
            continue

        best_cost, v_type, v_idx, t_idx, pos, dep = cands[0]

        if best_cost > RHO:
            # (b) Optional rejection: feasible but too expensive
            # Serving this request would cost more than the penalty for leaving it unserved
            continue

        # --- Execute the best insertion ---
        v_trips = state.drone_trips[v_idx] if v_type == "drone" else state.truck_trips[v_idx]
        cfg = state.data
        vel, cap, d_lim = (
            (cfg["drone_vel"], cfg["drone_cap"], cfg["drone_lim"] * cfg["drone_vel"])
            if v_type == "drone"
            else (cfg["truck_vel"], cfg["truck_cap"], float("inf"))
        )

        if t_idx < 0:
            # Negative index → create a new trip at position `pos`
            new_trip = Trip(v_type, v_idx, [req], dep)
            new_trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time)
            v_trips.insert(pos, new_trip)
        else:
            # Non-negative → insert into existing trip at position `pos`
            trip = v_trips[t_idx]
            trip.stops.insert(pos, req)
            trip.depart_time = dep
            trip.eval(state.dist_matrix, vel, cap, d_lim, state.max_wait_time)

        # Move request from unserved to served
        state.unserved.remove(req)

    return state


if __name__ == "__main__":
    result = solve_greedy(VRPState(DATA, LW, ALPHA, GAMMA))
    print_solution(result, "GREEDY SOLUTION")
