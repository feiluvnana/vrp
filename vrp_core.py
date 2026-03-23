import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set

# =============================================================================
# VRP CORE — Data Structures for the Vehicle Routing Problem
# =============================================================================
# Problem: Given a fleet of trucks and drones operating from a single depot,
# serve a set of customer requests while minimizing total travel distance
# and the number of unserved (rejected) requests.
#
# Constraints (from PDF pages 1-2):
#   (a) Time windows:  vehicle must arrive before the customer's latest_end
#   (b) Waiting time:  W_i = q_i - p_i <= L_w  (freshness / perishability)
#   (c) Capacity:      total load per trip <= vehicle capacity (M_T or M_D)
#   (d) Drone radius:  total flight distance per drone trip <= L_D
#   (e) Horizon:       all trips must finish within [0, T]
#
# Request types (PDF page 1):
#   C1 = truck-only requests (can_drone = False)
#   C2 = flexible requests   (can_drone = True, servable by truck OR drone)
# =============================================================================


def load_data(path: str) -> dict:
    """Load problem instance from JSON. Returns default config if file missing."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "requests": [],
            "truck_vel": 15.0, "drone_vel": 30.0,
            "truck_cap": 400.0, "drone_cap": 2.27,
            "drone_lim": 700.0,
            "truck_num": 1, "drone_num": 1,
        }


DATA = load_data("data.json")

# --- Objective function weights (PDF page 3) ---
# Global Objective: min J = α · (Dist_T + Dist_D) + γ · Σ(1 - S_i)
#   α (ALPHA): Weight for total distance (usually 1.0)
#   γ (GAMMA): Global penalty per unserved request in the final evaluation.
#              This value MUST be significantly greater than the maximum possible
#              round-trip distance in your dataset. If max round-trip is 90,000m,
#              and GAMMA is 10,000, ALNS will intentionally drop requests to "save"
#              80,000 cost units. Good value for this dataset: 500,000.0.
ALPHA = 1.0
GAMMA = 500000.0  # Global penalty for ALNS and final objective evaluation.

# --- Insertion cost weights (PDF page 5) ---
# Incremental Cost: ΔJ = π₁ · Δd_truck + π₂ · Δd_drone
# These weights prioritize WHICH vehicle to use during greedy insertion.
#   π₁ (PI_1): Cost multiplier for adding 1 meter of TRUCK distance.
#   π₂ (PI_2): Cost multiplier for adding 1 meter of DRONE distance.
# Example: Setting PI_2 = 0.5 makes the solver explicitly prefer drone deliveries.
PI_1 = 1.0
PI_2 = 1.0

# --- Constraint parameter ---
# L_w: maximum waiting time from pickup to depot return (seconds)
LW = 3600.0


# =============================================================================
# REQUEST — A single customer delivery request
# =============================================================================
@dataclass(frozen=True)
class Request:
    """
    One customer order. Frozen (immutable) so it can be used in sets.
    Fields map directly to the PDF notation:
      r_i = request_time,  e_i = earliest_start,  l_i = latest_end,
      d_i = demand,        (x, y) = location
    """
    __slots__ = ["id", "x", "y", "demand", "can_drone",
                 "request_time", "earliest_start", "latest_end"]
    id: int
    x: float
    y: float
    demand: float
    can_drone: bool          # True = C2 (truck or drone), False = C1 (truck only)
    request_time: float      # r_i — when request appears in the system
    earliest_start: float    # e_i — earliest allowed pickup time
    latest_end: float        # l_i — latest allowed pickup time


def build_dist_matrix(requests: list) -> list:
    """
    Precompute Euclidean distances between all nodes.
    Index 0 = depot at origin (0,0). Indices 1..N = request locations.
    Avoids repeated sqrt calculations during search (O(N²) once vs O(N²) per eval).
    """
    nodes = [(0.0, 0.0)] + [(r.x, r.y) for r in requests]
    n = len(nodes)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((nodes[i][0] - nodes[j][0])**2 +
                          (nodes[i][1] - nodes[j][1])**2)
            mat[i][j] = mat[j][i] = d
    return mat


# =============================================================================
# TRIP — One depot-to-depot journey by a single vehicle
# =============================================================================
@dataclass
class Trip:
    """
    An atomic routing unit: one vehicle leaves the depot, visits a sequence
    of stops, and returns to the depot.  A vehicle can perform multiple
    consecutive Trips (e.g., truck_trips[k] = [Trip₁, Trip₂, ...]).

    The `eval()` method simulates the trip chronologically and checks all
    five constraint families (a)–(e) from the PDF.
    """
    vehicle_type: str        # "truck" or "drone"
    vehicle_index: int       # which vehicle (0, 1, 2, ...)
    stops: List[Request] = field(default_factory=list)
    depart_time: float = 0.0 # when the vehicle leaves the depot for this trip

    # --- Computed by eval() ---
    return_time: float = 0.0
    total_dist: float = 0.0
    total_load: float = 0.0
    service_times: Dict[int, float] = field(default_factory=dict)
    post_service_wait: Dict[int, float] = field(default_factory=dict)

    def eval(self, dist_matrix, velocity, capacity, dist_limit, max_wait) -> bool:
        """
        Simulate this trip from depot departure to depot return.
        Returns True if ALL constraints are satisfied, False otherwise.

        Constraint checks in order:
          0. Departure ≥ max request_time of loaded cargo (can't leave before cargo exists)
          1. Capacity:   cumulative load ≤ vehicle capacity     [constraint (c)]
          2. Distance:   cumulative dist ≤ dist_limit           [constraint (d), drones]
          3. Time window: arrival_time ≤ l_i                    [constraint (a)]
          4. Waiting:     q_depot - p_i ≤ L_w for each stop     [constraint (b)]
        """
        # 0. Can't depart before all loaded requests have appeared
        if self.stops and self.depart_time < max(r.request_time for r in self.stops):
            return False

        prev = 0  # start at depot (index 0 in dist_matrix)
        time, dist, load = self.depart_time, 0.0, 0.0
        svc = {}

        for req in self.stops:
            # (c) Capacity check
            load += req.demand
            if load > capacity:
                return False

            # Travel to next stop
            cur = req.id + 1  # +1 because index 0 = depot
            leg = dist_matrix[prev][cur]
            dist += leg

            # (d) Distance limit check (critical for drones with battery constraints)
            if dist > dist_limit:
                return False

            # (a) Time window check
            arrival = time + leg / velocity
            if arrival > req.latest_end:
                return False

            # Pickup time: p_i = max(arrival, e_i)  — wait if arrived early
            pickup = max(arrival, req.earliest_start)
            svc[req.id] = pickup

            prev = cur
            time = pickup

        # Return to depot
        leg_back = dist_matrix[prev][0]
        dist += leg_back
        if dist > dist_limit:
            return False

        q_depot = time + leg_back / velocity  # time of depot arrival

        # (b) Waiting time constraint: W_i = q_depot - p_i ≤ L_w
        waits = {}
        for rid, p_i in svc.items():
            w = q_depot - p_i
            if w > max_wait:
                return False
            waits[rid] = w

        # All constraints passed — commit results
        self.return_time = q_depot
        self.total_dist = dist
        self.total_load = load
        self.service_times = svc
        self.post_service_wait = waits
        return True

    def __repr__(self):
        ids = [r.id for r in self.stops]
        return (f"Trip({self.vehicle_type}#{self.vehicle_index}: {ids}, "
                f"dep={self.depart_time:.0f}, ret={self.return_time:.0f})")


# =============================================================================
# VRPSTATE — The complete solution representation
# =============================================================================
class VRPState:
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                  SOLUTION REPRESENTATION                         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  A VRP "solution" is a set of routes that assign requests to     ║
    ║  vehicles.  We represent it as a hierarchy of 4 levels:          ║
    ║                                                                  ║
    ║    VRPState                          ← the full solution         ║
    ║    ├── truck_trips[k] : List[Trip]   ← all trips for truck k     ║
    ║    ├── drone_trips[m] : List[Trip]   ← all trips for drone m     ║
    ║    └── unserved       : Set[Request] ← rejected requests         ║
    ║                                                                  ║
    ║  Each Trip contains:                                             ║
    ║    Trip                                                          ║
    ║    ├── stops: [req₁, req₂, ...]  ← visit order within this trip  ║
    ║    ├── depart_time               ← when vehicle leaves depot     ║
    ║    └── return_time               ← when vehicle returns to depot ║
    ║                                                                  ║
    ║  Example with 2 trucks and 1 drone:                              ║
    ║    truck_trips[0] = [Trip(stops=[r2,r5]), Trip(stops=[r8])]      ║
    ║    truck_trips[1] = [Trip(stops=[r1,r3,r7])]                     ║
    ║    drone_trips[0] = [Trip(stops=[r4,r6])]                        ║
    ║    unserved = {r9}   ← r9 couldn't be feasibly served            ║
    ║                                                                  ║
    ║  Why this design?                                                ║
    ║  1. Trip overlap detection is trivial:                           ║
    ║     just check trip[i].return_time ≤ trip[i+1].depart_time       ║
    ║  2. ALNS destroy/repair is easy:                                 ║
    ║     remove a Request from a Trip → add to unserved               ║
    ║     take from unserved → insert into a Trip                      ║
    ║  3. Objective calculation is straightforward:                    ║
    ║     sum all trip distances + penalty × |unserved|                ║
    ╚══════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, data, max_wait_time, dist_weight, penalty_weight):
        self.data = data
        self.max_wait_time = max_wait_time
        self.dist_weight = dist_weight       # α in the objective
        self.penalty_weight = penalty_weight # γ in the objective

        # Parse raw JSON arrays into Request objects
        self.requests = [
            Request(
                id=i, x=r[0], y=r[1], demand=r[2], can_drone=bool(r[3]),
                request_time=r[4], earliest_start=r[5], latest_end=r[6],
            )
            for i, r in enumerate(data["requests"])
        ]
        self.dist_matrix = build_dist_matrix(self.requests)

        # --- Route encoding: one list of Trips per vehicle ---
        self.truck_trips: List[List[Trip]] = [[] for _ in range(data["truck_num"])]
        self.drone_trips: List[List[Trip]] = [[] for _ in range(data["drone_num"])]

        # All requests start as unserved; the solver moves them into Trips
        self.unserved: Set[Request] = set(self.requests)

    def objective(self) -> float:
        """
        Objective function (PDF page 3):
          J = α · (Dist_T + Dist_D) + γ · |unserved|
        Lower is better. γ is set very high so the solver strongly prefers
        serving requests over rejecting them.
        """
        total_dist = sum(
            trip.total_dist
            for v_trips in self.truck_trips + self.drone_trips
            for trip in v_trips
        )
        return self.dist_weight * total_dist + self.penalty_weight * len(self.unserved)

    def validate_global(self) -> bool:
        """
        Check that no vehicle has overlapping trips in time.
        For each vehicle, trip[i].return_time must be ≤ trip[i+1].depart_time.
        """
        for v_trips in self.truck_trips + self.drone_trips:
            for i in range(1, len(v_trips)):
                if v_trips[i].depart_time < v_trips[i - 1].return_time:
                    return False
        return True

    def copy(self):
        """
        Deep copy for ALNS: each iteration destroys/repairs a copy,
        keeping the original solution intact for comparison.
        """
        import copy as _copy
        new = _copy.copy(self)

        def clone_trips(all_v_trips):
            return [
                [
                    Trip(t.vehicle_type, t.vehicle_index, list(t.stops), t.depart_time,
                         t.return_time, t.total_dist, t.total_load,
                         dict(t.service_times), dict(t.post_service_wait))
                    for t in trips
                ]
                for trips in all_v_trips
            ]

        new.truck_trips = clone_trips(self.truck_trips)
        new.drone_trips = clone_trips(self.drone_trips)
        new.unserved = set(self.unserved)
        return new


def print_solution(state: VRPState, name="SOLUTION"):
    """Print a human-readable summary of the solution."""
    print(f"\n=== {name} | Objective: {state.objective():.2f} ===")
    for i, trips in enumerate(state.truck_trips):
        for t in trips:
            print(f"  Truck {i}: dep={t.depart_time:.0f}  ret={t.return_time:.0f}  "
                  f"stops={[r.id for r in t.stops]}")
    for i, trips in enumerate(state.drone_trips):
        for t in trips:
            print(f"  Drone {i}: dep={t.depart_time:.0f}  ret={t.return_time:.0f}  "
                  f"stops={[r.id for r in t.stops]}")
    if state.unserved:
        print(f"  Unserved ({len(state.unserved)}): {sorted(r.id for r in state.unserved)}")
