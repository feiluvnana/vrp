import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple

def load_data(file_path):
    try:
        with open(file_path, "r") as f: return json.load(f)
    except FileNotFoundError:
        return {"requests": [], "truck_vel": 15.0, "drone_vel": 30.0, "truck_cap": 400.0, "drone_cap": 2.27, "drone_lim": 700.0, "truck_num": 1, "drone_num": 1}

DATA = load_data("data.json")
ALPHA, GAMMA, LW = 1.0, 10000.0, 3600.0

@dataclass(frozen=True)
class Request:
    # Use __slots__ to reduce memory footprint for thousands of instances
    __slots__ = ['id', 'x', 'y', 'demand', 'can_drone', 'request_time', 'earliest_start', 'latest_end']
    id: int
    x: float
    y: float
    demand: float
    can_drone: bool
    request_time: float      # Time when the request becomes available
    earliest_start: float    # Delivery time window start
    latest_end: float        # Delivery time window end

def build_dist_matrix(requests):
    """Precompute distance matrix to avoid repeated sqrt calculations (O(N^2))."""
    # Index 0 is the Depot, indices 1 to N are requests
    nodes = [(0.0, 0.0)] + [(r.x, r.y) for r in requests]
    n = len(nodes)
    matrix = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d = math.sqrt((nodes[i][0]-nodes[j][0])**2 + (nodes[i][1]-nodes[j][1])**2)
            matrix[i][j] = matrix[j][i] = d
    return matrix

@dataclass
class Trip:
    vehicle_type: str       # 'truck' or 'drone'
    vehicle_index: int      
    stops: List[Request] = field(default_factory=list)
    depart_time: float = 0.0
    return_time: float = 0.0
    total_dist: float = 0.0
    total_load: float = 0.0
    service_times: Dict[int, float] = field(default_factory=dict)
    post_service_wait: Dict[int, float] = field(default_factory=dict)

    def eval(self, dist_matrix, velocity, capacity, distance_limit, max_wait_limit) -> bool:
        """Validate all constraints (capacity, battery, time windows, waiting)."""
        # Vehicle cannot depart warehouse before its requests exist in the system
        if self.stops and self.depart_time < max(req.request_time for req in self.stops):
            return False

        prev_node_idx = 0 # Start at Depot (index 0)
        current_time, dist_acc, load_acc = self.depart_time, 0.0, 0.0
        service_times = {}

        for req in self.stops:
            load_acc += req.demand
            if load_acc > capacity: return False
            
            # Lookup precomputed distance: req.id + 1 maps Request ID to matrix index
            curr_node_idx = req.id + 1
            leg = dist_matrix[prev_node_idx][curr_node_idx]
            dist_acc += leg
            if dist_acc > distance_limit: return False
            
            arrival_time = current_time + leg / velocity
            if arrival_time > req.latest_end: return False
            
            # Actual delivery time is arrival time buffered by earliest window
            pickup_time = max(arrival_time, req.earliest_start)
            service_times[req.id] = pickup_time
            
            prev_node_idx = curr_node_idx
            current_time = pickup_time

        # Calculate leg back to depot (index 0)
        leg_back = dist_matrix[prev_node_idx][0]
        dist_acc += leg_back
        if dist_acc > distance_limit: return False
        
        q_depot = current_time + leg_back / velocity
        
        post_service_wait = {}
        for rid, p_i in service_times.items():
            # FRESHNESS/WAIT CONSTRAINT: Time from delivery until vehicle returns to warehouse
            wait_time = q_depot - p_i
            if wait_time > max_wait_limit: return False
            post_service_wait[rid] = wait_time
        
        self.return_time, self.total_dist = q_depot, dist_acc
        self.total_load, self.service_times = load_acc, service_times
        self.post_service_wait = post_service_wait
        return True

class VRPState:
    """Represents a complete VRP solution state."""
    def __init__(self, data, max_wait_time, dist_weight, penalty_weight):
        self.data = data
        self.max_wait_time, self.dist_weight, self.penalty_weight = max_wait_time, dist_weight, penalty_weight
        self.requests = [Request(i, r[0], r[1], r[2], bool(r[3]), r[4], r[5], r[6]) for i, r in enumerate(data["requests"])]
        self.dist_matrix = build_dist_matrix(self.requests)
        self.truck_trips = [[] for _ in range(data["truck_num"])]
        self.drone_trips = [[] for _ in range(data["drone_num"])]
        self.unserved = set(self.requests)

    def objective(self) -> float:
        """Minimize weighted distance + penalty for dropped orders."""
        dists = sum(t.total_dist for v in self.truck_trips + self.drone_trips for t in v)
        return self.dist_weight * dists + self.penalty_weight * len(self.unserved)

    def copy(self):
        """Create a deep copy for ALNS neighborhood search."""
        import copy
        new_state = copy.copy(self)
        new_state.truck_trips = [copy.deepcopy(v) for v in self.truck_trips]
        new_state.drone_trips = [copy.deepcopy(v) for v in self.drone_trips]
        new_state.unserved = self.unserved.copy()
        return new_state

def print_solution(state: VRPState, name="SOL"):
    print(f"\n=== {name} | Objective: {state.objective():.2f} ===")
    for i, v in enumerate(state.truck_trips):
        for trip in v: print(f"Truck {i}: Dep {trip.depart_time:.1f} | Ret {trip.return_time:.1f} | Stops {[r.id for r in trip.stops]}")
    for i, v in enumerate(state.drone_trips):
        for trip in v: print(f"Drone {i}: Dep {trip.depart_time:.1f} | Ret {trip.return_time:.1f} | Stops {[r.id for r in trip.stops]}")
    if state.unserved: print(f"Unserved: {[r.id for r in state.unserved]}")
