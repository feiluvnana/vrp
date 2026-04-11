import math
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set

def load_data(path: str) -> dict:
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
MAX_WAITING_TIME = 3600.0

@dataclass(frozen=True)
class Request:
    __slots__ = ["id", "x", "y", "demand", "can_drone",
                 "request_time", "time_window_start", "time_window_end"]
    id: int
    x: float
    y: float
    demand: float
    can_drone: bool
    request_time: float
    time_window_start: float
    time_window_end: float

def build_dist_matrix(requests: list) -> list:
    nodes = [(0.0, 0.0)] + [(r.x, r.y) for r in requests]
    n = len(nodes)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((nodes[i][0] - nodes[j][0])**2 +
                          (nodes[i][1] - nodes[j][1])**2)
            mat[i][j] = mat[j][i] = d
    return mat

@dataclass
class Trip:
    vehicle_type: str
    vehicle_index: int
    stops: List[Request] = field(default_factory=list)
    depart_time: float = 0.0
    return_time: float = 0.0
    total_dist: float = 0.0
    total_load: float = 0.0
    service_times: Dict[int, float] = field(default_factory=dict)
    post_service_wait: Dict[int, float] = field(default_factory=dict)

    def eval(self, dist_matrix, velocity, capacity, dist_limit, max_wait) -> bool:
        if self.stops and self.depart_time < max(r.request_time for r in self.stops):
            return False

        prev = 0
        time, dist, load = self.depart_time, 0.0, 0.0
        svc = {}

        for req in self.stops:
            load += req.demand
            if load > capacity:
                return False

            cur = req.id + 1
            leg = dist_matrix[prev][cur]
            dist += leg

            if dist > dist_limit:
                return False

            arrival = time + leg / velocity
            if arrival > req.time_window_end:
                return False

            pickup = max(arrival, req.time_window_start)
            svc[req.id] = pickup

            prev = cur
            time = pickup

        leg_back = dist_matrix[prev][0]
        dist += leg_back
        if dist > dist_limit:
            return False

        q_depot = time + leg_back / velocity

        waits = {}
        for rid, p_i in svc.items():
            w = q_depot - p_i
            if w > max_wait:
                return False
            waits[rid] = w

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

class VRPState:
    def __init__(self, data, max_wait_time, dist_weight, penalty_weight):
        self.data = data
        self.max_wait_time = max_wait_time
        self.dist_weight = dist_weight
        self.penalty_weight = penalty_weight

        self.requests = [
            Request(
                id=i, x=r[0], y=r[1], demand=r[2], can_drone=bool(r[3]),
                request_time=r[4], time_window_start=r[5], time_window_end=r[6],
            )
            for i, r in enumerate(data["requests"])
        ]
        self.dist_matrix = build_dist_matrix(self.requests)

        self.truck_trips: List[List[Trip]] = [[] for _ in range(data["truck_num"])]
        self.drone_trips: List[List[Trip]] = [[] for _ in range(data["drone_num"])]
        self.unserved: Set[Request] = set(self.requests)

    def objective(self) -> float:
        total_dist = sum(
            trip.total_dist
            for v_trips in self.truck_trips + self.drone_trips
            for trip in v_trips
        )
        return self.dist_weight * total_dist + self.penalty_weight * len(self.unserved)

    def validate_global(self) -> bool:
        for v_trips in self.truck_trips + self.drone_trips:
            for i in range(1, len(v_trips)):
                if v_trips[i].depart_time < v_trips[i - 1].return_time:
                    return False
        return True

    def copy(self):
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
