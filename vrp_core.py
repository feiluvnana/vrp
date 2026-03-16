import math
import json
import os
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

# Load data parameters
def load_data(file_path="data.json"):
    """Load problem data from JSON."""
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, file_path)
    
    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"WARNING: {file_path} not found, using default data.")
    return {"requests": [], "truck_vel": 15.0, "drone_vel": 30.0, "truck_cap": 400.0, "drone_cap": 2.27, "drone_lim": 700.0, "truck_num": 1, "drone_num": 1}

# Global constants
DATA = load_data("data.json")
ALPHA = 1.0       # Distance weight
GAMMA = 10000.0   # Unserved request penalty
LW = 3600.0       # Max wait time (seconds)
DEPOT = [0.0, 0.0]

@dataclass(frozen=True)
class Request:
    """Customer request."""
    id: int
    x: float
    y: float
    demand: float
    can_drone: bool
    r_i: float      # Release time
    e_i: float      # Earliest time
    l_i: float      # Latest time

@dataclass
class Trip:
    """A trip route: Depot -> Stops -> Depot."""
    vtype: str  
    vidx: int   
    stops: List[Request] = field(default_factory=list)
    depart_time: float = 0.0
    return_time: float = 0.0
    total_dist: float = 0.0
    total_load: float = 0.0
    pis: Dict[int, float] = field(default_factory=dict)
    wis: Dict[int, float] = field(default_factory=dict)

    def eval(self, start_pos, vel, cap, dist_lim, Lw) -> bool:
        """Evaluate trip constraints (capacity, battery/distance, time windows, waiting time)."""
        pos = start_pos
        t = self.depart_time
        dist_acc = 0.0
        load_acc = 0.0
        pis = {}

        for req in self.stops:
            load_acc += req.demand
            if load_acc > cap:
                return False
            
            leg = math.sqrt((pos[0] - req.x)**2 + (pos[1] - req.y)**2)
            dist_acc += leg
            if dist_acc > dist_lim:
                return False
            
            arrival_time = t + leg / vel
            if arrival_time > req.l_i:
                return False
            
            pickup_time = max(arrival_time, req.e_i)
            pis[req.id] = pickup_time
            
            pos = [req.x, req.y]
            t = pickup_time

        leg_back = math.sqrt((pos[0] - 0.0)**2 + (pos[1] - 0.0)**2)
        dist_acc += leg_back
        if dist_acc > dist_lim:
            return False
        
        q_depot = t + leg_back / vel
        
        wis = {}
        for rid, p_i in pis.items():
            W_i = q_depot - p_i
            if W_i > Lw:
                return False
            wis[rid] = W_i
        
        self.return_time = q_depot
        self.total_dist = dist_acc
        self.total_load = load_acc
        self.pis = pis
        self.wis = wis
        return True

class VRPState:
    """VRP solution state."""
    def __init__(self, data, lw, alpha, gamma):
        self.data = data
        self.lw = lw
        self.alpha = alpha
        self.gamma = gamma
        
        self.requests = [
            Request(
                id=i, x=r[0], y=r[1], demand=r[2],
                can_drone=bool(r[3]), r_i=r[4], e_i=r[5], l_i=r[6]
            ) for i, r in enumerate(data["requests"])
        ]
        
        self.truck_trips: List[List[Trip]] = [[] for _ in range(data["truck_num"])]
        self.drone_trips: List[List[Trip]] = [[] for _ in range(data["drone_num"])]
        self.unserved: Set[Request] = set(self.requests)

    def objective(self) -> float:
        """Calculate objective value: total distance + penalties."""
        dist_truck = sum(t.total_dist for v_trips in self.truck_trips for t in v_trips)
        dist_drone = sum(t.total_dist for v_trips in self.drone_trips for t in v_trips)
        return self.alpha * (dist_truck + dist_drone) + self.gamma * len(self.unserved)

    def copy(self):
        """Deep copy for ALNS."""
        import copy
        return copy.deepcopy(self)

def print_solution(state: VRPState, name="SOLVER"):
    """Print results in JSON format."""
    dropped = sorted([r.id for r in state.unserved])
    v_idx = 0
    
    for trips in state.truck_trips:
        route_data = {}
        for t in trips:
            for s in t.stops:
                route_data[str(int(t.return_time))] = s.id
            route_data[str(int(t.return_time) + 1)] = 0 
        print(json.dumps({"__": "ROUTE", "_": "route_log", "vehicle": v_idx, "route": route_data, "dropped": dropped}))
        v_idx += 1
        
    for trips in state.drone_trips:
        route_data = {}
        for t in trips:
            for s in t.stops:
                route_data[str(int(t.return_time))] = s.id
            route_data[str(int(t.return_time) + 1)] = 0
        print(json.dumps({"__": "ROUTE", "_": "route_log", "vehicle": v_idx, "route": route_data, "dropped": dropped}))
        v_idx += 1

    obj_val = state.objective()
    num_dropped = len(state.unserved)
    print(json.dumps({"__": name, "_": "full_result", "result": [round(obj_val, 2), num_dropped]}, ensure_ascii=False))
