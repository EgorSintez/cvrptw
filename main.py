import argparse
import json
import math
import sys

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def cvrptw(**data):
    demands = data["demands"]
    depot_idx = data["depot_idx"]
    time_matrix = data["time_matrix"]
    time_windows = data["time_windows"]
    service_time = data["service_time"]
    capacity = data["capacity"]
    travel_time = data["travel_time"]
    timeout = data["timeout"]

    num_nodes = len(demands)

    assert num_nodes == len(time_matrix)
    assert num_nodes == len(time_windows)
    assert demands[depot_idx] == 0

    INF = 1_000_000

    reload_nodes = range(num_nodes, num_nodes + math.ceil(sum(demands) / capacity))
    num_vehicles = num_nodes

    total_nodes = num_nodes + len(reload_nodes)

    manager = pywrapcp.RoutingIndexManager(total_nodes, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    def demand_callback(i):
        node = manager.IndexToNode(i)
        if node in reload_nodes:
            return -capacity

        return demands[node]

    demand_evaluator_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimension(evaluator_index=demand_evaluator_index,
                         slack_max=capacity,
                         capacity=capacity,
                         fix_start_cumul_to_zero=True,
                         name="Capacity")

    capacity_dimension = routing.GetDimensionOrDie("Capacity")

    for i in range(num_nodes):
        capacity_dimension.SlackVar(i).SetValue(0)

    def transit_callback(i, j):
        from_node = manager.IndexToNode(i)
        to_node = manager.IndexToNode(j)

        if from_node in reload_nodes and to_node in reload_nodes:
            return INF

        if from_node in reload_nodes and to_node == depot_idx:
            return INF

        if from_node == depot_idx and to_node in reload_nodes:
            return INF

        if from_node in reload_nodes:
            from_node = depot_idx

        if to_node in reload_nodes:
            to_node = depot_idx

        return time_matrix[from_node][to_node] + service_time[from_node]

    cost_evaluator_index = routing.RegisterTransitCallback(transit_callback)

    for i in reload_nodes:
        node = manager.IndexToNode(i)
        routing.AddDisjunction([node], 0)

    routing.SetArcCostEvaluatorOfAllVehicles(cost_evaluator_index)

    routing.AddDimension(evaluator_index=cost_evaluator_index,
                         slack_max=0,
                         capacity=travel_time,
                         fix_start_cumul_to_zero=True,
                         name="Time")

    routing.AddDimension(evaluator_index=cost_evaluator_index,
                         slack_max=0,
                         capacity=INF,
                         fix_start_cumul_to_zero=False,
                         name="Window")

    windows_dimension = routing.GetDimensionOrDie("Window")

    for i, (a, b) in enumerate(time_windows):
        if i == depot_idx:
            continue

        if i in reload_nodes:
            continue

        node = manager.NodeToIndex(i)
        windows_dimension.CumulVar(node).SetRange(a, b)

    for i in range(num_nodes):
        routing.SetFixedCostOfVehicle(INF, i)

    params = pywrapcp.DefaultRoutingSearchParameters()

    params.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(timeout)

    solution = routing.SolveWithParameters(params)

    assert solution

    nodes, vehicles, time = [], [], []
    vehicle = 0

    time_dim = routing.GetDimensionOrDie("Window")

    for i in range(len(demands)):
        index = routing.Start(i)
        next_index = solution.Value(routing.NextVar(index))

        if routing.IsEnd(next_index):
            continue

        while True:
            time_var = time_dim.CumulVar(index)
            time_to = solution.Max(time_var)
            node = manager.IndexToNode(index)

            if node >= len(demands):
                node = depot_idx

            nodes.append(node)
            vehicles.append(vehicle)
            time.append(time_to)

            if routing.IsEnd(index):
                break

            index = solution.Value(routing.NextVar(index))
        vehicle += 1

    return {
        "nodes": nodes,
        "vehicles": vehicles,
        "time": time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=argparse.FileType())
    opt = parser.parse_args()

    req = json.load(opt.file)
    opt.file.close()

    res = cvrptw(**req)
    json.dump(res, sys.stdout, indent=2)
