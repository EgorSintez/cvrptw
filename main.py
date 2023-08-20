import argparse
import json
import math
import pprint

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def solve(nodes: int,
          vehicles: int,
          depot: int,
          demands: list,
          time_matrix: list,
          time_windows: list,
          service_time: list,
          capacity: int,
          travel_time: int,
          timeout: int):

    assert demands[depot] == 0
    assert nodes == len(time_matrix)
    assert nodes == len(time_windows)

    INF = 1_000_000

    reload_nodes = range(nodes, nodes + math.ceil(sum(demands) / capacity))
    total_nodes = nodes + len(reload_nodes)

    manager = pywrapcp.RoutingIndexManager(total_nodes, vehicles, depot)
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

    for i in range(nodes):
        capacity_dimension.SlackVar(i).SetValue(0)

    def transit_callback(i, j):
        from_node = manager.IndexToNode(i)
        to_node = manager.IndexToNode(j)

        if from_node in reload_nodes and to_node in reload_nodes:
            return INF

        if from_node in reload_nodes and to_node == depot:
            return INF

        if from_node == depot and to_node in reload_nodes:
            return INF

        if from_node in reload_nodes:
            from_node = depot

        if to_node in reload_nodes:
            to_node = depot

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
                         name="TravelTime")

    routing.AddDimension(evaluator_index=cost_evaluator_index,
                         slack_max=0,
                         capacity=INF,
                         fix_start_cumul_to_zero=False,
                         name="TimeWindow")

    windows_dim = routing.GetDimensionOrDie("TimeWindow")

    for i, (a, b) in enumerate(time_windows):
        if i == depot:
            continue

        if i in reload_nodes:
            continue

        node = manager.NodeToIndex(i)
        windows_dim.CumulVar(node).SetRange(a, b)

    for i in range(vehicles):
        routing.SetFixedCostOfVehicle(INF, i)

    params = pywrapcp.DefaultRoutingSearchParameters()

    params.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
    params.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(timeout)

    solution = routing.SolveWithParameters(params)

    assert solution

    time_dim = routing.GetDimensionOrDie("TimeWindow")

    nodes_, vehicles_, time_ = [], [], []

    for i in range(vehicles):
        index = routing.Start(i)

        while True:
            time_var = time_dim.CumulVar(index)
            time_to = solution.Max(time_var)

            time_.append(time_to)
            vehicles_.append(i)

            node = manager.IndexToNode(index)
            node = depot if node in reload_nodes else node
            nodes_.append(node)

            if routing.IsEnd(index):
                break

            index = solution.Value(routing.NextVar(index))

    return nodes_, vehicles_, time_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=argparse.FileType())
    opt = parser.parse_args()

    req = json.load(opt.file)
    opt.file.close()

    nodes, vehicles, time = solve(**req)
    print(pprint.pformat({"nodes": nodes, "vehicles": vehicles, "time": time}, width=100).replace("'","\""))
