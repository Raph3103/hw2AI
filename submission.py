from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, Robot, Package, ChargeStation
import random
import time
import math


def get_nearest_package(env: WarehouseEnv, robot: Robot) -> Package:
    if(len(env.packages) == 0):
        return None
    if(len(env.packages) == 1 and env.packages[0].on_board):
        return env.packages[0]
    if(len(env.packages) == 1 and not env.packages[0].on_board):
        return None
    if(len(env.packages) == 2 and env.packages[0].on_board and env.packages[1].on_board):
        minimum_distance_package = min(manhattan_distance(env.packages[0].position, robot.position), manhattan_distance(env.packages[1].position, robot.position))
        if(minimum_distance_package == manhattan_distance(env.packages[0].position, robot.position)):
            return env.packages[0]
        else:
            return env.packages[1]
    if(len(env.packages) == 2 and env.packages[0].on_board and not env.packages[1].on_board):
        return env.packages[0]
    if(len(env.packages) == 2 and not env.packages[0].on_board and env.packages[1].on_board):
        return env.packages[1]
    if(len(env.packages) == 2 and not env.packages[0].on_board and not env.packages[1].on_board):
        return None
    if(len(env.packages) > 2):
        available_packages = [p for p in env.packages if p.on_board]
        p = sorted(available_packages, key=lambda p: manhattan_distance(robot.position, p.position))[0]
        return p


def get_nearest_station(env: WarehouseEnv, robot: Robot) -> ChargeStation:
    if(len(env.charge_stations) == 0):
        return None
    if(len(env.charge_stations) == 1):
        return env.charge_stations[0]
    if(len(env.charge_stations) == 2):
        minimum_distance_station = min(manhattan_distance(env.charge_stations[0].position, robot.position), manhattan_distance(env.charge_stations[1].position, robot.position))
        if(minimum_distance_station == manhattan_distance(env.charge_stations[0].position, robot.position)):
            return env.charge_stations[0]
        else:
            return env.charge_stations[1]
    if(len(env.charge_stations) > 2):
        available_stations = [s for s in env.charge_stations]
        s = sorted(available_stations, key=lambda s: manhattan_distance(robot.position, s.position))[0]
        return s


def get_distance(env: WarehouseEnv, robot_id):
    if env.robot_is_occupied(robot_id):
        robot = env.get_robot(robot_id)
        opponent_dist = manhattan_distance(robot.position, robot.package.destination)
        return opponent_dist

    else:
        robot = env.get_robot(robot_id)
        nearest_package = get_nearest_package(env, robot)
        opponent_dist = manhattan_distance(robot.position, nearest_package.position)
        opponent_dist = opponent_dist + manhattan_distance(nearest_package.position, nearest_package.destination)
        return opponent_dist


def time_exceeded(start_time, time_limit):
    current_time = time.time()
    epsilon = 0.02
    if (current_time - start_time) >= (time_limit - epsilon):
        return True
    else:
        return False

    # TODO: section a : 3


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    opp_dist = get_distance(env, robot_id=(robot_id + 1) % 2)
    if robot.package is None:
        min_val = get_nearest_package(env, robot)
        if (manhattan_distance(robot.position, min_val.position) > robot.battery):
            return ((robot.credit * 1000) - manhattan_distance(robot.position, get_nearest_station(env,
                                                                                                   robot).position) + 10 * robot.battery)

        return (robot.credit * 100) - manhattan_distance(robot.position, min_val.position) + 10 * robot.battery
    else:

        return ((robot.credit * 100) - manhattan_distance(robot.position,robot.package.destination) + 10 * robot.battery) - opp_dist


class AgentMinimax(Agent):
    # TODO: section b : 1
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def run_step_aux(self, env: WarehouseEnv, agent_id, time_limit, depth):

        possibilities = env.get_legal_operators(agent_id)
        children = []
        for possibility in possibilities:
            copy = env.clone()
            copy.apply_operator(agent_id, possibility)
            children.append(copy)

        children_heuristics = []
        opponent_id = (agent_id + 1) % 2
        Turn = opponent_id
        for child in children:
            children_heuristics = [self.RB_minimax(child, agent_id, Turn, depth) for child in children]
        max_heuristic = max(children_heuristics)
        i = 0
        for child in children_heuristics:
            if max_heuristic == children_heuristics[i]:
                return possibilities[i]
            i += 1

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit

        depth = 0
        while True:

            next_move = self.run_step_aux(env, agent_id, time_limit, depth)
            depth = depth + 1
            if time_exceeded(self.start_time, time_limit) == True:
                return next_move

    def RB_minimax(self, env: WarehouseEnv, agent_id, Turn, depth):

        if time_exceeded(self.start_time, self.time_limit):
            return smart_heuristic(env, agent_id)
        if env.done() or depth == 0:
            return smart_heuristic(env, agent_id)

        possibilities = env.get_legal_operators(Turn)
        children = []
        for possibility in possibilities:
            copy = env.clone()
            copy.apply_operator(Turn, possibility)
            children.append(copy)

        if Turn == agent_id:
            current_max = - float('inf')
            for child in children:
                v = self.RB_minimax(child, agent_id, (Turn + 1) % 2, depth - 1)
                current_max = max(v, current_max)
            return current_max

        else:
            current_min = float('inf')
            for child in children:
                v = self.RB_minimax(child, agent_id, (Turn + 1) % 2, (depth - 1))
                current_min = min(v, current_min)
            return current_min


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def __init__(self):
        self.start_time = None
        self.time_limit = None

    def run_step_aux(self, env: WarehouseEnv, agent_id, time_limit, depth):

        infinite = float('inf')
        minus_inf = - float('inf')

        possibilities = env.get_legal_operators(agent_id)
        children = []
        for possibility in possibilities:
            copy = env.clone()
            copy.apply_operator(agent_id, possibility)
            children.append(copy)

        children_heuristics = []
        opponent_id = (agent_id + 1) % 2
        next_turn = opponent_id
        for child in children:
            children_heuristics = [self.RB_AlphaBeta(child, agent_id, next_turn, depth, minus_inf, infinite) for child
                                   in children]
        max_heuristic = max(children_heuristics)

        i = 0
        for child in (children_heuristics):
            if max_heuristic == children_heuristics[i]:
                return possibilities[i]
            i += 1

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit

        depth = 0
        while True:

            next_move = self.run_step_aux(env, agent_id, time_limit, depth)
            depth = depth + 1
            if time_exceeded(self.start_time, time_limit) == True:
                return next_move

    def RB_AlphaBeta(self, env: WarehouseEnv, agent_id, turn, depth, alpha, beta):

        if time_exceeded(self.start_time, self.time_limit):
            return smart_heuristic(env, agent_id)
        if env.done() or depth == 0:
            return smart_heuristic(env, agent_id)

        possibilities = env.get_legal_operators(turn)
        children = []
        for possibility in possibilities:
            copy = env.clone()
            copy.apply_operator(turn, possibility)
            children.append(copy)

        next_turn = (turn + 1) % 2

        if turn == agent_id:
            current_max = - float('inf')
            for child in children:
                v = self.RB_AlphaBeta(child, agent_id, next_turn, depth - 1, alpha, beta)
                current_max = max(v, current_max)
                alpha = max(current_max, alpha)
                if current_max >= beta:
                    return float('inf')

            return current_max

        else:
            current_min = float('inf')
            for child in children:
                v = self.RB_AlphaBeta(child, agent_id, next_turn, (depth - 1), alpha, beta)
                current_min = min(v, current_min)
                beta = min(current_min, beta)
                if current_min <= alpha:
                    return (- float('inf'))
            return current_min


class AgentExpectimax(Agent):
    def run_step_aux(self, env: WarehouseEnv, agent_id, time_limit, depth):
        possibilities = env.get_legal_operators(agent_id)
        children = []
        for possibility in possibilities:
            copy = env.clone()
            copy.apply_operator(agent_id, possibility)
            children.append(copy)

        children_heuristics = []
        opponent_id = (agent_id + 1) % 2
        Turn = opponent_id
        for child in children:
            children_heuristics = [self.RB_Expectimax(child, agent_id, Turn, depth) for child in children]
        max_heuristic = max(children_heuristics)
        i = 0
        for child in children_heuristics:
            if max_heuristic == children_heuristics[i]:
                return possibilities[i]
            i += 1


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.start_time = time.time()
        self.time_limit = time_limit

        depth = 0
        while True:

            next_move = self.run_step_aux(env, agent_id, time_limit, depth)
            depth = depth + 1
            if time_exceeded(self.start_time, time_limit) == True:
                return next_move

    def RB_Expectimax(self, env: WarehouseEnv, agent_id, turn, depth):
        if time_exceeded(self.start_time, self.time_limit):
            return smart_heuristic(env, agent_id)
        if env.done() or depth == 0:
            return smart_heuristic(env, agent_id)
        possibilities = env.get_legal_operators(turn)
        children = []
        sum_weights = len(possibilities)
        probabilities = [1] * len(possibilities)
        for possibility in possibilities:
            copy = env.clone()
            copy.apply_operator(turn, possibility)
            children.append(copy)

        if turn == agent_id:
            curr_max = -math.inf
            for c in children:
                v = self.RB_Expectimax(c, agent_id, depth - 1, turn)
                curr_max = max(v, curr_max)
            return curr_max

        else:
            stations_position = [charge.position for charge in env.charge_stations]
            for index, child in enumerate(children):
                if child.get_robot(turn).position in stations_position:
                    probabilities[index] += 1
                    sum_weights += 1

                for p in probabilities:
                    p = p / sum_weights

                    sum_to_ret = 0
                    for index, child in enumerate(children):
                        sum_to_ret += probabilities[index] * self.RB_Expectimax(child, agent_id, turn, depth - 1)

                return sum_to_ret/len(children)


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)

