"""Created: 2022/4/28"""
import networkx as nx
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple
from tqdm import tqdm
import time
from tabulate import tabulate

np.random.seed(777)


class CompleteGraph:
    def __init__(self, n=20):
        self.n = n  # total number of nodes
        self.G = nx.complete_graph(n)
        self.nodes = np.array(self.G.nodes)
        self.edges = list(self.G.edges)
        self.edges_cost = np.zeros([self.n, self.n])
        self.init_cost()
        print(1)

    def init_cost(self):
        for v1, v2 in self.edges:
            weight = int(np.random.rand() * 100)
            self.edges_cost[v1, v2] = weight
            self.edges_cost[v2, v1] = weight

    def compute_cost(self, partition_1, partition_2) -> int:
        cost = 0
        for v1 in partition_1:
            for v2 in partition_2:
                cost += self.edges_cost[v1, v2]
        return cost

    def brute_force(self):
        all_possible = combinations(self.nodes, 10)
        min_cost = (self.n/2) ** 2 * 100
        for p1 in all_possible:
            p2 = self.nodes.copy()
            p2 = np.delete(p2, p1)
            cost = self.compute_cost(p1, p2)
            min_cost = min(min_cost, cost)

        print(min_cost)
        return min_cost

    def simulated_annealing(self, c_max=5000, T0=None, alpha=0.997):
        if T0 is None:
            T0 = np.sqrt(self.n) * 100
        T = T0

        # init solution
        np.random.shuffle(self.nodes)  # inplace
        p1: np.ndarray
        p2: np.ndarray
        p1, p2 = np.split(self.nodes, 2)  # partition 1, partition 2
        cost = self.compute_cost(p1, p2)

        for step in range(c_max):
            # randomly pick a v1 in p1 and a v2 in p2
            new_p1: np.ndarray
            new_p2: np.ndarray
            new_p1 = p1.copy()
            new_p2 = p2.copy()
            v1 = np.random.choice(new_p1)
            v2 = np.random.choice(new_p2)
            new_p1 = np.append(new_p1[new_p1!=v1], v2)  # del v1 in p1 and add v2 to p1
            new_p2 = np.append(new_p2[new_p2!=v2], v1)
            new_cost = self.compute_cost(new_p1, new_p2)
            if new_cost < cost:
                # accept the better solution
                cost = new_cost
                p1 = new_p1.copy()
                p2 = new_p2.copy()
            else:
                if np.random.rand() < np.exp((cost-new_cost) / T):
                    print(step, cost-new_cost, np.exp((cost-new_cost) / T))
                    # accept the worse solution
                    cost = new_cost
                    p1 = new_p1.copy()
                    p2 = new_p2.copy()
                else:
                    # refuse the worse solution
                    pass
            T *= alpha  # annealing
            print(f"The cost is: {cost}, p1 is: {p1}, p2 is: {p2}")
        return p1, p2, cost


if __name__ == '__main__':
    cg = CompleteGraph()
    cg.brute_force()
    cg.simulated_annealing()

