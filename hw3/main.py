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
    def __init__(self, n=20, max_weight=100):
        self.max_weight = max_weight  # weight of each edge lies between [0,multiplier]
        self.n = n  # total number of nodes
        self.G = nx.complete_graph(n)
        self.nodes = np.array(self.G.nodes)
        self.edges = list(self.G.edges)
        self.edges_cost = np.zeros([self.n, self.n])
        self.init_cost()

    def init_cost(self):
        for v1, v2 in self.edges:
            weight = int(np.random.rand() * self.max_weight)
            self.edges_cost[v1, v2] = weight
            self.edges_cost[v2, v1] = weight

    def compute_cost(self, partition_1, partition_2) -> int:
        cost = 0
        for v1 in partition_1:
            for v2 in partition_2:
                cost += self.edges_cost[v1, v2]
        return cost

    def brute_force(self):
        t0 = time.time()
        all_possible = combinations(self.nodes, 10)
        min_cost = (self.n / 2) ** 2 * self.max_weight
        for p1 in all_possible:
            p2 = self.nodes.copy()
            p2 = np.delete(p2, p1)
            cost = self.compute_cost(p1, p2)
            min_cost = min(min_cost, cost)
        t = time.time() - t0
        return min_cost, t

    def simulated_annealing(self, c_max=None, T0=None, alpha=0.997):
        t0 = time.time()
        if T0 is None:
            T0 = np.sqrt(self.n) * self.max_weight / 2
        if c_max is None:
            c_max = 100 * self.n
        T = T0

        # step 1: init solution
        np.random.shuffle(self.nodes)  # inplace
        p1: np.ndarray
        p2: np.ndarray
        p1, p2 = np.split(self.nodes, 2)  # partition 1, partition 2
        cost = self.compute_cost(p1, p2)

        for step in range(c_max):
            # step 2: randomly pick a v1 in p1 and a v2 in p2
            new_p1: np.ndarray
            new_p2: np.ndarray
            new_p1 = p1.copy()
            new_p2 = p2.copy()
            v1 = np.random.choice(new_p1)
            v2 = np.random.choice(new_p2)
            new_p1 = np.append(new_p1[new_p1 != v1], v2)  # del v1 in p1 and add v2 to p1
            new_p2 = np.append(new_p2[new_p2 != v2], v1)
            new_cost = self.compute_cost(new_p1, new_p2)
            # step 3: accept or refuse
            if new_cost < cost:
                # accept the better solution
                cost = new_cost
                p1 = new_p1.copy()
                p2 = new_p2.copy()
            else:
                if np.random.rand() < np.exp((cost - new_cost) / T):
                    # accept the worse solution
                    cost = new_cost
                    p1 = new_p1.copy()
                    p2 = new_p2.copy()
                else:
                    # refuse the worse solution
                    pass
            T *= alpha  # step 4: cooling
        t = time.time() - t0
        return p1, p2, cost, t

    def draw_graph(self, p1, p2):
        layout = nx.bipartite_layout(self.G, p1)
        nx.draw(self.G, layout, with_labels=True,
                node_color='#edf8b1', node_size=600, edgecolors='black')
        cut = nx.Graph()
        cut.add_nodes_from(list(range(self.n)))
        for v1 in p1:
            for v2 in p2:
                cut.add_edge(v1, v2)
        # nx.draw_networkx_edges(cut, layout, width=4, edge_color='tab:red', alpha=0.5)
        plt.show()


if __name__ == '__main__':
    cg = CompleteGraph()
    c1, t1 = cg.brute_force()
    p1, p2, c2, t2 = cg.simulated_annealing()
    cg.draw_graph(p1, p2)
    print(f"Done! The true answer from brute force solution: {c1}, with time {t1}; \n"
          f"The answer from simulated annealing solution: {c2}, with time {t2}")
    table = [["algorithm", "min_cost", "time"], ["Brute force", c1, t1], ["simulated annealing", c2, t2]]
    print(tabulate(table, tablefmt="fancy_grid"))
