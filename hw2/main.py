"""Created: 2022/4/12"""
import networkx as nx
from itertools import combinations
import numpy as np

class Clique:
    def __init__(self, n, m=4):
        self.n = n  # total number of nodes
        self.m = m  # m is the number of nodes in a clique
        self.n_edges_clique = int((m-1)*m/2)  # number of edges in a clique
        self.G = nx.complete_graph(n)
        self.nodes = list(self.G.nodes)
        self.edges = list(self.G.edges)  # 暗含了这里的输出顺序的信息。
        self.cliques = list(combinations(self.nodes, self.m))
        self.cliques_edges = []
        self.edges_color = np.zeros(len(self.edges)) - 1  # -1 means uncolored
        for nodes in self.cliques:
            self.cliques_edges.append(list(combinations(nodes, 2)))


    def naive(self):
        """
        Naive solution without any refining.
        :return: answer for the problem
        """
        w_black = -1
        w_white = -1

        for i in range(len(self.edges_color)):
            # W_black
            self.edges_color[i] = 0
            for j in range(len(self.cliques)):
                w_black += self.indicator_variable(j)
            # W_white
            self.edges_color[i] = 1
            for j in range(len(self.cliques)):
                w_white += self.indicator_variable(j)
            if w_black < w_white:
                self.edges_color[i] = 0
            else:
                self.edges_color[i] = 1
        return self.edges_color



    def edge_ij_2_index(self, i, j):
        """Transforms the edge index to the index of the edge in the list of edges"""
        if i >= j:
            return -1
        return int((2*self.n-1-i) * i /2 + (j-i-1))


    def indicator_variable(self, i):
        """
        The indicator variable to indicate the prob of becoming a monochromatic K4 of the ith clique
        :param i: the index of the clique
        :return: the indicator variable
        """
        clique_edges = self.cliques_edges[i]
        colors_flag = -1
        count = 0
        for edge in clique_edges:  # search for color for every edges
            color = self.edges_color[self.edge_ij_2_index(*edge)]
            if color != -1:  # colored edge
                if colors_flag == -1:  # no colored found for the clique yet
                    colors_flag = color
                    count += 1
                elif color != colors_flag:  # impossible to be monochromatic clique
                    return 0
                else:  # the same color
                    count += 1
        if colors_flag == -1:  # no colored found for the clique yet
            return np.power(1/2, self.n_edges_clique-1)
        return np.power(1/2, self.n_edges_clique - count)

    def test(self):
        self.edges_color[0] = 0
        self.edges_color[1] = 0
        self.edges_color[2] = 0





if __name__ == '__main__':
    c = Clique(5)
    ans = c.solve()
    print(1)
