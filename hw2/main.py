"""Created: 2022/4/12"""
import networkx as nx
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple

np.random.seed(777)


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
        self.edges_2_cliques = self.inverted_index(self.cliques_edges)
        self.indicator_memory = []

        # vars needed during drawing
        self.default_layout = nx.spring_layout(self.G)
        self.white = 'tab:red'
        self.black = 'tab:blue'
        self.node_color = "#e5f5e0"
        self.node_edge_color = "gray"


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


    def solve(self):
        """
        better solution
        :return: answer for the problem
        """
        # init indicator memory
        for i in range(len(self.cliques)):
            self.indicator_memory.append(self.indicator_variable(i))
        w_black = -1
        w_white = -1

        for edge_index in range(len(self.edges_color)):
            # W_black
            self.edges_color[edge_index] = 0
            for clique in self.edges_2_cliques[edge_index]:
                self.indicator_memory[clique] = self.indicator_variable(clique)
                w_black = sum(self.indicator_memory)
            # W_white
            self.edges_color[edge_index] = 1
            for clique in self.edges_2_cliques[edge_index]:
                self.indicator_memory[clique] = self.indicator_variable(clique)
                w_white = sum(self.indicator_memory)
            if w_black < w_white:
                self.edges_color[edge_index] = 0
            else:
                self.edges_color[edge_index] = 1
        return self.edges_color

    def table(self):
        """
        better solution
        :return: answer for the problem
        """
        # init indicator memory
        for i in range(len(self.cliques)):
            self.indicator_memory.append(np.power(1/2, 5))
        self.cliques_color = [-1]*len(self.cliques)  # init with -1

        for edge_index in range(len(self.edges_color)):
            w_black = 0
            w_white = 0
            # W_black
            black_indicator_memory = self.indicator_memory.copy()
            black_cliques_color = self.cliques_color.copy()
            self.edges_color[edge_index] = 0
            for clique in self.edges_2_cliques[edge_index]:
                if black_indicator_memory[clique] == 0:  # already heterochromatic
                    continue
                if black_cliques_color[clique] == -1:  # no color for the clique yet
                    black_cliques_color[clique] = 0
                    w_black += self.indicator_memory[clique]
                    continue
                else:
                    if black_cliques_color[clique] != 0:  # not monochromatic now
                        black_indicator_memory[clique] = 0
                    else:  # still monochromatic now
                        black_indicator_memory[clique] = black_indicator_memory[clique] * 2
                        w_black += black_indicator_memory[clique]

            # W_white
            white_indicator_memory = self.indicator_memory.copy()
            white_cliques_color = self.cliques_color.copy()
            self.edges_color[edge_index] = 1
            for clique in self.edges_2_cliques[edge_index]:
                if white_indicator_memory[clique] == 0:  # already heterochromatic
                    continue
                if white_cliques_color[clique] == -1:  # no color for the clique yet
                    white_cliques_color[clique] = 1
                    w_white += self.indicator_memory[clique]
                    continue
                else:
                    if white_cliques_color[clique] != 1:  # not monochromatic now
                        white_indicator_memory[clique] = 0
                    else:  # still monochromatic now
                        white_indicator_memory[clique] = white_indicator_memory[clique] * 2
                        w_white += white_indicator_memory[clique]

            if w_black < w_white:  # color as black
                self.edges_color[edge_index] = 0
                self.indicator_memory = black_indicator_memory.copy()
                self.cliques_color = black_cliques_color.copy()
            else:  # color as white
                self.edges_color[edge_index] = 1
                self.indicator_memory = white_indicator_memory.copy()
                self.cliques_color = white_cliques_color.copy()
        return self.edges_color

    def inverted_index(self, cliques: List[List[Tuple]]):
        """
        本质是倒排索引
        :param cliques: 其中每个元素是这个clique中对应的边的集合
        :return:
        """
        _inverted_index = {}
        for i in range(len(cliques)):
            edges = cliques[i]
            for edge in edges:
                index = self.edge_ij_2_index(*edge)  # index of edge
                _inverted_index[index] = _inverted_index.get(index, [])
                _inverted_index[index].append(i)  # add the clique to the edge

        return _inverted_index

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


    def draw_graph(self):
        draw_edges_color = []
        for c in self.edges_color:
            if c == 1:
                draw_edges_color.append(self.white)
            else:
                draw_edges_color.append(self.black)
        nx.draw(self.G, self.default_layout, with_labels=True, node_size=800,
                node_color=self.node_color,
                edgecolors=self.node_edge_color,
                edge_color=draw_edges_color,
                width=2)
        # nx.draw_networkx_edges(self.G,self.default_layout, width=8, edge_color=draw_edges_color, alpha=0.5)
        plt.show()




if __name__ == '__main__':
    c = Clique(200)
    ans = c.table()
    c.draw_graph()
    print(1)
