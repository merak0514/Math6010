"""Created: 2022/3/22"""
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import imageio
import os

np.random.seed(777)


class MinDominatingSet:
    def __init__(self, p=0.2, nodes_num=16):
        self.P = p
        self.nodes_num = nodes_num
        self.G = nx.Graph()
        self.G.add_nodes_from(list(range(self.nodes_num)))
        self.generate_random_edges()
        self.dominating_set = []  # 选定为集合内的点
        self.undominated_set = set(range(nodes_num))  # 未被dom的点
        self.neibour_and_undomed = {}  # 维护一个邻接且尚未被dom的点的dict，方便后续查找
        for i in range(self.nodes_num):
            self.neibour_and_undomed[i] = list(self.G.adj[i].keys())

    def find_dominating_set(self):
        first_point = np.argmax(self.G.degree, axis=0)[1]  # 找到其中度数最大的点
        done = self.update(first_point)
        if done:
            return

        while True:
            i_i = self.inverted_index(self.neibour_and_undomed)
            if i_i:
                selected = max(i_i, key=i_i.get)
            else:
                selected = np.random.choice(list(self.undominated_set))
            done = self.update(selected)
            if done:
                self.default_layout = nx.spring_layout(self.G)
                return

    def update(self, selected):
        self.dominating_set.append(selected)  # 选定为集合内的点
        self.undominated_set.discard(selected)  # 将选定的点从未被dom的集合中删除
        adjs = list(self.G.adj[selected].keys())  # 这个点的邻接点
        self.undominated_set.difference_update(adjs)  # 把这个点邻接的都加进来。
        if len(self.undominated_set) == 0:
            return True

        # 以下三行维护邻接undom表。
        self.neibour_and_undomed[selected] = []
        for i in adjs:
            if selected in self.neibour_and_undomed[i]:
                self.neibour_and_undomed[i].remove(selected)
        return False

    @staticmethod
    def inverted_index(self, d: dict):
        # 本质是倒排索引后找到最长的一个
        _inverted_index = {}
        for k, v in d.items():
            for i in v:
                _inverted_index[i] = _inverted_index.get(i, 0) + 1
        return _inverted_index

    def generate_random_edges(self):
        self.G.remove_edges_from(self.G.edges())  # clear edges
        adjacency_matrix = np.random.rand(self.nodes_num, self.nodes_num)
        adjacency_matrix = np.triu(adjacency_matrix)
        mask_1 = np.where(adjacency_matrix >= 1 - self.P)
        adjacency_matrix[mask_1] = 1
        for i in range(len(mask_1[0])):
            if mask_1[0][i] != mask_1[1][i]:  # 无自环
                self.G.add_edge(mask_1[0][i], mask_1[1][i])

    def draw_graph(self, save_path=None, colored_nodes=-1, layout=None):
        if not layout:
            layout = self.default_layout
        if colored_nodes == -1:
            colors = ['#fc9272' if node in self.dominating_set else '#bcbddc' for node in self.G.nodes]
        else:  # 为差分绘图准备
            count = 0
            colors = []
            strange_colored_nodes = self.dominating_set[:colored_nodes]
            for node in range(self.nodes_num):
                if node in strange_colored_nodes and count < colored_nodes:
                    colors.append('#fc9272')
                    count += 1
                else:
                    colors.append('#bcbddc')
        nx.draw(self.G, layout, node_color=colors, with_labels=True, node_size=1000)
        # plt.show()
        if save_path:
            plt.savefig(save_path)
        plt.clf()

    def generate_gif(self):
        gif_images = []
        for i in range(len(self.dominating_set) + 1):
            self.draw_graph(save_path=f'img/{i}.jpg', colored_nodes=i)
            gif_images.append(imageio.imread(f'img/{i}.jpg'))
            os.remove(f'img/{i}.jpg')
        imageio.mimsave("test.gif", gif_images, fps=1)


if __name__ == '__main__':
    m = MinDominatingSet()
    # nx.draw(m.G, with_labels=True)
    # plt.show()
    m.find_dominating_set()
    print(m.dominating_set)
    m.draw_graph(save_path='img/result.jpg')
    m.generate_gif()
