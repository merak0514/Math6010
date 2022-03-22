"""Created: 2022/3/22"""
from MinDominatingSetLib import *


if __name__ == '__main__':
    m = MinDominatingSet()
    m.find_dominating_set()
    print(m.dominating_set)
    m.draw_graph(save_path='img/result.jpg')
    m.generate_gif()
