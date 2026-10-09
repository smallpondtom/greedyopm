import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

import sys

sys.path.append("visualization")
from utils import *

plt.rc('text', usetex=False)
plt.rc('font', family='serif')
def tree_visualization(config):

    path_results = config["path_results"]

    with open(path_results + "/Decoder.pkl", "rb") as f:
        D = pickle.load(f)

    func = D.functions_f
    index_learnt = np.array([ind["index"] for ind in func.values()])

    indices_to_plot = [x-1 for x in config["plot"]["indices_to_plot"]]

    if indices_to_plot == -1:
        indices_to_plot = list(index_learnt)
    else:
        indices_to_plot = list(indices_to_plot)

    for k in indices_to_plot:
        if k in index_learnt:
            nodes = {}
            key = get_key(func, k)
            nb_deps_k = func[key]["nb_deps"]
            list_non_learnt_coeffs = []
            for t in range(1, k+2):
                nodes[f"{t}"] = TreeNode(f"{t}", values=t)
            for i in range(nb_deps_k):
                if i in index_learnt:
                    key = get_key(func, i)
                    nb_deps_i = func[key]["nb_deps"]
                    for j in range(nb_deps_i):
                        nodes[f"{j+1}"].connect_to(nodes[f"{i+1}"])
                else:
                    list_non_learnt_coeffs.append(i)

                nodes[f"{i+1}"].connect_to(nodes[f"{k+1}"])
        else:
            print(f"\n Coefficient {k+1} has not been learnt or does not exist")
            continue

        root_nodes = []
        for t in list_non_learnt_coeffs:
            root_nodes.append(nodes[f"{t+1}"])
        G = build_directed_graph_from_tree(root_nodes)

        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer

        pos = nx.multipartite_layout(G, subset_key="layer")
        # scaled_pos = {node: (x * 0.5, y*0.8) for node, (x, y) in pos.items()}

        fig, ax = plt.subplots()
        nx.draw_networkx(G, pos=pos, ax=ax, with_labels=False, node_color="skyblue", node_size=700)
        nx.draw_networkx_labels(G, pos=pos, font_weight="bold", font_size=15)
        # ax.set_title("Graph structure")
        fig.tight_layout()
        plt.show()
