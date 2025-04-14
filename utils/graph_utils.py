import os
import re
from copy import deepcopy

import yaml
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from networkx.algorithms.cycles import simple_cycles
from tqdm import tqdm  # For progress tracking
import numpy as np
from collections import deque
import igraph as ig
import math
import pandas as pd
import json
from datasets import load_dataset
from tqdm.auto import tqdm
import ast
import io
from PIL import Image

tqdm.pandas()


def get_ccs_from_digraph_based_on_cc(G):
    components = list(nx.weakly_connected_components(G))
    ccs = []
    for comp in tqdm(components, "Extracting disjoint components (based on weakly connected components) from the graph..."):
        subG = G.subgraph(comp).copy()
        ccs.append(subG)
    return ccs


def get_ccs_from_digraph_based_on_roots(G):
    roots = [node for node in G.nodes if G.in_degree(node) == 0]
    ccs = []
    for root in tqdm(roots, "Extracting disjoint components (based on tree roots) from the graph..."):
        subtree_nodes = nx.dfs_tree(G, source=root).nodes
        subtree = G.subgraph(subtree_nodes).copy()
        ccs.append(subtree)
    return ccs


def remove_self_loops_and_cycles(model_graph):
    print("Removing self-loops and cycles...")

    self_loops = list(nx.selfloop_edges(model_graph))
    print(f"Found {len(self_loops)} self-loops, removing them...")
    model_graph.remove_edges_from(self_loops)

    cycles = list(nx.simple_cycles(model_graph))
    # TODO: Importamt, how do we remove the cycles? I think that removing the first edge may not be enough, e.g., if it is a merge of 3 models, a root and 2 others that are cildren of the root. We need to remove the root, not the others
    print(f"Found {len(cycles)} cycles, removing them...")
    for cycle in cycles:
        # TODO: Find the latest node and remove and edge from it
        model_graph.remove_edge(cycle[0], cycle[1])
    return model_graph