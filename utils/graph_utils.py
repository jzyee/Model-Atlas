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
    """Removes self-loops and cycles from the graph."""
    print("Removing self-loops and cycles...")
    self_loops = list(nx.selfloop_edges(model_graph))
    if self_loops:
        print(f"Found {len(self_loops)} self-loops, removing them...")
        model_graph.remove_edges_from(self_loops)

    # Remove cycles
    cycles_removed_count = 0
    while True:
        try:
            cycle = nx.find_cycle(model_graph, orientation="original")
            if not cycle:
                break
            print(f"Found cycle with {len(cycle)} edges: {cycle}")
            
            # cycle is a list of edges, each edge is (u, v) for simple graphs
            # Let's try to remove the first edge in the cycle
            first_edge = cycle[0]
            
            # Handle different possible formats
            if len(first_edge) >= 2:
                u, v = first_edge[0], first_edge[1]
                print(f"Attempting to remove edge: {u} -> {v}")
                
                # Check if the edge actually exists before trying to remove it
                if model_graph.has_edge(u, v):
                    try:
                        model_graph.remove_edge(u, v)
                        print(f"Successfully removed edge: {u} -> {v}")
                        cycles_removed_count += 1
                    except nx.NetworkXError as e:
                        print(f"Error removing edge ({u}, {v}): {e}")
                        print(f"Edge exists check: {model_graph.has_edge(u, v)}")
                        print(f"Node {u} exists: {u in model_graph.nodes}")
                        print(f"Node {v} exists: {v in model_graph.nodes}")
                        break
                else:
                    print(f"Edge ({u}, {v}) does not exist in graph!")
                    print(f"Available edges from {u}: {list(model_graph.successors(u)) if u in model_graph.nodes else 'Node not found'}")
                    print(f"Available edges to {v}: {list(model_graph.predecessors(v)) if v in model_graph.nodes else 'Node not found'}")
                    break
            else:
                print(f"Unexpected edge format: {first_edge}")
                break
                
        except nx.NetworkXNoCycle:
            print("No more cycles found.")
            break
        except Exception as e:
            print(f"An unexpected error occurred during cycle detection/removal: {e}")
            if 'cycle' in locals() and cycle:
                print(f"Problematic cycle data: {cycle}")
            break

    if cycles_removed_count > 0:
        print(f"Removed {cycles_removed_count} cycles.")
    else:
        print("No cycles were removed (or an error occurred).")

    print(f"Graph after removing self-loops and cycles has {model_graph.number_of_nodes()} nodes and {model_graph.number_of_edges()} edges.")
    return model_graph