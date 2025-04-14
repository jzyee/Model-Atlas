import os
import re
import math
import yaml
import random
import pickle
import argparse
import pandas as pd
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from tqdm.auto import tqdm
from datetime import datetime
from functools import lru_cache

tqdm.pandas()

def compute_subtree_stats_dfs(G, node, memo):
    """
    Compute aggregated stats for the descendants of a node.
    Returns a dict with keys:
      - monthly_downloads: sum of downloads for all descendants
      - all_time_downloads: sum of all_time_downloads for all descendants
      - subgraph_size: number of descendants (not counting the node itself)
      - subgraph_merges: number of descendants with base_model_relation == "merge"
    The node's own values are not included; they can be added separately.
    """
    if node in memo:
        return memo[node]

    total_monthly = 0
    total_all_time = 0
    total_merges = 0
    total_size = 0

    for child in G.successors(node):
        # Get the child's own values
        try:
            child_monthly = int(G.nodes[child].get("downloads", "0"))
        except Exception:
            child_monthly = 0
        try:
            child_all_time = int(G.nodes[child].get("downloadsAllTime", "0"))
        except Exception:
            child_all_time = 0
        child_merge = 1 if G.nodes[child].get("base_model_relation", "unknown") == "merge" else 0

        total_monthly += child_monthly
        total_all_time += child_all_time
        total_merges += child_merge
        total_size += 1  # count this child

        # Recursively add the child's descendant stats
        child_stats = compute_subtree_stats_dfs(G, child, memo)
        total_monthly += child_stats["monthly_downloads"]
        total_all_time += child_stats["all_time_downloads"]
        total_merges += child_stats["subgraph_merges"]
        total_size += child_stats["subgraph_size"]

    memo[node] = {
        "monthly_downloads": total_monthly,
        "all_time_downloads": total_all_time,
        "subgraph_size": total_size,
        "subgraph_merges": total_merges
    }
    return memo[node]

def main(args):
    processed_dataset_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}.csv")
    dataset = pd.read_csv(processed_dataset_path, low_memory=False)
    trees_save_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}___cc_breakdown___root_type_{args.root_type}.pkl")

    exported_graphml_dir = os.path.join(args.output_cc_dir, f"{args.processed_hub_stats_fname}___cc_breakdown___root_type_{args.root_type}", "graphml")
    exported_networkx_dir = os.path.join(args.output_cc_dir, f"{args.processed_hub_stats_fname}___cc_breakdown___root_type_{args.root_type}", "networkx_pkl")
    os.makedirs(exported_graphml_dir, exist_ok=True)
    os.makedirs(exported_networkx_dir, exist_ok=True)

    with open(trees_save_path, 'rb') as f:
        trees = pickle.load(f)

    filtered_dataset = dataset[dataset['id'].isin([node_id for tree in trees for node_id in tree.nodes()])]
    filtered_dataset = dataset[dataset['base_model'].isna() & (dataset['base_model_relation'] != "root")]
    filtered_dataset = filtered_dataset.sort_values(by='downloads', ascending=False)

    graphs_and_roots = {tree_to_insp: [node for node in tree_to_insp.nodes if tree_to_insp.in_degree(node) == 0] for tree_to_insp in trees}
    graphs_and_roots = {k: v for k, v in graphs_and_roots.items() if len(k) > 1}
    graphs_and_roots2 = {}
    for k, v in tqdm(sorted(graphs_and_roots.items(), key=lambda item: len(item[0]), reverse=True), desc="Sorting roots..."):
        for node in v:
            val = k.nodes[node].get("createdAt", None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                k.nodes[node]["createdAt"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        all_roots = sorted(v, key=lambda item: datetime.strptime(k.nodes[item].get("createdAt", datetime.now().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S').timestamp())
        graphs_and_roots2[k] = all_roots

    graphs_and_roots = graphs_and_roots2
    roots_and_graphs = {v[0]: k for k, v in sorted(graphs_and_roots.items(), key=lambda item: len(item[0]), reverse=True)}

    # Filter out trees with less than 1 nodes
    trees = [tree for tree in trees if len(tree.nodes) > 1]
    trees = sorted(trees, key=lambda x: len(x), reverse=True)

    for tree in tqdm(trees, desc="Saving networkx trees..."):
        first_root = [node for node in tree.nodes if tree.in_degree(node) == 0][0]
        with open(os.path.join(exported_networkx_dir, f"n_nodes_{len(tree.nodes):05d}___root_{first_root.replace('/', '___')}.pkl"), 'wb') as f:
            pickle.dump({"cc": tree}, f)

    node_atts_to_del = ['trendingScore',
                        'config',
                        'adapterhub',
                        'siblings',
                        'region',
                        "model_cls_name",
                        "modality",
                        "anndata_version",
                        "tissue",
                        "annotated",
                        "lastModified",
                        "scvi_version",
                        "dataset_size",
                        "arxiv",
                        "library_name",
                        "BaseLM",
                        "template",
                        "diffusers",
                        "loss",
                        "likes",
                        'doi']

    for tree in tqdm(trees, desc="Cleaning tree nodes..."):
        for node_id, node_attr in tree.nodes(data=True):
            for node_att_to_del in node_atts_to_del:
                if node_att_to_del in node_attr:
                    del tree.nodes[node_id][node_att_to_del]

    for tree in tqdm(trees, desc="Unifying tree Nones..."):
        unknown_att = ['author', 'tags', "pipeline_tag", 'architectures', 'license', 'dataset']
        minus1_att = ['downloads', "downloadsAllTime"]

        for node_id, node_attrs in tree.nodes(data=True):
            for node_attr in node_attrs:
                node_attr_val = tree.nodes[node_id][node_attr]
                if node_attr_val is None or (isinstance(node_attr_val, float) and math.isnan(node_attr_val)):
                    if node_attr in unknown_att:
                        node_attr_val = "unknown"
                    elif node_attr in minus1_att:
                        node_attr_val = float(-1)
                    else:
                        node_attr_val = float('nan')
                tree.nodes[node_id][node_attr] = node_attr_val

    filtered_dataset.loc[:, 'subgraph_size'] = None
    filtered_dataset.loc[:, 'subgraph_merges'] = None
    filtered_dataset.loc[:, 'self_subgraph_monthly_downloads'] = None
    filtered_dataset.loc[:, 'self_subgraph_all_time_downloads'] = None
    filtered_dataset.loc[:, 'modality'] = None

    modalies = {'meta-llama/Meta-Llama-3-8B': "NLP",
                'black-forest-labs/FLUX.1-dev': "vision",
                'mistralai/Mistral-7B-v0.1': "NLP",
                'meta-llama/Llama-3.1-8B': "NLP",
                'mistral-community/Mistral-7B-v0.2': "NLP",
                'stabilityai/stable-diffusion-xl-base-1.0': "vision",
                'Qwen/Qwen2.5-7B': "NLP",
                'google/gemma-2-9b': "NLP",
                'meta-llama/Llama-3.2-3B': "NLP",
                'mistralai/Mistral-Nemo-Base-2407': "NLP",
                'openai/whisper-small': "audio",
                'google/vit-base-patch16-224-in21k': "vision",
                'CompVis/stable-diffusion-v1-1': "vision",
                'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T': "NLP",
                'microsoft/Phi-3.5-mini-instruct': "NLP",
                'openai/whisper-tiny': "audio",
                'openai/whisper-large-v3': "audio",
                'facebook/wav2vec2-base': "audio",
                'stabilityai/stable-diffusion-2-base': "vision",
                'openai/whisper-medium': "audio",
                'google/vit-base-patch16-224': "vision",
                'facebook/detr-resnet-50': "vision",
                'microsoft/swin-tiny-patch4-window7-224': "vision",
                'MCG-NJU/videomae-base': "vision",
                'openai/whisper-base': "audio",
                'VietAI/vit5-large': "vision",
                'openai/whisper-large-v2': "audio",
                'Qwen/Qwen2-VL-7B': "VLM",
                'facebook/deit-tiny-patch16-224': "vision",
                'facebook/deit-small-patch16-224': "vision",
                'Qwen/Qwen2-VL-2B': "VLM",
                'microsoft/beit-base-patch16-224': "vision",
                'facebook/deit-base-patch16-224': "vision",
                "microsoft/resnet-50": "vision",
                'openai/clip-vit-large-patch14': "vision",
                'openai/clip-vit-base-patch32': "vision",
                'timm/mobilenetv3_small_100.lamb_in1k': "vision"
                }

    for tree in tqdm(trees, desc="Calculating subtree download and size, trees..."):
        # Find the tree root (node with in_degree == 0)
        tree_root = [node for node in tree.nodes if tree.in_degree(node) == 0][0]
        memo = {}
        compute_subtree_stats_dfs(tree, tree_root, memo)

        for node_id, node_attrs in tqdm(tree.nodes(data=True), desc="Calculating subtree download and size, nodes..."):
            try:
                node_monthly = int(tree.nodes[node_id].get("downloads", "0"))
            except Exception:
                node_monthly = 0
            try:
                node_all_time = int(tree.nodes[node_id].get("downloadsAllTime", "0"))
            except Exception:
                node_all_time = 0

            descendant_stats = memo.get(node_id, {"monthly_downloads": 0, "all_time_downloads": 0, "subgraph_size": 0, "subgraph_merges": 0})

            tree.nodes[node_id]["self_subgraph_monthly_downloads"] = float(node_monthly + descendant_stats["monthly_downloads"])
            tree.nodes[node_id]["self_subgraph_all_time_downloads"] = float(node_all_time + descendant_stats["all_time_downloads"])
            tree.nodes[node_id]["subgraph_size"] = float(descendant_stats["subgraph_size"])
            tree.nodes[node_id]["subgraph_merges"] = float(descendant_stats["subgraph_merges"])
            if tree_root in modalies:
                tree.nodes[node_id]["modality"] = modalies[tree_root]

            filtered_dataset.loc[filtered_dataset['id'] == node_id, 'self_subgraph_monthly_downloads'] = float(node_monthly + descendant_stats["monthly_downloads"])
            filtered_dataset.loc[filtered_dataset['id'] == node_id, 'self_subgraph_all_time_downloads'] = float(node_all_time + descendant_stats["all_time_downloads"])
            filtered_dataset.loc[filtered_dataset['id'] == node_id, 'subgraph_size'] = float(descendant_stats["subgraph_size"])
            filtered_dataset.loc[filtered_dataset['id'] == node_id, 'subgraph_merges'] = float(descendant_stats["subgraph_merges"])
            if tree_root in modalies:
                filtered_dataset.loc[filtered_dataset['id'] == node_id, 'modality'] = modalies[tree_root]

        nx.write_graphml(tree, os.path.join(exported_graphml_dir, f"n_nodes_{len(tree.nodes):05d}__root_{tree_root.replace('/', '___')}.graphml"))

    filtered_dataset.to_csv(os.path.join(exported_graphml_dir, "filtered_dataset.csv"), index=False)


def default_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_outputs_dir", type=str, default="processed_hub_stats")
    parser.add_argument("--processed_hub_stats_fname", type=str, default="processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents")
    parser.add_argument("--output_cc_dir", type=str, default="individual_ccs")

    parser.add_argument("--root_type", type=str, default="root", choices=["root", "weak_cc"])
    return parser


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    os.makedirs(args.output_cc_dir, exist_ok=True)

    main(args)