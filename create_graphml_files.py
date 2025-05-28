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

# Enable progress bars for pandas operations
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
    
    Args:
        G: NetworkX DiGraph representing the model relationships
        node: The node (model) for which to compute descendant statistics
        memo: Dictionary for memoization to prevent redundant computation
        
    Returns:
        Dictionary with aggregated statistics for the node's descendants
    """
    # Return memoized result if already computed for this node
    if node in memo:
        return memo[node]

    # Initialize counters for aggregated statistics
    total_monthly = 0
    total_all_time = 0
    total_merges = 0
    total_size = 0

    # Iterate through all children (successors) of the current node
    for child in G.successors(node):
        # Get the child's own monthly download count, defaulting to 0 if missing or invalid
        try:
            child_monthly = int(G.nodes[child].get("downloads", "0"))
        except Exception:
            child_monthly = 0
            
        # Get the child's own all-time download count, defaulting to 0 if missing or invalid
        try:
            child_all_time = int(G.nodes[child].get("downloadsAllTime", "0"))
        except Exception:
            child_all_time = 0
            
        # Check if the child is a merge of multiple models
        child_merge = 1 if G.nodes[child].get("base_model_relation", "unknown") == "merge" else 0

        # Add the child's own statistics to the totals
        total_monthly += child_monthly
        total_all_time += child_all_time
        total_merges += child_merge
        total_size += 1  # count this child

        # Recursively compute and add the child's descendant stats
        child_stats = compute_subtree_stats_dfs(G, child, memo)
        total_monthly += child_stats["monthly_downloads"]
        total_all_time += child_stats["all_time_downloads"]
        total_merges += child_stats["subgraph_merges"]
        total_size += child_stats["subgraph_size"]

    # Store the result in the memoization dictionary
    memo[node] = {
        "monthly_downloads": total_monthly,
        "all_time_downloads": total_all_time,
        "subgraph_size": total_size,
        "subgraph_merges": total_merges
    }
    return memo[node]

def main(args):
    """
    Main function to create GraphML files from processed model graphs.
    
    This function:
    1. Loads the processed dataset and connected components
    2. Computes statistics for each node and its descendants
    3. Exports the graph data in both NetworkX pickle and GraphML formats
    
    Args:
        args: Command-line arguments
    """
    # Define input paths for the processed dataset and connected components
    processed_dataset_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}.csv")
    dataset = pd.read_csv(processed_dataset_path, low_memory=False)
    trees_save_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}___cc_breakdown___root_type_{args.root_type}.pkl")

    # Define output directories for GraphML and NetworkX pickle files
    exported_graphml_dir = os.path.join(args.output_cc_dir, f"{args.processed_hub_stats_fname}___cc_breakdown___root_type_{args.root_type}", "graphml")
    exported_networkx_dir = os.path.join(args.output_cc_dir, f"{args.processed_hub_stats_fname}___cc_breakdown___root_type_{args.root_type}", "networkx_pkl")
    os.makedirs(exported_graphml_dir, exist_ok=True)
    os.makedirs(exported_networkx_dir, exist_ok=True)

    # Load the connected components (trees) from the pickle file
    with open(trees_save_path, 'rb') as f:
        trees = pickle.load(f)    # Filter the dataset for nodes that appear in the trees
    node_ids = [node_id for tree in trees for node_id in tree.nodes()]
    filtered_dataset = dataset[dataset['id'].isin(node_ids)]
    
    # Also identify models that have no base model but aren't roots (possibly an error)
    # But don't overwrite our main filtered dataset
    error_models = dataset[dataset['base_model'].isna() & (dataset['base_model_relation'] != "root")]
    if not error_models.empty:
        print(f"Found {len(error_models)} models with no base model but not marked as root")
    
    filtered_dataset = filtered_dataset.sort_values(by='downloads', ascending=False)

    # Find the root node(s) for each tree (nodes with no incoming edges)
    graphs_and_roots = {tree_to_insp: [node for node in tree_to_insp.nodes if tree_to_insp.in_degree(node) == 0] for tree_to_insp in trees}
    # Keep only graphs with at least 2 nodes
    graphs_and_roots = {k: v for k, v in graphs_and_roots.items() if len(k) > 1}
    
    # Create a new dictionary with sorted roots (by creation date)
    graphs_and_roots2 = {}
    for k, v in tqdm(sorted(graphs_and_roots.items(), key=lambda item: len(item[0]), reverse=True), desc="Sorting roots..."):
        # Fix missing creation dates
        for node in v:
            val = k.nodes[node].get("createdAt", None)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                # If creation date is missing, set it to current date
                k.nodes[node]["createdAt"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Sort roots by creation date (oldest first)
        all_roots = sorted(v, key=lambda item: datetime.strptime(k.nodes[item].get("createdAt", datetime.now().strftime('%Y-%m-%d %H:%M:%S')), '%Y-%m-%d %H:%M:%S').timestamp())
        graphs_and_roots2[k] = all_roots

    # Replace the original dictionary with the sorted version
    graphs_and_roots = graphs_and_roots2
    # Create a reverse mapping from root nodes to their graphs
    roots_and_graphs = {v[0]: k for k, v in sorted(graphs_and_roots.items(), key=lambda item: len(item[0]), reverse=True)}

    # Filter out trees with only one node
    trees = [tree for tree in trees if len(tree.nodes) > 1]
    # Sort trees by size (largest first)
    trees = sorted(trees, key=lambda x: len(x), reverse=True)    # Save each tree as a NetworkX pickle file
    for tree in tqdm(trees, desc="Saving networkx trees..."):
        try:
            # Find root nodes of the tree (nodes with no incoming edges)
            root_nodes = [node for node in tree.nodes if tree.in_degree(node) == 0]
            
            # Skip trees with no root nodes - these might have had edges removed due to errors
            if not root_nodes:
                print(f"Warning: Tree with {len(tree.nodes)} nodes has no root nodes. Skipping.")
                continue
                
            # Use the first root node for the filename
            first_root = root_nodes[0]
            
            # Ensure the networkx_pkl directory exists
            os.makedirs(exported_networkx_dir, exist_ok=True)
            
            # Create a safe filename without characters that could cause path issues
            safe_root_name = first_root.replace('/', '___')
            safe_root_name = re.sub(r'[^\w\s.-]', '_', safe_root_name)  # Replace any non-alphanumeric/underscore with _
            
            # Keep the filename under a reasonable length to avoid path length issues
            if len(safe_root_name) > 100:
                safe_root_name = safe_root_name[:100]
                
            filename = f"n_nodes_{len(tree.nodes):05d}___root_{safe_root_name}.pkl"
            filepath = os.path.join(exported_networkx_dir, filename)
            
            # Save the pickle file
            with open(filepath, 'wb') as f:
                pickle.dump({"cc": tree}, f)
        except Exception as e:
            print(f"Error saving tree: {e}")
            # Continue processing other trees
            continue

    # List of node attributes to remove for cleaner output
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

    # Clean tree nodes by removing unnecessary attributes
    for tree in tqdm(trees, desc="Cleaning tree nodes..."):
        for node_id, node_attr in tree.nodes(data=True):
            for node_att_to_del in node_atts_to_del:
                if node_att_to_del in node_attr:
                    del tree.nodes[node_id][node_att_to_del]

    # Standardize null/missing values in node attributes
    for tree in tqdm(trees, desc="Unifying tree Nones..."):
        # Attributes that should be set to "unknown" if missing
        unknown_att = ['author', 'tags', "pipeline_tag", 'architectures', 'license', 'dataset']
        # Attributes that should be set to -1 if missing
        minus1_att = ['downloads', "downloadsAllTime"]

        for node_id, node_attrs in tree.nodes(data=True):
            for node_attr in node_attrs:
                node_attr_val = tree.nodes[node_id][node_attr]
                # Check if value is None or NaN
                if node_attr_val is None or (isinstance(node_attr_val, float) and math.isnan(node_attr_val)):
                    if node_attr in unknown_att:
                        # Set to "unknown" for string attributes
                        node_attr_val = "unknown"
                    elif node_attr in minus1_att:
                        # Set to -1 for download counts
                        node_attr_val = float(-1)
                    else:
                        # Set to NaN for other attributes
                        node_attr_val = float('nan')
                tree.nodes[node_id][node_attr] = node_attr_val

    # Add columns for subgraph statistics to the dataset
    filtered_dataset.loc[:, 'subgraph_size'] = None
    filtered_dataset.loc[:, 'subgraph_merges'] = None
    filtered_dataset.loc[:, 'self_subgraph_monthly_downloads'] = None
    filtered_dataset.loc[:, 'self_subgraph_all_time_downloads'] = None
    filtered_dataset.loc[:, 'modality'] = None

    # Dictionary mapping root models to their modality (manually defined)
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
                }    # Calculate download and size statistics for each tree and its nodes
    for tree in tqdm(trees, desc="Calculating subtree download and size, trees..."):
        # Find all root nodes (nodes with in_degree == 0)
        root_nodes = [node for node in tree.nodes if tree.in_degree(node) == 0]
        
        # Skip trees with no root nodes
        if not root_nodes:
            print(f"Warning: Tree with {len(tree.nodes)} nodes has no root nodes during statistics calculation. Skipping.")
            continue
            
        # Use the first root node for calculations
        tree_root = root_nodes[0]
        
        # Initialize memoization dict for efficient computation
        memo = {}
        
        # Compute descendant statistics for the tree root
        try:
            compute_subtree_stats_dfs(tree, tree_root, memo)
        except Exception as e:
            print(f"Error computing statistics for tree with root {tree_root}: {e}")
            continue

        # Update node attributes and dataset with the computed statistics
        for node_id, node_attrs in tqdm(tree.nodes(data=True), desc="Calculating subtree download and size, nodes..."):
            # Get the node's own download counts
            try:
                node_monthly = int(tree.nodes[node_id].get("downloads", "0"))
            except Exception:
                node_monthly = 0
            try:
                node_all_time = int(tree.nodes[node_id].get("downloadsAllTime", "0"))
            except Exception:
                node_all_time = 0

            # Get the statistics for the node's descendants
            descendant_stats = memo.get(node_id, {"monthly_downloads": 0, "all_time_downloads": 0, "subgraph_size": 0, "subgraph_merges": 0})

            # Update node attributes with the combined statistics (node + descendants)
            tree.nodes[node_id]["self_subgraph_monthly_downloads"] = float(node_monthly + descendant_stats["monthly_downloads"])
            tree.nodes[node_id]["self_subgraph_all_time_downloads"] = float(node_all_time + descendant_stats["all_time_downloads"])
            tree.nodes[node_id]["subgraph_size"] = float(descendant_stats["subgraph_size"])
            tree.nodes[node_id]["subgraph_merges"] = float(descendant_stats["subgraph_merges"])
            # Assign modality based on the tree root if it's in the modalities dictionary
            if tree_root in modalies:
                tree.nodes[node_id]["modality"] = modalies[tree_root]            # Also update these values in the filtered dataset, but only if the node exists in the dataset
            if node_id in filtered_dataset['id'].values:
                try:
                    filtered_dataset.loc[filtered_dataset['id'] == node_id, 'self_subgraph_monthly_downloads'] = float(node_monthly + descendant_stats["monthly_downloads"])
                    filtered_dataset.loc[filtered_dataset['id'] == node_id, 'self_subgraph_all_time_downloads'] = float(node_all_time + descendant_stats["all_time_downloads"])
                    filtered_dataset.loc[filtered_dataset['id'] == node_id, 'subgraph_size'] = float(descendant_stats["subgraph_size"])
                    filtered_dataset.loc[filtered_dataset['id'] == node_id, 'subgraph_merges'] = float(descendant_stats["subgraph_merges"])
                    if tree_root in modalies:
                        filtered_dataset.loc[filtered_dataset['id'] == node_id, 'modality'] = modalies[tree_root]
                except Exception as e:
                    print(f"Error updating dataset for node {node_id}: {e}")
                    continue# Write the tree to a GraphML file
        try:
            # Create a safe filename
            safe_root_name = tree_root.replace('/', '___')
            safe_root_name = re.sub(r'[^\w\s.-]', '_', safe_root_name)  # Replace any non-alphanumeric/underscore with _
            
            # Keep the filename under a reasonable length
            if len(safe_root_name) > 100:
                safe_root_name = safe_root_name[:100]
                
            graphml_path = os.path.join(exported_graphml_dir, f"n_nodes_{len(tree.nodes):05d}__root_{safe_root_name}.graphml")
            nx.write_graphml(tree, graphml_path)
        except Exception as e:
            print(f"Error writing GraphML for tree with root {tree_root}: {e}")

    # Save the filtered dataset to CSV
    filtered_dataset.to_csv(os.path.join(exported_graphml_dir, "filtered_dataset.csv"), index=False)


def default_argument_parser():
    """
    Create an argument parser with default values for creating GraphML files.
    
    Returns:
        An argparse.ArgumentParser object with predefined arguments
    """
    parser = argparse.ArgumentParser()
    # Directory containing processed data
    parser.add_argument("--processed_outputs_dir", type=str, default="processed_hub_stats")
    # Base filename for processed data
    parser.add_argument("--processed_hub_stats_fname", type=str, default="processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents")
    # Directory to save output connected components
    parser.add_argument("--output_cc_dir", type=str, default="individual_ccs")
    # Type of root node used in graph construction
    parser.add_argument("--root_type", type=str, default="root", choices=["root", "weak_cc"])
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    args = default_argument_parser().parse_args()
    # Create output directory if it doesn't exist
    os.makedirs(args.output_cc_dir, exist_ok=True)
    # Run the main function
    main(args)