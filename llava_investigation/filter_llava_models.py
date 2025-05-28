import os
import glob
import pickle
import argparse
import pandas as pd
import networkx as nx
from tqdm import tqdm
from copy import deepcopy

def extract_llava_models(graph, output_dir):
    """
    Extract LLaVA models and their relationships from a graph
    
    Args:
        graph: A NetworkX DiGraph containing all models
        output_dir: Directory to save the extracted LLaVA graph
        
    Returns:
        A subgraph containing only LLaVA models and their direct relationships
    """
    # Find all nodes with "llava" in their ID (case-insensitive)
    llava_nodes = [node for node in graph.nodes() if "llava" in node.lower()]
    
    if not llava_nodes:
        print("No LLaVA models found in the graph.")
        return None
    
    print(f"Found {len(llava_nodes)} LLaVA models in the graph.")
    
    # Create a subgraph with the LLaVA models
    llava_graph = graph.subgraph(llava_nodes).copy()
    
    # Save the subgraph as GraphML
    nx.write_graphml(llava_graph, os.path.join(output_dir, "llava_models.graphml"))
    print(f"Saved LLaVA model graph to {os.path.join(output_dir, 'llava_models.graphml')}")
    
    # Save the node data as CSV for reference
    nodes_data = []
    for node in llava_nodes:
        node_data = graph.nodes[node]
        node_data['id'] = node
        nodes_data.append(node_data)
    
    df = pd.DataFrame(nodes_data)
    df.to_csv(os.path.join(output_dir, "llava_models.csv"), index=False)
    print(f"Saved LLaVA model data to {os.path.join(output_dir, 'llava_models.csv')}")
    
    return llava_graph

def find_llava_trees(trees_dir, output_dir):
    """
    Find GraphML files in the trees directory that contain LLaVA models as roots
    
    Args:
        trees_dir: Directory containing GraphML files
        output_dir: Directory to save the filtered files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all GraphML files
    graphml_files = glob.glob(os.path.join(trees_dir, "*.graphml"))
    
    # Filter for files with "llava" in the root name
    llava_files = [f for f in graphml_files if "llava" in f.lower()]
    
    if not llava_files:
        print("No GraphML files with LLaVA roots found.")
        return
    
    print(f"Found {len(llava_files)} GraphML files with LLaVA roots.")
    
    # Copy the files to the output directory
    for file in llava_files:
        filename = os.path.basename(file)
        graph = nx.read_graphml(file)
        nx.write_graphml(graph, os.path.join(output_dir, filename))
        print(f"Saved {filename} to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Filter and extract LLaVA models from the graph")
    parser.add_argument("--processed_outputs_dir", type=str, default="processed_hub_stats",
                       help="Directory containing processed data")
    parser.add_argument("--processed_hub_stats_fname", type=str, 
                       default="processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents",
                       help="Base filename for processed data")
    parser.add_argument("--graphml_dir", type=str, 
                       help="Directory containing GraphML files (if not provided, will use the entire graph)")
    parser.add_argument("--output_dir", type=str, default="llava_models",
                       help="Directory to save the filtered LLaVA models")
    parser.add_argument("--root_type", type=str, default="root", choices=["root", "weak_cc"],
                       help="Type of root node used in graph construction")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If a GraphML directory is provided, find trees with LLaVA roots
    if args.graphml_dir and os.path.exists(args.graphml_dir):
        find_llava_trees(args.graphml_dir, args.output_dir)
    else:
        # Otherwise, load the entire graph and extract LLaVA models
        model_graph_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}___entire_graph.pkl")
        
        if not os.path.exists(model_graph_path):
            print(f"Graph file not found: {model_graph_path}")
            print("Please run process_hf_stats.py first to create the graph.")
            return
        
        print(f"Loading graph from {model_graph_path}...")
        with open(model_graph_path, 'rb') as f:
            model_graph = pickle.load(f)
        
        extract_llava_models(model_graph, args.output_dir)

if __name__ == "__main__":
    main() 