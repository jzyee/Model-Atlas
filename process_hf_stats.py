import os
import pickle
import argparse
import pandas as pd
import networkx as nx
from tqdm import tqdm
from tqdm.auto import tqdm
from datasets import load_dataset
from utils.data_processing_utils import preprocess_df_dataset
from utils.graph_utils import get_ccs_from_digraph_based_on_roots, get_ccs_from_digraph_based_on_cc, remove_self_loops_and_cycles

# Enable progress bars for pandas operations
tqdm.pandas()


def str2bool(v):
    """
    Convert string representation of boolean to actual boolean value.
    Used for command line argument parsing.
    
    Args:
        v: String value to convert
        
    Returns:
        Boolean representation of the string
        
    Raises:
        argparse.ArgumentTypeError: If the string cannot be interpreted as a boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_model_graph_from_df(df):
    """
    Build a directed graph from a pandas DataFrame containing model information.
    
    Each model becomes a node in the graph, and edges represent parent-child
    relationships between models (e.g., a model is derived from another).
    
    Args:
        df: Pandas DataFrame containing model information
        
    Returns:
        A NetworkX DiGraph representing the model relationships
    """
    # Initialize an empty directed graph
    graph = nx.DiGraph()

    # First, add all models as nodes in the graph
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
        model_id = row["id"]  # The model identifier (e.g., "bert-base-uncased")
        node_data = row.to_dict()  # All model metadata becomes node attributes
        graph.add_node(model_id, **node_data)

    # Then, add edges between models based on their parent-child relationships
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
        model_id = row["id"]  # Current model
        base_model = row.get("base_model", None)  # Parent model(s) of the current model

        # Skip if there's no parent model
        if pd.isnull(base_model) or base_model is None:
            continue

        # If base_model is a string representation of a list, convert it to an actual list
        if isinstance(base_model, str):
            try:
                base_model = eval(base_model)  # Safely evaluate string to list
            except:
                # If evaluation fails, keep it as a string
                pass

        # Handle the case where a model has multiple parent models
        if isinstance(base_model, list):
            for base in base_model:
                # Only add an edge if the parent model exists in our graph
                if base in graph.nodes:
                    # Edge direction: parent -> child
                    graph.add_edge(base, model_id)
        # Handle the case where a model has a single parent model
        elif isinstance(base_model, str):
            if base_model in graph.nodes:
                graph.add_edge(base_model, model_id)

    return graph


def default_argument_parser():
    """
    Create an argument parser with default values for processing HF stats.
    
    Returns:
        An argparse.ArgumentParser object with predefined arguments
    """
    parser = argparse.ArgumentParser()
    # Directory to save processed data
    parser.add_argument("--processed_outputs_dir", type=str, default="processed_hub_stats")
    # Base filename for processed data
    parser.add_argument("--processed_hub_stats_fname", type=str, default="processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents")
    # Whether to load already processed dataset
    parser.add_argument("--load_dataset_from_disk", type=str2bool, default=True)
    # Type of root node to use for constructing graphs
    parser.add_argument("--root_type", type=str, default="root", choices=["root", "weak_cc"])

    return parser


def fix_sd_data(df):
    """
    Manually fix data for specific Stable Diffusion models that might have
    missing or incorrect information in the dataset.
    
    Args:
        df: Pandas DataFrame containing model information
        
    Returns:
        DataFrame with corrected information for Stable Diffusion models
    """
    # Fix metadata for runwayml/stable-diffusion-v1-5
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'base_model'] = "['CompVis/stable-diffusion-v1-2']"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'base_model_clean'] = "CompVis/stable-diffusion-v1-2"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'base_model_relation'] = "finetune"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'downloads'] = 0
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'createdAt'] = "2023-02-27 00:00:00"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'author'] = "runwayml"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'tags'] = ""

    # Fix metadata for stabilityai/stable-diffusion-2
    df.loc[df['id'] == 'stabilityai/stable-diffusion-2', 'base_model'] = "['stabilityai/stable-diffusion-2-base']"
    df.loc[df['id'] == 'stabilityai/stable-diffusion-2', 'base_model_clean'] = "stabilityai/stable-diffusion-2-base"
    df.loc[df['id'] == 'stabilityai/stable-diffusion-2', 'base_model_relation'] = "finetune"
    return df


def main():
    """
    Main function to process Hugging Face model statistics and build a model graph.
    
    The process involves:
    1. Loading and preprocessing the dataset
    2. Building a graph of model relationships
    3. Extracting connected components from the graph
    4. Saving the results to disk
    """
    # Parse command line arguments
    args = default_argument_parser().parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.processed_outputs_dir, exist_ok=True)
    
    # Define path for processed dataset
    processed_dataset_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}.csv")
    
    # Either load existing processed dataset or create a new one
    if args.load_dataset_from_disk and os.path.exists(processed_dataset_path):
        # Load previously processed dataset from disk
        print(f"Loading processed dataset from {processed_dataset_path}")
        dataset = pd.read_csv(processed_dataset_path, low_memory=False)
    else:
        # Load raw dataset from Hugging Face hub-stats
        print("Loading and processing raw dataset from Hugging Face")
        dataset = load_dataset("cfahlgren1/hub-stats", "models")
        dataset = dataset['train'].to_pandas()  # Convert to pandas DataFrame
        
        # Preprocess the dataset (clean data, extract info, etc.)
        dataset = preprocess_df_dataset(args, dataset)
        
        # Fix specific data issues for Stable Diffusion models
        dataset = fix_sd_data(dataset)
        
        # Save the processed dataset to disk for future use
        dataset.to_csv(processed_dataset_path, index=False)
        print(f"Saved processed dataset to {processed_dataset_path}")

    # Define paths for saving the model graph and connected components
    model_graph_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}___entire_graph.pkl")
    cc_save_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}___cc_breakdown___root_type_{args.root_type}.pkl")

    # Build the graph of model relationships
    print("Building the graph...")
    model_graph = build_model_graph_from_df(dataset)
    
    # Save the complete graph to disk
    with open(model_graph_path, 'wb') as f:
        pickle.dump(model_graph, f)
    print(f"Initial graph has {model_graph.number_of_nodes()} nodes and {model_graph.number_of_edges()} edges...")

    # Remove self loops and cycles to ensure a proper directed acyclic graph (DAG)
    model_graph = remove_self_loops_and_cycles(model_graph)
    
    # Extract connected components from the graph based on the specified method
    if args.root_type == "root":
        # Extract components based on tree roots (nodes with no incoming edges)
        ccs = get_ccs_from_digraph_based_on_roots(model_graph)
    elif args.root_type == "weak_cc":
        # Extract components based on weakly connected components
        ccs = get_ccs_from_digraph_based_on_cc(model_graph)
        
    # Save the connected components to disk
    print(f"Found {len(ccs)} disjoint components, saving them to {cc_save_path}...")
    with open(cc_save_path, 'wb') as f:
        pickle.dump(ccs, f)


if __name__ == "__main__":
    main()
