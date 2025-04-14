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

tqdm.pandas()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def build_model_graph_from_df(df):
    graph = nx.DiGraph()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
        model_id = row["id"]
        node_data = row.to_dict()
        graph.add_node(model_id, **node_data)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
        model_id = row["id"]
        base_model = row.get("base_model", None)

        if pd.isnull(base_model) or base_model is None:
            continue

        if isinstance(base_model, str):
            base_model = eval(base_model)

        if isinstance(base_model, list):
            for base in base_model:
                if base in graph.nodes:
                    graph.add_edge(base, model_id)
        elif isinstance(base_model, str):
            if base_model in graph.nodes:
                graph.add_edge(base_model, model_id)

    return graph


def default_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_outputs_dir", type=str, default="processed_hub_stats")
    parser.add_argument("--processed_hub_stats_fname", type=str, default="processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents")
    parser.add_argument("--load_dataset_from_disk", type=str2bool, default=True)

    parser.add_argument("--root_type", type=str, default="root", choices=["root", "weak_cc"])

    return parser


def fix_sd_data(df):
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'base_model'] = "['CompVis/stable-diffusion-v1-2']"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'base_model_clean'] = "CompVis/stable-diffusion-v1-2"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'base_model_relation'] = "finetune"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'downloads'] = 0
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'createdAt'] = "2023-02-27 00:00:00"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'author'] = "runwayml"
    df.loc[df['id'] == 'runwayml/stable-diffusion-v1-5', 'tags'] = ""

    df.loc[df['id'] == 'stabilityai/stable-diffusion-2', 'base_model'] = "['stabilityai/stable-diffusion-2-base']"
    df.loc[df['id'] == 'stabilityai/stable-diffusion-2', 'base_model_clean'] = "stabilityai/stable-diffusion-2-base"
    df.loc[df['id'] == 'stabilityai/stable-diffusion-2', 'base_model_relation'] = "finetune"
    return df


def main():
    args = default_argument_parser().parse_args()
    os.makedirs(args.processed_outputs_dir, exist_ok=True)
    processed_dataset_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}.csv")
    if args.load_dataset_from_disk and os.path.exists(processed_dataset_path):
        dataset = pd.read_csv(processed_dataset_path, low_memory=False)
    else:
        dataset = load_dataset("cfahlgren1/hub-stats", "models")
        dataset = dataset['train'].to_pandas()
        dataset = preprocess_df_dataset(args, dataset)
        dataset = fix_sd_data(dataset)
        dataset.to_csv(processed_dataset_path, index=False)

    model_graph_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}___entire_graph.pkl")
    cc_save_path = os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}___cc_breakdown___root_type_{args.root_type}.pkl")

    print("Building the graph...")
    model_graph = build_model_graph_from_df(dataset)
    with open(model_graph_path, 'wb') as f:
        pickle.dump(model_graph, f)
    print(f"Initial graph has {model_graph.number_of_nodes()} nodes and {model_graph.number_of_edges()} edges...")

    model_graph = remove_self_loops_and_cycles(model_graph)
    if args.root_type == "root":
        ccs = get_ccs_from_digraph_based_on_roots(model_graph)
    elif args.root_type == "weak_cc":
        ccs = get_ccs_from_digraph_based_on_cc(model_graph)
    print(f"Found {len(ccs)} disjoint components, saving them to {cc_save_path}...")
    with open(cc_save_path, 'wb') as f:
        pickle.dump(ccs, f)


if __name__ == "__main__":
    main()
