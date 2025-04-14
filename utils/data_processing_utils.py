import os
import ast
import math
import json
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from tqdm.auto import tqdm

tqdm.pandas()


def extract_architectures(config_str):
    if pd.isnull(config_str) or config_str is None:
        return None

    try:
        config_data = json.loads(config_str)
        return config_data.get('architectures', None)
    except json.JSONDecodeError:
        return None


def extract_param_count(config_data): # TODO: This does not work as intended, should not look at the total
    if pd.isnull(config_data) or config_data is None:
        return None
    try:
        # config_data = json.loads(config_str)
        return config_data.get('total', None)
    except json.JSONDecodeError:
        return None

def extract_finetuning_data(base_model_list):
    if not base_model_list or not isinstance(base_model_list, list):
        return None, None

    finetuning_types = ["adapter", "merge", "quantized", "finetune", "root", "unknown", "virtual_recovered"]

    base_model_clean = set()
    base_model_relation = set()

    for item in base_model_list:
        if isinstance(item, str) and ':' in item:
            possible_type, value = item.split(':', 1)
            if possible_type in finetuning_types:
                base_model_relation.add(possible_type)
                base_model_clean.add(value)
            else:
                base_model_clean.add(item)
        else:
            base_model_clean.add(item)
    return list(base_model_clean), list(base_model_relation)


def split_tags(row_tags):
    recognized_keys = [
        'doi', 'diffusers', 'template', 'arxiv', 'anndata_version',
        'annotated', 'model_cls_name', 'loss', 'BaseLM', 'dataset', 'modality', 'dataset_size', 'base_model',
        'scvi_version', 'adapterhub', 'license', 'tissue', 'region'
    ]

    normal_tags = []
    kv_dict = {}

    for tag in row_tags:
        if ':' in tag:
            possible_key, val = tag.split(':', 1)
            if possible_key in recognized_keys:
                if possible_key not in kv_dict:
                    kv_dict[possible_key] = []
                kv_dict[possible_key].append(val)
            else:
                normal_tags.append(tag)
        else:
            normal_tags.append(tag)

    return normal_tags, kv_dict


def clean_metadata(df):
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
                        "scvi_version",
                        "dataset_size",
                        "arxiv",
                        "BaseLM",
                        "template",
                        "diffusers",
                        "loss",
                        "likes",
                        'doi',
                        'cardData',
                        'inferenceProviderMapping',
                        'safetensors',
                        'transformersInfo',
                        'gguf'
                        ]
    df = df.drop(columns=node_atts_to_del)

    def clean_base_model_relation(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "unknown"
        elif x == "[]":
            return "unknown"
        elif isinstance(x, str):
            x = '_'.join(ast.literal_eval(x))
        return x

    def clean_pipeline_tag(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "unknown"
        elif x == "text-retrieval" or x == "table-to-text":
            return "filtered_out"
        elif x == "other":
            return "unknown"
        return x

    df["base_model_relation"] = df["base_model_relation"].progress_apply(clean_base_model_relation)
    df["pipeline_tag"] = df["pipeline_tag"].progress_apply(clean_pipeline_tag)
    df["license"] = df["license"].progress_apply(lambda x: '_'.join(ast.literal_eval(x)) if isinstance(x, str) else "unknown")
    df["dataset"] = df["dataset"].progress_apply(lambda x: '_'.join(ast.literal_eval(x)) if isinstance(x, str) else "unknown")
    df["architectures"] = df["architectures"].progress_apply(lambda x: '_'.join(ast.literal_eval(x)) if isinstance(x, str) else "unknown")
    # df["base_model"] = df["base_model"].apply(lambda x: '_'.join(ast.literal_eval(x)) if isinstance(x, str) else "unknown")
    df["base_model_clean"] = df["base_model"].progress_apply(lambda x: '_'.join(ast.literal_eval(x)) if isinstance(x, str) else "unknown")

    bla = 1
    return df


def preprocess_df_dataset(args, dataset):
    cols_to_drop = ['_id']
    dataset = dataset.drop(columns=cols_to_drop)

    dataset[['tags_clean', 'tag_dict']] = dataset['tags'].progress_apply(lambda x: pd.Series(split_tags(x)))
    dataset['tags'] = dataset['tags_clean']
    dataset = dataset.drop(columns=['tags_clean'])  # Drop the temporary column
    all_keys = set()
    for dct in tqdm(dataset['tag_dict']):
        for k in dct.keys():
            all_keys.add(k)

    for key in tqdm(all_keys):
        dataset[key] = dataset['tag_dict'].progress_apply(lambda d: d.get(key, None))

    dataset[['base_model', 'base_model_relation']] = dataset['base_model'].progress_apply(lambda x: pd.Series(extract_finetuning_data(x)))

    dataset = dataset.drop(columns=['tag_dict'])
    dataset['architectures'] = dataset['config'].progress_apply(extract_architectures)
    # dataset['param_count'] = dataset['safetensors'].progress_apply(extract_param_count)

    dataset.to_csv(os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}_BEFORE_MERGE.csv"), index=False)
    dataset = pd.read_csv(os.path.join(args.processed_outputs_dir, f"{args.processed_hub_stats_fname}_BEFORE_MERGE.csv"), low_memory=False)

    dataset = manually_clean_cycles(deepcopy(dataset))
    dataset = load_and_merge_manually_merged_df(deepcopy(dataset))
    dataset = load_and_merge_extracted_parents_df(deepcopy(dataset))
    dataset = filter_by_sibling_extensions(deepcopy(dataset))

    # addded_stats_df = merge_eval_stats(deepcopy(filtered_dataset)) # Note: this is to add the eval metrics
    dataset = create_virtual_nodes(deepcopy(dataset))
    dataset = clean_metadata(deepcopy(dataset))
    return dataset


def manually_clean_cycles(df):
    df.loc[df['id'] == "Wonder-Griffin/JudgeLLM2", ['base_model', 'base_model_relation']] = [None, None]
    df.loc[df['id'] == "BexRedpill/distilbert-on-polarity-yelp-reviews", ['base_model', 'base_model_relation']] = [None, None]
    df.loc[df['id'] == "mHossain/bengali_pos_v1_300000", ['base_model', 'base_model_relation']] = [None, None]
    df.loc[df['id'] == "quastrinos/race-openbook-finetuned-deberta-v3-large-mcqa-TPU", ['base_model', 'base_model_relation']] = [None, None]
    df.loc[df['id'] == "weiren119/distilhubert-finetuned-gtzan", ['base_model', 'base_model_relation']] = [None, None]
    return df


def contains_target_extension(siblings_str):
    target_extensions = {
        ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".pb", ".h5",
        ".npy", ".npz", ".onnx", ".tflite", ".xmodel", ".dlc", ".uff",
        ".pbtxt", ".msgpack", ".pkl", ".mar", ".gguf", ".ggml",
        ".pdparams", ".pdopt", ".imatrix", ".jit", ".t7", ".mdl"
    }
    try:
        if not siblings_str or siblings_str.strip() == "[]":
            return False
        return any(ext in siblings_str for ext in target_extensions)

    except Exception as e:
        print(f"Error processing: {e}")
        return False


def filter_by_sibling_extensions(df):
    filtered_df = df[df['siblings'].progress_apply(contains_target_extension)]
    return filtered_df


def load_and_merge_manually_merged_df(original_df):
    # Load df_manual and rename its ID column
    df_manual = pd.read_csv(os.path.join("processed_hub_stats", "manually_annotated_v6.csv"))
    df_manual.rename(columns={"current_model": "id"}, inplace=True)

    # Format the base_model column in df_manual
    df_manual["base_model"] = df_manual["base_model"].apply(lambda x: f"[{', '.join(repr(item.strip()) for item in x.split(','))}]" if isinstance(x, str) else None)
    df_manual["base_model_relation"] = df_manual["base_model_relation"].apply(lambda x: f"[{', '.join(repr(item.strip()) for item in x.split(','))}]" if isinstance(x, str) else None)

    # Merge so that we keep all rows from original_df and pick up 'base_model' from df_manual where IDs match
    # Use suffixes to avoid getting base_model_x and base_model_y
    df_merged = original_df.merge(df_manual[["id", "base_model", "base_model_relation"]], on="id", how="left", suffixes=("_original", "_manual"))

    # If base_model_manual is not null, use that value; otherwise use base_model_original
    df_merged["base_model"] = df_merged["base_model_manual"].combine_first(df_merged["base_model_original"])
    df_merged["base_model_relation"] = df_merged["base_model_relation_manual"].combine_first(df_merged["base_model_relation_original"])

    # Clean up unnecessary columns
    df_merged.drop(columns=["base_model_original", "base_model_manual"], inplace=True)
    df_merged.drop(columns=["base_model_relation_original", "base_model_relation_manual"], inplace=True)

    # Example assertion: check that a particular id has been updated correctly
    # Replace this ID value with whichever you want to test
    test_id = "bunnycore/Qwen-2.1-7b-Persona-lora_model"
    merged_val = df_merged.loc[df_merged["id"] == test_id, "base_model"].values[0]
    manual_val = df_manual.loc[df_manual["id"] == test_id, "base_model"].values[0]

    assert merged_val == manual_val, f"base_model for ID='{test_id}' not updated correctly!"

    return df_merged


def create_virtual_nodes(df):
    # Find all the values in the "base_model" column that are lists
    # (i.e. they have multiple base models)
    multi_base_models = df["base_model"].apply(lambda x: eval(x) if not pd.isnull(x) else None)
    # Find all the items in those lists that are not in the "id" column
    # (i.e. they are not actual model IDs)
    virtual_nodes = dict()
    existing_nodes = set(df["id"].values)
    for row in multi_base_models:
        if row is None:
            continue
        else:
            for item in row:
                if item not in existing_nodes:
                    if item not in virtual_nodes.keys():
                        virtual_nodes[item] = []
                    virtual_nodes[item].append(item)
    # sort the virtual nodes by the number of times they appear
    virtual_nodes = dict(sorted(virtual_nodes.items(), key=lambda item: len(item[1]), reverse=True))

    # Add the virtual nodes to the dataframe
    virtual_nodes_df = pd.DataFrame(virtual_nodes.keys(), columns=["id"])
    virtual_nodes_df["base_model_relation"] = "[\'virtual_recovered\']"

    # Merge the virtual nodes with the original dataframe
    merged_df = pd.concat([df, virtual_nodes_df], ignore_index=True)

    return merged_df


def load_and_merge_extracted_parents_df(original_df):
    # Load df_manual and rename its ID column
    df_manual = pd.read_csv(os.path.join("processed_hub_stats", "09_01_25_model_cards_extracted_parent_model_clean.csv"))
    df_manual["base_model"] = df_manual["base_model"].apply(lambda x: x.strip("/") if isinstance(x, str) else None)

    # remove any row from df_manual for which there is a row in original_df that has the same id and the value of the base_model column is not None or nan
    df_manual_filtered = df_manual[~df_manual['id'].isin(original_df[~original_df['base_model'].isnull()]['id'])]

    df_manual_filtered["base_model_relation"] = "[\'unknown_readme_extracted\']"
    df_manual_filtered["base_model"] = df_manual_filtered["base_model"].apply(lambda x: f"[{', '.join(repr(item.strip()) for item in x.split(','))}]" if isinstance(x, str) else None)

    df_merged = original_df.merge(df_manual_filtered[["id", "base_model", "base_model_relation"]], on="id", how="left", suffixes=("_original", "_manual"))

    # If base_model_manual is not null, use that value; otherwise use base_model_original
    df_merged["base_model"] = df_merged["base_model_manual"].combine_first(df_merged["base_model_original"])
    df_merged["base_model_relation"] = df_merged["base_model_relation_manual"].combine_first(df_merged["base_model_relation_original"])

    # Clean up unnecessary columns
    df_merged.drop(columns=["base_model_original", "base_model_manual"], inplace=True)
    df_merged.drop(columns=["base_model_relation_original", "base_model_relation_manual"], inplace=True)

    return df_merged


# region: Merge eval stats

def merge_eval_stats(df):
    raw_evals_csv_path = os.path.join("processed_dataset_AFTER_DEADLINE", "09_01_25_model_cards_extracted_eval_metrics.csv")
    print(f"Loading the raw evals csv from {raw_evals_csv_path}...")
    raw_evals_df = pd.read_csv(raw_evals_csv_path, low_memory=False)
    cleaned_evals_df = eval_stats_clean_empty_rows(raw_evals_df)
    expanded_evals_df = eval_stats_expand_model_index(cleaned_evals_df)
    to_keep = ['AI2 Reasoning Challenge (25-Shot) (acc_norm)',
               'HellaSwag (10-Shot) (acc_norm)', 'MMLU (5-Shot) (acc)',
               'TruthfulQA (0-shot) (mc2)', 'Winogrande (5-shot) (acc)',
               'GSM8k (5-shot) (acc)']
    expanded_evals_df = expanded_evals_df[['id'] + to_keep]
    merged_df = df.merge(expanded_evals_df, on="id", how="left")
    return merged_df


def eval_stats_clean_empty_rows(df):
    def has_nonempty_results(model_index):
        bla = 1
        if isinstance(model_index, list):
            if len(model_index) > 1:
                bla = 1
            if "results" in model_index[0]:
                len_results = len(model_index[0]["results"])
                return len_results > 0
            else:
                return False
        else:
            bool_res = False
            bla = 1
        return bool_res

    df["model_index_json"] = df["model_index"].progress_apply(json.loads)
    return df[df["model_index_json"].progress_apply(has_nonempty_results)]


def eval_stats_expand_model_index(df, column='model_index_json'):
    # Ensure the column is parsed as a list of dictionaries
    df[column] = df[column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    rows = []

    for idx, entry in tqdm(df.iterrows(), total=len(df)):
        model_index = entry[column]
        if len(model_index) > 1:
            print(f"Model index with {len(model_index)} entries: {model_index}")
        row = {}
        for model in model_index:
            # model_name = model.get('name', '')
            # row = {}
            row = {'id': entry["id"]}
            for result in model.get('results', []):
                try:
                    # task_type = result.get('task', {}).get('type', '')
                    dataset_name = result.get('dataset', {}).get('name', '')
                    for metric in result.get('metrics', []):
                        metric_type = metric.get('type', '')
                        metric_value = metric.get('value', '')
                        column_name = f'{dataset_name} ({metric_type})'
                        row[column_name] = metric_value
                except Exception as e:
                    print(f"Error processing entry: {model_index}")
        rows.append(row)

    unique_keys = set().union(*rows)
    num_unique_keys = len(unique_keys)
    # expanded_df = pd.DataFrame(rows).fillna('')
    metric_counts = {}
    for row in rows:
        for key in row.keys():
            if key == "id":
                continue
            if key not in metric_counts:
                metric_counts[key] = 0
            metric_counts[key] = metric_counts.get(key, 0) + 1
    # filter from metric_counts those that have less than 5 occurrences
    metric_counts = {k: v for k, v in metric_counts.items() if v >= 100}
    metric_counts = dict(sorted(metric_counts.items(), key=lambda item: item[1], reverse=True)[:30])

    # sort metric counts by value

    filtered_rows = []
    for row in rows:
        for key in row.keys():
            if key == "id":
                continue
            if key in metric_counts:
                filtered_rows.append({"id": row["id"], key: row[key]})

    unique_keys = set().union(*filtered_rows)
    num_unique_keys = len(unique_keys)
    expanded_df = pd.DataFrame(filtered_rows).fillna(-1.0)
    return expanded_df

# endregion: Merge eval stats
