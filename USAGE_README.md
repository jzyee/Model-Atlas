# Model Atlas: Usage Guide

This guide explains how to use the Model Atlas repository to generate graph visualizations of Hugging Face model relationships.

## Overview

The Model Atlas repository helps you analyze and visualize relationships between models in the Hugging Face Hub. The workflow involves:

1. Processing Hugging Face statistics and creating a model graph
2. Extracting connected components from the graph
3. Creating GraphML files that can be loaded into visualization software like Gephi

## Installation

### Prerequisites

- Python 3.7+
- Required packages:
  - pandas
  - networkx
  - datasets
  - tqdm
  - pyyaml
  - igraph
  - matplotlib
  - pillow

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Model-Atlas.git
   cd Model-Atlas
   ```

2. Install required packages:
   ```bash
   pip install pandas networkx datasets tqdm pyyaml python-igraph matplotlib pillow
   ```

## Usage

The workflow consists of two main steps:

### 1. Process Hugging Face Statistics

This step downloads and processes model statistics from Hugging Face:

```bash
python process_hf_stats.py --processed_outputs_dir=processed_hub_stats --root_type=root
```

**Parameters:**
- `--processed_outputs_dir`: Directory to save processed data (default: "processed_hub_stats")
- `--processed_hub_stats_fname`: Base filename for processed data (default: "processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents")
- `--load_dataset_from_disk`: Whether to load already processed dataset (default: True)
- `--root_type`: Type of root node to use for constructing graphs (choices: "root", "weak_cc", default: "root")

### 2. Create GraphML Files

This step creates GraphML files that can be loaded into visualization software:

```bash
python create_graphml_files.py --processed_outputs_dir=processed_hub_stats --output_cc_dir=individual_ccs
```

**Parameters:**
- `--processed_outputs_dir`: Directory containing processed data (default: "processed_hub_stats")
- `--processed_hub_stats_fname`: Base filename for processed data (default: "processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents")
- `--output_cc_dir`: Directory to save output connected components (default: "individual_ccs")
- `--root_type`: Type of root node used in graph construction (choices: "root", "weak_cc", default: "root")

## Output Files

The process generates several outputs:

1. **Processed Dataset CSV**: Contains cleaned and processed model information
2. **Graph Pickle Files**: Serialized NetworkX graph objects representing model relationships
3. **GraphML Files**: Graph representations that can be loaded into visualization software like Gephi

Output locations:
- Processed dataset: `processed_hub_stats/processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents.csv`
- GraphML files: `individual_ccs/processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents___cc_breakdown___root_type_root/graphml/`

## Visualization with Gephi

1. Download and install [Gephi](https://gephi.org/)
2. Launch Gephi and create a new project
3. Go to File → Open and select one of the GraphML files
4. Use Gephi's layout algorithms (e.g., ForceAtlas2) to arrange the graph
5. Customize node size, color, and labels based on attributes like downloads, creation date, etc.

## Example Workflow

```bash
# Step 1: Process Hugging Face stats
python process_hf_stats.py --processed_outputs_dir=my_output --root_type=root

# Step 2: Create GraphML files
python create_graphml_files.py --processed_outputs_dir=my_output --output_cc_dir=my_graphs --root_type=root
```

The resulting GraphML files will be in: `my_graphs/processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents___cc_breakdown___root_type_root/graphml/`

## Additional Notes

- The `root_type` parameter determines how the graph components are extracted:
  - `root`: Creates trees based on root nodes (nodes with no incoming edges)
  - `weak_cc`: Creates components based on weakly connected components
  
- The script calculates various statistics for each node, including:
  - Number of descendants (subgraph size)
  - Cumulative downloads for the node and its descendants
  - Number of merge operations in the subtree

- The largest graphs typically represent the most influential model families in the Hugging Face ecosystem 