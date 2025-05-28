# Filtering for LLaVA Models

This guide explains how to use the Model Atlas repository to specifically focus on LLaVA (Large Language and Vision Assistant) models, which combine vision and language capabilities.

## Using the LLaVA Filter Script

We've provided a special script `filter_llava_models.py` to help you focus on LLaVA models. There are two ways to use this script:

### Option 1: Extract LLaVA Models from the Entire Graph

This option searches through the entire model graph to find any models with "llava" in their name:

```bash
# First, process the Hugging Face stats (if you haven't already)
python process_hf_stats.py

# Then, run the LLaVA filter script
python filter_llava_models.py --output_dir=llava_output
```

This will:
1. Load the entire model graph
2. Find all models with "llava" in their name
3. Create a subgraph containing only these models
4. Save the subgraph as a GraphML file
5. Save the model data as a CSV file

### Option 2: Find LLaVA Model Trees from GraphML Files

If you've already generated GraphML files using `create_graphml_files.py`, you can use this option to find trees that have LLaVA models as their roots:

```bash
# First, process the Hugging Face stats and create GraphML files
python process_hf_stats.py
python create_graphml_files.py

# Then, run the LLaVA filter script with the graphml_dir parameter
python filter_llava_models.py --graphml_dir=individual_ccs/processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents___cc_breakdown___root_type_root/graphml --output_dir=llava_output
```

This will:
1. Search through all GraphML files in the specified directory
2. Find files with "llava" in their filename (which indicates a LLaVA model as the root)
3. Copy these files to the output directory

## Parameters

- `--processed_outputs_dir`: Directory containing processed data (default: "processed_hub_stats")
- `--processed_hub_stats_fname`: Base filename for processed data (default: "processed_hub_stats___manual_annot_v6___with_auto_model_card_extracted_parents")
- `--graphml_dir`: Directory containing GraphML files (if provided, uses Option 2)
- `--output_dir`: Directory to save the filtered LLaVA models (default: "llava_models")
- `--root_type`: Type of root node used in graph construction (choices: "root", "weak_cc", default: "root")

## Visualizing LLaVA Models

Once you have the filtered GraphML files, you can visualize them using Gephi:

1. Open Gephi and create a new project
2. Import the GraphML file(s) from your output directory
3. Use the ForceAtlas2 layout algorithm to arrange the graph
4. Style the nodes based on attributes like downloads, creation date, etc.

This will give you a focused view of the LLaVA model ecosystem, showing how different LLaVA models are related and evolved from each other. 