###############
#DATASET YAML #
###############

# List of triplet paths
triplet_ids_paths=(
    "/nfs/primary/SPLADE/splade_sans/data/sans_strat1_7neg.tsv"
    "/nfs/primary/SPLADE/splade_sans/data/sans_strat2_7neg.tsv"
    "/nfs/primary/SPLADE/splade_sans/data/sans_strat3_7neg.tsv"
)

# Output directory for YAML files
output_dir="./yaml_files"
mkdir -p "$output_dir"

# # Template file
# template="./yaml_files/dataset_template.yaml"

# # Generate YAML files
# strategy_number=1  # Initialize strategy counter
# for triplet_path in "${triplet_ids_paths[@]}"; do
#     for group_size in {2..8}; do
#         # Construct the filename as strategy[1,2,3]_group[2..8]
#         yaml_name="strategy${strategy_number}_group${group_size}.yaml"
#         # Debug output
#         echo "Generating for strategy: $strategy_number, group_size: $group_size, triplet_path: $triplet_path, yaml_name: $yaml_name"

#         # Replace placeholders in the template and save the YAML file
#         sed -e "s|<TRIPLET_PATH>|\"$triplet_path\"|g" \
#             -e "s|<GROUP_SIZE>|$group_size|g" \
#             "$template" > "$output_dir/$yaml_name"
        
#         echo "Generated $output_dir/$yaml_name"
#     done
#     strategy_number=$((strategy_number + 1))  # Increment strategy number after processing all group sizes
# done

###################
# EXPERIMENT YAML #
###################

#!/bin/bash

# Template files
templates=("./yaml_files/unicoil_template.yaml" "./yaml_files/tilde_template.yaml" "./yaml_files/splade_max_template.yaml")
templates=("./yaml_files/unicoil_frozen_template.yaml" "./yaml_files/splade_max_frozen_template.yaml")


# Loop over strategies and groups
for strategy in {1..3}; do
    for group in {2..8}; do
        # Construct dataset and experiment name
        dataset="strategy${strategy}_group${group}"
        
        for template in "${templates[@]}"; do
            # Get model name from template file
            model=$(basename "$template" "_frozen_template.yaml")
            experiment_name="${model}_strategy${strategy}_group${group}_frozen_qenc"

            # Create output file name
            output_file="./yaml_files/${model}_strategy${strategy}_group${group}_frozen_qenc.yaml"

            # echo "${dataset}"
            # echo "${experiment_name}"
            # echo "${template}"
            # echo "${output_file}"
            # echo "\n"
            
            # Replace placeholders and write to the new file
            sed -e "s|<DATASET>|$dataset|g" -e "s|<EXPERIMENT_NAME>|$experiment_name|g" "$template" > "$output_file"
        done
    done
done

echo "Files generated successfully."
