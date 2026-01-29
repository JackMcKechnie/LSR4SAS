#!/bin/bash

# Define the input templates and output directory
TEMPLATE_DIR="/nfs/primary/SPLADE/splade_sans/yaml_files"
OUTPUT_DIR="/nfs/primary/SPLADE/splade_sans/yaml_files"    

# Define the models and datasets
MODELS=("splade" "tilde" "unicoil")
DATASETS=("strategy1" "strategy2" "strategy3") # Replace with your dataset names

# Process each model and dataset combination
for MODEL in "${MODELS[@]}"; do
  TEMPLATE_FILE="${TEMPLATE_DIR}/${MODEL}_distil_template.yaml"

  # Check if the template file exists
  if [[ ! -f "$TEMPLATE_FILE" ]]; then
    echo "Template file $TEMPLATE_FILE not found! Skipping..."
    continue
  fi

  for DATASET in "${DATASETS[@]}"; do
    EXP_NAME="${MODEL}_${DATASET}_distil"
    OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}_${DATASET}_distil.yaml"

    DATASET_REPLACEMENT="ohsumed_${DATASET}_distil"
    # Replace <DATASET> and <EXP_NAME> using sed and save to output file
    sed -e "s/<DATASET>/$DATASET_REPLACEMENT/g" -e "s/<EXP_NAME>/$EXP_NAME/g" "$TEMPLATE_FILE" > "$OUTPUT_FILE"

    echo "Created $OUTPUT_FILE"
  done
done
