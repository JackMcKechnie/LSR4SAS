#!/bin/bash

# Define the input template file and output directory
TEMPLATE_FILE="/nfs/primary/SPLADE/splade_sans/learned-sparse-retrieval-1.0.0/lsr/configs/dataset/ohsumed_distil_template.yaml"
OUTPUT_DIR="/nfs/primary/SPLADE/splade_sans/yaml_files/"

# Loop through strategies and create YAML files
for STRAT in 1 2 3; do
  # Define the output file path
  OUTPUT_FILE="${OUTPUT_DIR}ohsumed_strategy${STRAT}_distil.yaml"

  # Use sed to replace <STRAT> in the template and write to the output file
  sed "s/<STRAT>/$STRAT/g" "$TEMPLATE_FILE" > "$OUTPUT_FILE"

  echo "Created $OUTPUT_FILE"
done