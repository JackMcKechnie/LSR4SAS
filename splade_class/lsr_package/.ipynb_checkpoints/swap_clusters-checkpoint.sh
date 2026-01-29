#!/bin/bash

# Function to replace clusters
replace_clusters() {
    local file="$1"
    
    # Check if file is readable and writable
    if [ ! -r "$file" ] || [ ! -w "$file" ]; then
        echo "Skipping $file: No read/write permissions"
        return
    fi
    
    # Use sed to replace 'nfs' with 'nfs'
    # sed -i 's/nfs/nfs/g' "$file"
    sed -i 's/nfs/nfs/g' "$file"

    
    # Check if replacement was made
    if [ $? -eq 0 ]; then
        echo "Modified: $file"
    fi
}

# Check if directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Directory to search
SEARCH_DIR="$1"

# Validate directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory $SEARCH_DIR does not exist."
    exit 1
fi

find "$SEARCH_DIR" -type f \( -not -path "*/outputs/*" -and -not -path "*/wandb/*" \) -print0 | while IFS= read -r -d '' file; do
    replace_clusters "$file"
done

echo "Replacement process completed."