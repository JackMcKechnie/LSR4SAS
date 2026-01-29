echo "Starting"
#!/bin/bash

# List of non-empty directories
folders=( */ )

# Filter only non-empty directories
non_empty_folders=()
for dir in "${folders[@]}"; do
    # Remove trailing slash
    dir_name="${dir%/}"
    
    # Skip directories containing "run"
    if [[ "$dir_name" == *run* ]]; then
        continue
    fi
    
    # Check if directory is non-empty
    if [ -d "$dir_name" ] && [ "$(ls -A "$dir_name")" ]; then
        non_empty_folders+=("$dir_name")
    fi
done


# Delete each directory and echo
for dir in "${non_empty_folders[@]}"; do
    echo "Deleting: $dir"
    rm -r "$dir"
done
