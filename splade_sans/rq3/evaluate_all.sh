cd ../learned-sparse-retrieval-1.0.0/outputs

file_array=($(ls | grep -i "distil" | grep -vi "msmarco"))

cd ../..
# Print the array contents
for file in "${file_array[@]}"; do
    echo "Evaluating $file"
    ./launch_evaluate.sh "$file" "1"
done


