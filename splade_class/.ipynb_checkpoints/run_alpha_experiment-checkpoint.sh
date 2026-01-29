for alpha in $(seq 0.1 0.1 0.9); do
    python train_batches.py --alpha "${alpha}"
    
    python splade_class.py \
        --index_loc "./indices/${alpha}_l1reg_add" \
        --run_output_path "./runs/${alpha}_l1reg_add.run" \
        --experiment_name "${alpha}_l1reg_add" \
        --parameter "./trained_parameter_${alpha}_l1reg.pkl" \
        --combination_method "add"
    
    python splade_class.py \
        --index_loc "./indices/${alpha}_l1reg_multiply" \
        --run_output_path "./runs/${alpha}_l1reg_multiply.run" \
        --experiment_name "${alpha}_l1reg_multiply" \
        --parameter "./trained_parameter_${alpha}_l1reg.pkl" \
        --combination_method "multiply"
done

