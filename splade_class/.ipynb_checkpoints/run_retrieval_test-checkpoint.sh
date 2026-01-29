# CUDA_VISIBLE_DEVICES=3 python splade_class.py \
#     --index_loc "./indices/l1reg_add_negate" \
#     --run_output_path "./runs/l1reg_add_negate.run" \
#     --experiment_name "l1reg_add_negate" \
#     --parameter "./trained_parameter_all_data.pkl" \
#     --combination_method "add"

# CUDA_VISIBLE_DEVICES=3 python splade_class.py \
#     --index_loc "./indices/l1reg_multiply_negate" \
#     --run_output_path "./runs/l1reg_multiply_negate.run" \
#     --experiment_name "l1reg_multiply_negate" \
#     --parameter "./trained_parameter_all_data.pkl" \
#     --combination_method "multiply"

#  CUDA_VISIBLE_DEVICES=3 python splade_class.py \
#     --index_loc "./indices/noreg_add_negate" \
#     --run_output_path "./runs/noreg_add_negate.run" \
#     --experiment_name "noreg_add_negate" \
#     --parameter "./trained_parameter_no_regularisation.pkl" \
#     --combination_method "add"

# CUDA_VISIBLE_DEVICES=3 python splade_class.py \
#     --index_loc "./indices/noreg_multiply_negate" \
#     --run_output_path "./runs/noreg_multiply_negate.run" \
#     --experiment_name "noreg_multiply_negate" \
#     --parameter "./trained_parameter_no_regularisation.pkl" \
#     --combination_method "multiply"

# # CUDA_VISIBLE_DEVICES=3 python splade_class.py \
# #     --index_loc "./indices/cocondenser_distil" \
# #     --run_output_path "./runs/cocondenser_distil.run" \
# #     --experiment_name "cocondenser_distil" \
# #     --parameter "./trained_parameter_all_data.pkl" \
# #     --combination_method "none"

python splade_class_v2.py \
    --index_loc "./indices/test_v24" \
    --run_output_path "./runs/test_v24.run" \
    --experiment_name "noreg_multiply_negate" \
    --parameter "./trained_parameter_all_data.pkl" \
    --combination_method "add"