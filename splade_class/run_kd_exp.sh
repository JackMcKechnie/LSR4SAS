echo "Starting retrain experiments..."
# ./launch_train_and_test_dev.sh hn_t5_retrain_retrain "0"
./launch_train_and_test_dev.sh hn_t5_base_retrain "0"
# ./launch_train_and_test_dev.sh hn_t5_large_retrain "0"
echo "Retrain experiments done!"

# echo "Starting train from scratch experiments..."
# ./launch_train_and_test_dev.sh hn_t5_base_scratch "0" &
# ./launch_train_and_test_dev.sh hn_t5_large_scratch "1" &
# ./launch_train_and_test_dev.sh hn_t5_retrain_scratch "2" &

# wait
# echo "Train from scratch experiments done!"