#!/bin/bash

# Initialize an empty array to track PIDs of running processes
pids=()

# Loop over the range with step size of 10
for i in {1..30522..10}
do
    # Run the python script in the background
    python oracle_example.py --i $i &
    
    # Get the PID of the last background process
    pids+=($!)  # $! gives the PID of the last background process

    # If there are 100 processes running, wait for one to finish
    if [ ${#pids[@]} -ge 20 ]; then
        # Wait for the first process in the list to finish
        wait ${pids[0]}
        
        # Remove the finished process from the list
        pids=("${pids[@]:1}")
    fi
done

# Wait for any remaining processes to finish
wait
