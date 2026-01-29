#!/bin/bash

# Supply the PID of the process to monitor
PID=5198

# Log file to store memory usage and process status
LOGFILE="memory_usage_and_process.log"

# Interval between logs (in seconds)
INTERVAL=10

# Print header
echo "Timestamp    | Total RAM (h) | Used RAM (h) | Free RAM (h) | Buffers/Cache (h) | Available RAM (h) | Total RAM (B) | Used RAM (B) | Free RAM (B) | Buffers/Cache (B) | Available RAM (B) | Process Status" >> $LOGFILE
echo "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" >> $LOGFILE

# Loop to log memory usage and process status
while true
do
    # Get current timestamp
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # Get memory usage in human-readable format
    MEM_HUMAN=$(free -h | awk 'NR==2{printf "%s | %s | %s | %s | %s", $2, $3, $4, $6, $7}')

    # Get memory usage in bytes
    MEM_BYTES=$(free -b | awk 'NR==2{printf "%s | %s | %s | %s | %s", $2, $3, $4, $6, $7}')

    # Check if the process is running
    if ps -p $PID > /dev/null; then
        PROCESS_STATUS="Running"
    else
        PROCESS_STATUS="Not Running"
    fi

    # Log the data
    echo "$TIMESTAMP | $MEM_HUMAN | $MEM_BYTES | $PROCESS_STATUS" >> $LOGFILE

    # Wait for the specified interval
    sleep $INTERVAL
done
