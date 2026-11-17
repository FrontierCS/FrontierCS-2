#!/bin/bash

# Initialize variables
total_mem=0
iterations=10
interval=10
output_file="memory_usage.txt"

# Check if the output file exists and create if it does not
if [ ! -f $output_file ]; then
  echo "Creating new file: $output_file"
  echo "Memory Usage Report" > $output_file
  echo "-------------------" >> $output_file
else
  echo "Appending to existing file: $output_file"
fi

# Function to get memory usage in KB
get_memory_usage() {
  free -m | awk '/^Mem:/ {print $3}'
}

sleep 20

# Loop to collect memory usage
for ((i=1; i<=iterations; i++))
do
  mem_usage=$(get_memory_usage)
  echo "Iteration $i: $mem_usage MB" >> $output_file
  total_mem=$((total_mem + mem_usage))
  sleep $interval
done

# Calculate average memory usage
average_mem=$((total_mem / iterations))

# Output average memory usage to the file
echo "-------------------" >> $output_file
echo "Average Memory Usage: $average_mem MB" >> $output_file

# Print the result to the console
cat $output_file