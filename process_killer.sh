#!/bin/bash

process_name="$1"  # Get the process name from the first command-line argument

# Find processes with the given name and kill them
pgrep -x "$process_name" | while read -r pid; do
  kill -9 "$pid" && echo "Killed process $pid with name $process_name" || echo "Failed to kill process $pid with name $process_name"
done 
