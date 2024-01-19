#!/bin/bash

# Navigate to the "noise" directory
cd noise

# Loop through all files in the directory
for file in *; do
    # Check if it's a file and not a directory
    if [ -f "$file" ]; then
        # Generate a random prefix
        prefix=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)
        # Duplicate the file with the new prefix
        cp "$file" "${prefix}_$file"
    fi
done
