#!/bin/bash

# Specify the path to the datasets folder (change this path as needed)
datasets_folder="/path/to/your/datasets/camel_tools"

# Create the datasets folder if it doesn't exist
mkdir -p "$datasets_folder"

# Set the environment variable for camel-tools to point to the datasets folder
export CAMELTOOLS_DATA="$datasets_folder"

# Install camel-tools packages
camel_data -i all

# Add more camel_data commands as needed
# For example:
# camel_data command1 arg1 arg2
# camel_data command2 arg3
