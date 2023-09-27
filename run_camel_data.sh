#!/bin/bash

# Default datasets folder path (current working directory)
default_datasets_folder="$PWD"

# Use the provided folder path or the default one
datasets_folder="${1:-$default_datasets_folder}"

# Create the datasets folder if it doesn't exist
mkdir -p "$datasets_folder/camel_tools"

# Set the environment variable for camel-tools to point to the datasets folder
export CAMEL_TOOLS_DATA="$datasets_folder/camel_tools"

# Install camel-tools packages
export | camel_data -i all

