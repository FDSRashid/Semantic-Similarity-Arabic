import subprocess
import os

def setup_datasets():
    # Download datasets using camel_data
    subprocess.run(["camel_data", "-i", "all"])

if __name__ == "__main__":
    setup_datasets()
