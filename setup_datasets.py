import subprocess
import os

def setup_datasets():
    # Download datasets using camel_data
    subprocess.run(["camel_data", "-i", "all"])

    # Set the CAMELTOOLS_DATA environment variable
    os.environ["CAMELTOOLS_DATA"] = "~/.camel_tools"

if __name__ == "__main__":
    setup_datasets()
