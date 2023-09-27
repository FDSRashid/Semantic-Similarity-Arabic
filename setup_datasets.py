import subprocess
import os

def setup_datasets():
    # Download datasets using camel_data
    subprocess.run(["camel_data", "-i", "all"])

    # Set the CAMELTOOLS_DATA environment variable
    data_path = os.path.expanduser("~/.camel_tools")  # Expands '~' to the user's home directory
    os.environ["CAMELTOOLS_DATA"] = data_path


if __name__ == "__main__":
    setup_datasets()
