

from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess




class CustomInstall(install):
    user_options = install.user_options + [
        ('install-datasets', None, 'Install camel-tools datasets and set environment variable to the location of the download.'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.custom_install = False

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # Run the default installation first
        install.run(self)
        
        if self.custom_install:
            # Download datasets using camel_data
            subprocess.run(["camel_data", "-i", "all"])

            # Set the CAMELTOOLS_DATA environment variable
            os.environ["CAMELTOOLS_DATA"] = "~/.camel_tools"



setup(
    name='semantic_similarity_arabic',
    version='1.1.0',
    author="Ferdaws Rashid",
    packages=find_packages(),
    install_requires=[
        # Include any additional dependencies here
        'numpy>=1.18.0',
        'torch>=1.9.1',
        'faiss-cpu>=1.7.4',
        'camel-tools>=1.5.2',
        'transformers>=4.30.2',
        'scikit-learn',
        'pytest',
        'datasets',
        'umap-learn',
        'scipy'
    ],
    python_requires='>=3.7',
    cmdclass={'install': CustomInstall},
)





