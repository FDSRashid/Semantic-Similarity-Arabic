

from setuptools import setup, find_packages

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
    entry_points={
        'console_scripts': [
            'setup-datasets=Semantic-Similarity-Arabic.setup_datasets:setup_datasets',
        ],
    },
)





