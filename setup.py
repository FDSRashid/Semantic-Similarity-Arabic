

from setuptools import setup, find_packages

setup(
    name='semantic_similarity_arabic',
    version='1.0',
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
        'scipy',
        'gensim',
        'pot',
        'qalsadi',
        'Arabic-Stopwords'
    ],
    python_requires='>=3.10',
)





