# Semantic-Similarity-Arabic
This is a Class meant for specific functionality with Arabic Large Language Models. The class only uses one metric for Semantic Similarity,
Cosine Similarity for now, and is only meant to be used with transformers that return PyTorch objects. I intend on updating this to
work with TensorFlow objects as well. 
Pre-processing is done using the Camel-Tools library.
This classes uses the Faiss library for optimization of comparison for encoded sentences. I will update this to have detailed descriptions
of the models, math, and algorithms - for now, you can consult their library documentation for more details.

How to use: First clone this repository using this line `git clone https://github.com/FDSRashid/Semantic-Similarity-Arabic.git`.

  Then, move into the newly cloned repository using `cd Semantic-Similarity-Arabic/`.   Then, install using     `pip install .`. The install will take a few minutes. Finalyy, you can import a module that uses a specific metric by importing from its respectively named folder. this example code loads the CosineSimilarity class:     `from cosinesimilarity.CosineSimilarity import CosineSimilarity` .     Unless updated in future versions, this is how all metrics I implement will be named and organized.

Author : Ferdaws Rashid


Email: frashid@berkeley.edu
