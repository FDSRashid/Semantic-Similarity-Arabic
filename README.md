# Semantic-Similarity-Arabic
This is a Class meant for specific functionality with Arabic Large Language Models. The class only uses one metric for Semantic Similarity,
COsine Similarity for now, and is only meant to be used with transformers that return PyTorch objects. I intend on updating this to
work with TensorFlow objects as well. 
Pre-processing is done using the Camel-Tools library.
This classes uses the Faiss library for optimization of comparison for encoded sentences. I will update this to have detailed descriptions
of the models, math, and algorithms - for now, you can consult their library documentation for more details.
