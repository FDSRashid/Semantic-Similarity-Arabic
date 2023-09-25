# Semantic-Similarity-Arabic

## General Information
This is a Class meant for specific functionality with Arabic Large Language Models. The class only uses three metrics for Semantic Similarity: Cosine
Similarity , Euclidean Distance, and Jaccard Similarity (for now). It is only meant to be used with transformers that return PyTorch objects. 
Pre-processing is done using the Camel-Tools library.
This classes uses the Faiss library for optimization of comparison for encoded sentences. I will update this to have detailed descriptions
of the models, math, and algorithms - for now, you can consult their library documentation for more details. I have included a dimension reducer module. If you have embedded texts using my class,
you can reduce the dimensions of the embedded texts using the dimension reducer module. 



## How to use:
First clone this repository using this line 


`git clone https://github.com/FDSRashid/Semantic-Similarity-Arabic.git`


Then, move into the newly cloned repository using 

`cd Semantic-Similarity-Arabic/`

Note: for Google Colab Users preface with % 

Next, install using  the following line: 

  
  `pip install .`

  
  The install will take a few minutes. Finally, you can import a module that uses a specific metric by importing from its respectively named folder. This example code loads the CosineSimilarity class:    
  
  `from cosinesimilarity.cosine_similarity import CosineSimilarity` .  
  
Unless updated in future versions, this is how all metrics I implement will be named and organized.

## Usage
For any of the classes that calculate semantic similarity (eg Cosine Similarity), these are the following functions that are available metrics:
preprocess : Preprocess a text of arabic. It unifies all the orthographic varients of a letter, unicode varients, and removes Arabic Vowel Marks. 
calculate_similarity_matrix : finds the similarity matrix of a set of texts, using the metric of the class it is defined by. It also preprocesses beforehand.
find_most_similar_pair: Returns the most similar pair of texts, using the given similarity metric. Returns the score and the index of the texts.
find_most_similar_pairs: Returns a specific number of the most similar pairs of texts, using the given similarity metric. Returns the score and the index of the text.
find_most_similar_sentence: Returns the most similar text among a list of input texts to a single input text. Returns the most similar text and the index of the text.
find_most_similar_sentences: Returns a set number of most similar texts among a list of input texts to a single input text. Returns the most similar texts and the index of the texts.



For all of the calculation functions, preprocessing is done beforehad, so you can shove whatever text you have in there.
These will be the most useful functions for the user. Each Class implementation has its own set of functions useful for implementing its version. Explore if you'd like!



## Description of the metrics
The first type of metrics are embedding based. This first group of metrics involves using a pretrained transformer to transcribe sentences as numerical vectors. Large Language Models use transformers to output sentences into fixed size numerical vectors. The size of the vector depends on the model you want to use. Typically, models output vectors that are 768 elements long. Hugging Face is the best place to find transformer models. All you need is the name of the model you want to use, as a string. Here is a example of a arabic LLM : asafaya/bert-base-arabic. Simply navigate to https://huggingface.co and the search bar to find models you want to use. Since our code assumes arabic models, be sure to find arabic models. For information on transformers, just look up the wikipedia page. It's pretty informative.


Our first embedding metric is Cosine Similarity. Cosine Similarity is the measure of similarity of two vectors , derived from the Euclidean dot product : $$cos(\theta) = \frac{A \cdot B}{||A|| * ||B ||}$$


||X|| is the norm of vector X. For our discussions the norm used is the Euclidean norm, or $L^2$ norm. Its defined as $ ||x || = \sqrt{x \cdot x} $. Cosine Similarity is quiet literally the cosine of the angle formed by two vectors in n-dimensional space. For that reason, cosine similarity has a score from -1 to 1, where -1 is the opposite in meaning, 1 is the exaact same, 0 meaning the two sentences are orthogonal. For our intents and purposes, we will not have 0 in our metrics.


The second embedding metric is Euclidean Distance. While still working with vectors to represent the texts, Eucldean distance is simply the distance of two vectors in n-dimensional space: $$d(A, B) = || A -B ||$$


Theres no normilization done - the previous metric, COsine Similarity, had normalizing done by dividing by the norm of A times norm B. 


The third metric is Jaccard Similarity. To use Jaccard Similarity, I use camel-tools's morphological tokenizer. This function breaks down sentences in arabic into individual words, but also split prefixes, stems, and suffixes according to Arabic Morphological rules. 
Speakers of Arabic know that arabic has different word forms, and complicated system of prefixes and suffixes. This is important because Jaccard Similarity is based on a set of elements, not a numerical vector like the previous two. In our case, each element will be the element of the tokenized sentence, also separating prefixes and suffixes. My intent is to make Jaccard Similarity as accurate as possible for the given arabic text. The Jaccard Similarity is a straight forward formula - its defined as the size of the intersection of two sets divided by the size of the union of two sets : $$J(A, B) = \frac{|A \cap  B|}{| A \cup B |}$$ 


Intersection is the elements present in A and B, Union is the elements present in either A or B. 



## Notes and updates

Important Note: I am intending to implement a class that requires datasets from camel_tools. As of the latest update, Cosine Similarity and Euclidean Distance Classes will not require these datasets. However, to implement jaccard similarity, i am using their models specifically to tokenize arabic words and split words into suffixes and prefixes as well. I've linked their github on instructions to download their data sets. Please follow their instructions strictly - issues that come from the jaccard class 
pertaining to downloading the dataset I can't help with. Note there is different insructions for using the dataset on desktop and on google colab. Consult : https://github.com/CAMeL-Lab/camel_tools for all the needed information

Update : I've added a shell script that does the downloading the camel-tools dataset downloading for you. All you have to do
is specify where you want the 'camel_tools' folder to be, in . To run it, after cloning and pip installing, run the following line of shell : `./run_camel_data.sh` . This code will use the current working directory to place_camel_tools. If you want to specifiy a different location, run this instead : `./run_camel_data.sh /custom/path/to/datasets` . Remember to change the custom/path bit to your own location.


Keep in mind this is downloading data, if you dont want to repeat downloading the same data and wasting space just set the environment variable to the location where you first downloadeded camel_tools. Instructions are better shown on the camel_tools documentation so i strongly recomend going there.

## Authorship
Author : Ferdaws Rashid


Email: frashid@berkeley.edu
