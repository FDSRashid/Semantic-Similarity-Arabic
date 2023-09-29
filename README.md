[![CI](https://github.com/FDSRashid/Semantic-Similarity-Arabic/actions/workflows/master.yml/badge.svg)](https://github.com/FDSRashid/Semantic-Similarity-Arabic/actions/workflows/master.yml)



# Semantic-Similarity-Arabic

## General Information
This is a Class meant for specific functionality with Arabic Large Language Models. The class  uses five metrics for Semantic Similarity: Cosine
Similarity , Euclidean Distance, , Jensen-Shannon Divergence, Word-Movers-Distance, and Jaccard Similarity (for now). Classes that use embedding based metrics work with the transformer library and return torch tensors, so keep that in mind. The Word Movers Distance Class only uses the Aravec models, as there is only one repository i could find that had a word to vector model in Arabic.
Pre-processing is done using the Camel-Tools library.
This classes uses the Faiss library for optimization of comparison for encoded sentences. I will update this to have detailed descriptions
of the models, math, and algorithms - for now, you can consult their library documentation for more details. I have included a dimension reducer module. If you have embedded texts using my class,
you can reduce the dimensions of the embedded texts using the dimension reducer module. 



Please note: The classes that use transformers require tokenizers and models from the Hugging Face Library. So long as the model you want to use has those models, you can use them here. Simply go to https://huggingface.co , search arabic models, and the string of the name of the model can be used as the input for these classes. 


## How to Install:
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
For any of the classes that calculate semantic similarity (eg Cosine Similarity), these are the following functions that are available for all metrics:



preprocess : Preprocess a text of arabic. It unifies all the orthographic varients of a letter, unicode varients, and removes Arabic Vowel Marks. 



calculate_similarity_matrix : finds the similarity matrix of a set of texts, using the metric of the class it is defined by. It also preprocesses beforehand.



find_most_similar_pair: Returns the most similar pair of texts, using the given similarity metric. Returns the score and the index of the texts.



find_most_similar_pairs: Returns a specific number of the most similar pairs of texts, using the given similarity metric. Returns the score and the index of the text.



find_most_similar_sentence: Returns the most similar text among a list of input texts to a single input text. Returns the most similar text and the index of the text.



find_most_similar_sentences: Returns a set number of most similar texts among a list of input texts to a single input text. Returns the most similar texts and the index of the texts.




For all of the calculation functions, preprocessing is done beforehad, so you can shove whatever text you have in there.
These will be the most useful functions for the user. Each Class implementation has its own set of functions useful for implementing its version. Explore if you'd like!


The classes that use a transformer model have a option that lets you send the model to a gpu, if you have one in your machine. Simply set the gpu variable to True when you instantiate a Class object, and the model will be sent to your GPU. Handy for large datasets.


## Description of the metrics
The first type of metrics are embedding based. This first group of metrics involves using a pretrained transformer to transcribe sentences as numerical vectors. Large Language Models use transformers to output sentences into fixed size numerical vectors. The size of the vector depends on the model you want to use. Typically, models output vectors that are 768 elements long. Hugging Face is the best place to find transformer models. All you need is the name of the model you want to use, as a string. Here is a example of a arabic LLM : asafaya/bert-base-arabic. Simply navigate to https://huggingface.co and the search bar to find models you want to use. Since our code assumes arabic models, be sure to find arabic models. For information on transformers, just look up the wikipedia page. It's pretty informative.


Our first embedding metric is Cosine Similarity. Cosine Similarity is the measure of similarity of two vectors , derived from the Euclidean dot product : $$cos(\theta) = \frac{A \cdot B}{||A|| * ||B ||}$$


||X|| is the norm of vector X. For our discussions the norm used is the Euclidean norm, or $L^2$ norm. Its defined as $||x || = \sqrt{x \cdot x}$ . Cosine Similarity is quite literally the cosine of the angle formed by two vectors in n-dimensional space. For that reason, cosine similarity has a score from -1 to 1, where -1 is the opposite in meaning, 1 is the exaact same, 0 meaning the two sentences are orthogonal. For our intents and purposes, we will not have 0 in our metrics.


The second embedding metric is Euclidean Distance. While still working with vectors to represent the texts, Eucldean distance is simply the distance of two vectors in n-dimensional space: $$d(A, B) = || A -B ||$$


Theres no normilization done - the previous metric, COsine Similarity, had normalizing done by dividing by the norm of A times norm B. 


The third metric is Jaccard Similarity. To use Jaccard Similarity, I use camel-tools's morphological tokenizer. This function breaks down sentences in arabic into individual words, but also split prefixes, stems, and suffixes according to Arabic Morphological rules. 
Speakers of Arabic know that arabic has different word forms, and complicated system of prefixes and suffixes. This is important because Jaccard Similarity is based on a set of elements, not a numerical vector like the previous two. In our case, each element will be the element of the tokenized sentence, also separating prefixes and suffixes. My intent is to make Jaccard Similarity as accurate as possible for the given arabic text. The Jaccard Similarity is a straight forward formula - its defined as the size of the intersection of two sets divided by the size of the union of two sets : $$J(A, B) = \frac{|A \cap  B|}{| A \cup B |}$$ 


Intersection is the elements present in A and B, Union is the elements present in either A or B. 



The forth metric is Jensen-Shannon metric. We return to the embedding method, with a little adjustment. A transformer model, when modeling a embedding for a input text, works by gradually building up a numerical representation of a text, using many layers. The output of the final layer is called last_hidden_state. Each column represents a numerical representation of contextual information, semantic meaning of a specific word or token. Note - the model does not return columns based on a 1 word/token to one columns - layers of information are stored in the embeddings. Jensen-Shannon Divergence in general is measuring the similarity between two probability distributions. In our case, we want to find the probability distribution of the columns of our embedded text. To generate the probability distribution of columns of our sentence, we apply the softmax function onto the columns of a single embedded vector. The softmax function takes a vector input of N real numbers, and returns a probability distribution consising of the same number of probabilities. The softmax function is defined for our case as follows:

$$\sigma(z_i) = \frac{e^{z_i}}{\sum_{ j \in N} e^{z_j}}$$

For i from 1 to N, and z being a set of numbers that are of size N. Essentially, for each element of our input vector - which in our case is the embedded represenataion of the text, we get a output probability. 



The softmax function solves the first step of the Jensen-Shannon Divergence. The next is using that probability distribution. For the J-S Divergence, we will get two different probability distributions from two texts. Just like cosine similarity was a pairwise metric. The other formula that J-S requires is entropy - this measures disorder or information from a discrete probability distribution. Wouldn't you know it, softmax returns a discrete probability distribution. Entropy is defined as follows: $$H(X) = - \sum_{x \in X} p(x) * log_{2} p(x) $$ So, for every invidual probability we have, we shove that into this function.



The last step to J-S Divergence (exhale) is to plug the two probability distributions from two embedded texts into the J-S formula. Lets say we have two probability distributions A and B. Let M, the average of the two distributions, be defined as $M = \frac{1}{2}(A + B)$ . Then the Jensen-Shannon divergence of two probability distributions is as follows: $$JSD(A || B) = \frac{1}{2} H(M) - \frac{1}{2} \left(  H(A) + H(B)  \right) $$ This gets us a JS Divergence of two distributions, where 0 means minimmal divergence. For this metric, we cannot have negative divergence, and the closer to zero, the more similar the two texts are. 


Metric Number 5 is Word-Movers Distance. We are using a different type of language model for this, a word2vector model. These models convert words or phrases into vector represenations. Essentially, over the training data, these models develop a vocabulary set and a numerical representation of each word it takes in.  W2Vec models have two types: Continuous Bag of Words (CBOW) and Skip-gram. We are only using Arabic W2Vec models, and i could only find 1 - so we are limiting our options to that one repository , Aravec. Aravec has models based on CBOW and SKip-gram, models trained on wikipedia articles and tweets, and different models based on how long you want each embedded vector to be. Just know that their Wikipedia Skipgram model for unigram models dont work. 

The Word Movers Distance metric requires embeddings which map words to vectors - hence word2vec embeddings like Aravec are needed. It is simply the minimum distance to traverse words from text A to text B. It's based on the earth mover's distance, a measure of distance between two frequency distributions. For more information, consider looking at their documentation : https://radimrehurek.com/gensim/auto_examples/tutorials/run_wmd.html#. For our purposes, the closer the score is to zero, the more similar the texts are. 





## Notes and updates

Important Note: I am intending to implement a class that requires datasets from camel_tools. As of the latest update, Cosine Similarity and Euclidean Distance Classes will not require these datasets. However, to implement jaccard similarity, i am using their models specifically to tokenize arabic words and split words into suffixes and prefixes as well. I've linked their github on instructions to download their data sets. Please follow their instructions strictly - issues that come from the jaccard class 
pertaining to downloading the camel-tools dataset I can't help with. Note there is different insructions for using the dataset on desktop and on google colab. Consult : https://github.com/CAMeL-Lab/camel_tools for all the needed information.



As a TLDR  - you can download their datasets using this command line `camel_data -i all`. Then,
to set the environment variable as per their instructions you can do this shell line: `env CAMELTOOLS_DATA=/root/.camel_tools` . Note that Colab requires % before the shell, and ! before command lines. 


Keep in mind this is downloading data, if you dont want to repeat downloading the same data and wasting space just set the environment variable to the location where you first downloadeded camel_tools. Instructions are better shown on the camel_tools documentation so i strongly recomend going there.

## Authorship
Author : Ferdaws Rashid


Email: frashid@berkeley.edu
