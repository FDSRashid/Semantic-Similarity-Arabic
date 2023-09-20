

# -*- coding: utf-8 -*-
"""
    Author: Ferdaws Rashid
    email: frashid@berkeley.edu
"""
from semanticsimilarityarabic.semantic_similarity_arabic import SemanticSimilarityArabic
import numpy as np
import torch
import faiss


from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
#from google.colab import drive
#drive.mount('/gdrive')
#os.environ['CAMELTOOLS_DATA'] = '/gdrive/MyDrive/camel_tools'
#from camel_tools.tokenizers.word import simple_word_tokenize
#from camel_tools.morphology.database import MorphologyDB
#from camel_tools.morphology.analyzer import Analyzer

#from camel_tools.morphology.database import MorphologyDB
#from camel_tools.morphology.generator import Generator

#from camel_tools.tokenizers.word import simple_word_tokenize
#from camel_tools.disambig.mle import MLEDisambiguator

#from camel_tools.tokenizers.morphological import MorphologicalTokenizer


#from camel_tools.tagger.default import DefaultTagger

#from camel_tools.dialectid import DialectIdentifier
#from camel_tools.sentiment import SentimentAnalyzer

class CosineSimilarity(SemanticSimilarityArabic):
  """
    A class for processing and comparing the Cosine similarity of sentences using Arabic  Models.
    Note: All preproccessing is done for Arabic, so Please use only Arab Texts to use this model.
    Important: This class uses a autotokenizer to process sentences. To instantiate a instance of this class,
    you need a string to the pretrained model. One example is 'CAMeL-Lab/bert-base-arabic-camelbert-ca' There are plenty of others on hugging face.
    Important note, if you choose not to truncate, please specify the max_length of the model. this code will not work unless you do so.
    Args:
        model_name (str): The name of the pretrained model to use for encoding sentences.
        batch_size (int) = 10 : the size of the batches you want to process data by. depends on your computational power, and can be increased
        gpu (bool): wether you have a gpu or not. If so, it will send the model to the gpu to handle the more intensive work
        tokenizer_args (dict, optional): Additional arguments for the tokenizer.
            These arguments will be passed to the tokenizer during initialization.
            For a list of possible tokenizer arguments, refer to the Hugging Face Transformers documentation.
        model_args (dict, optional): Additional arguments for the model.
            These arguments will be passed to the model during initialization.
            For a list of possible model arguments, refer to the Hugging Face Transformers documentation.
    
    Example:
      >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca",
       tokenizer_args={"model_max_length": 512},
        ...     model_args={"num_labels": 2})
      


    Dependencies:
      -transformers
      -Camel-tools
      -torch
      -numpy
      -sci-kit learn

    Methods:
        preprocess(sentence: str) -> str:
            Preprocesses a sentence before encoding.
            
        preprocess_batch(sentences: List[str]) -> List[str]:
          Preprocesses a  batch of sentences before encoding.

        encode_sentences(sentence: List[str]) -> List[torch.Tensor]:
            Encodes a  list of sentences to a embeding. returns a list of respective tensors. be careful how its used. 

        find_most_similar_pair(sentences: List[str]) -> Tuple[str, str, float]:
            Finds the two most similar sentences from a list of sentences.

        find_most_similar_pairs(sentences: List[str], n: int) -> List[Tuple[str, str, float]]:
            Finds the n most similar sentences from a list of sentences.

        find_most_similar_sentence(sentences: List[str], sentence: str) -> Tuple[str, float, int]:
            Finds the most similar sentence for a input sentence from a list of sentences.
            returns the sentence, the similarity score, and which number of the sentence it returns
            

        preprocess_for_faiss(embedded_sentences: List[torch.Tensor]) -> np.ndarray
            make a C-contiguous array of encoded sentences for similarity calculations

        find_most_similar_sentences(sentences: List[str], sentence: str, n: int) -> Tuple[list[str], list[float], list[int]]:
            Finds the number n of the most similar sentence's for a input sentence from a list of sentences.
            returns a list of sentences, the similarity scores, and which number of the sentence's it returns

        calculate_similarity_matrix(sentences: List[str]) ->  np.nparray :
          calculates the similarity matrix of a list of sentences, doing pre-processing as well.
          to access the similarity score of the i'th and j'th sentences, find the matrix element at
          [i, j].

  """
  def __init__(self, model_name, batch_size = 10, gpu = False, tokenizer_args=None, model_args=None):
    try:
      self.tokenizer = AutoTokenizer.from_pretrained(model_name,  **(tokenizer_args or {}))
      self.model = AutoModel.from_pretrained(model_name, **(model_args or {}))
      self.batch_size = batch_size
      self.gpu = gpu
      if self.gpu:
          try:
            self.model.to('cuda')
          except Exception as e:
            raise ValueError(f"Warning: GPU not available. Falling back to CPU. Error: {e}")

    except Exception as e:
      raise ValueError(f"Failed to initialize model: {e}")
  def preprocess(self, sentence):
    """
        Preprocesses a sentence before encoding.
        First, normalize unicode, meaning all varients are a letter are returned to a canonical form. Then normalize alef, alef maksura, and tah marbuta, which
        is the same idea but for those letters.Finally, vowel marks are removed. This is done becasue it decreases data sparsity. 
        For further information,
        consult https://camel-tools.readthedocs.io/en/latest/index.html

        Args:
            sentence: A string of Arabic Words you wish to pre-process for NLP
            

        Returns:
            sentence (string), a string with proper formatting

        Example:
            Example usage of preprocess:
            
            >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca") #default size of batch is 10
            >>> result = model.preprocess(" فَسَمِعَ رَجُلا ")
            >>> print(result)
            " فسمع رجلا "
        """
    
    if not isinstance(sentence, str):
      raise ValueError("Input sentence must be a string")
    sentence = normalize_unicode(sentence)
    sentence = normalize_alef_ar(sentence)
    sentence = normalize_alef_maksura_ar(sentence)
    sentence = normalize_teh_marbuta_ar(sentence)
    sentence = dediac_ar(sentence)
    return sentence


  def preprocess_batch(self, sentences):
    """
        Preprocesses a  a batch of sentences before encoding.
        First, normalize unicode, meaning all varients are a letter are returned to a canonical form. Then normalize alef, alef maksura, and tah marbuta, which
        is the same idea but for those letters.Finally, vowel marks are removed. This is done becasue it decreases data sparsity. 
        For further information,
        consult https://camel-tools.readthedocs.io/en/latest/index.html

        Args:
            sentences: A  List of type string of Arabic Sentences you wish to pre-process for NLP
            

        Returns:
            sentence (string), a string with proper formatting

        Example:
            Example usage of preprocess_batch:
            
            >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca") #default size of batch is 10
            >>> result = model.preprocess(" فَسَمِعَ رَجُلا ")
            >>> print(result)
            " فسمع رجلا "
        """

    if not isinstance(sentences, list):
        # If a single sentence is provided, wrap it in a list for consistency
      sentences = [sentences]

    preprocessed_sentences = []

        # Split the sentences into batches
    for i in range(0, len(sentences), self.batch_size):
      batch = sentences[i:i + self.batch_size]
     # Combine the sentences with 'fin' separator
      combined_sentences = ' fin '.join(batch)

      # Preprocess the combined string
      preprocessed_text = self.preprocess(combined_sentences)

      # Split the preprocessed text back into sentences using 'fin' as the separator
      preprocessed_batch = preprocessed_text.split(' fin ')

      preprocessed_sentences.extend(preprocessed_batch)

    return preprocessed_sentences

  def encode_sentences(self, sentences):
        # Preprocess all sentences in a batch
      """
        Encodes a Sentence or List of Sentences uses the Pre-Trained Model defined by the Class Instantiation. For example, you can put 'Camel-Bert' as the model you want to use 
        and it will use their model. Please consult the source of the model you are using - this class assumes that the model will return features in PyTorch. To learn more about Hosted
        Large Language Models, consult https://huggingface.co And look at examples on how to use a tokenizer. This Class uses a AutoTokenizer to encode sentences. 

        Args:
            sentence: A single String of List of Strings that are the sentances you wish to encode.
            

        Returns:
            embedded_sentences list : A list of embedded sentences. each element is the last hidden state of the corresponding output

        Example:
            Example usage of encode_sentences:
            
            >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca") #default size of batch is 10
            >>> result = model.encode_sentences(" فَسَمِعَ رَجُلا ")
            >>> print(result.shape)
            (1, 768)
        """  
      if not isinstance(sentences, list):
        # If a single sentence is provided, wrap it in a list for consistency
        sentences = [sentences]
          
      preprocessed_sentences = self.preprocess_batch(sentences)
      max_sequence_length = self.tokenizer.model_max_length   

      encoded_embeddings = []
      for sentence in preprocessed_sentences:
        tokenized_sentence = self.tokenizer(sentence, return_tensors = 'pt', padding = True)
          
        input_ids = tokenized_sentence['input_ids']
        attention_mask = tokenized_sentence['attention_mask']
        token_type_ids = tokenized_sentence['token_type_ids']
        if input_ids.shape[1] <= max_sequence_length:
        # Process the entire sentence
          chunk = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
          }
          if self.gpu:
            chunk = {key: value.to('cuda') for key, value in chunk.items()}
          output = self.model(**chunk)
          cls_representation = output.last_hidden_state[:, 0, :]
          cls_representation = cls_representation.to('cpu')
          encoded_embeddings.append(cls_representation)
        else:
          # Truncate the sentence into chunks and process them separately
          
          for j in range(0, input_ids.shape[1], max_sequence_length):
            chunk_input_ids = input_ids[:, j:j + max_sequence_length]
            chunk_attention_mask = attention_mask[:, j:j + max_sequence_length]
            chunk_token_type_ids = token_type_ids[:, j:j + max_sequence_length]
            chunk = {
                'input_ids': chunk_input_ids,
                'attention_mask': chunk_attention_mask,
                'token_type_ids': chunk_token_type_ids
            }
            if self.gpu:
              chunk = {key: value.to('cuda') for key, value in chunk.items()}
            output = self.model(**chunk)
            cls_representation = output.last_hidden_state[:, 0, :]
            cls_representation = cls_representation.to('cpu')
            encoded_embeddings.append(cls_representation)
          

        # Convert to NumPy array and append to embeddings list
          
          
      return encoded_embeddings
        
          
          
            
            
  def preprocess_for_faiss(self, encoded_embeddings):
    """
    detaches each element, converts each element to numpy array.  then it converts to a C-contiguous array. finally, it stacks all the elements vertically using np.vstack    
    this is done so that the encoded sentences are compatible for faiss algorithms. it also is made compatible with sklearn cosine similarity functions
      Args: 
        encoded_embeddings : a list of torch.Tensors which are the encoded sentences. if not list, wrapts it in list. teehee
      Returns:
        sentences_array : a np array of vertically stacked encoded sentences.


    
    """
    if not isinstance(encoded_embeddings, list):
        # If a single sentence is provided, wrap it in a list for consistency
        encoded_embeddings = [encoded_embeddings]
    expected_shape = None
    for i, sentence in enumerate(encoded_embeddings):
      if not torch.is_tensor(sentence):
         raise ValueError(f"Element at index {i} is not a tensor.")
      shape = sentence.shape
        
        # Check if it matches the expected shape (assuming all tensors should have the same shape)
      if expected_shape is None:
        expected_shape = shape
      elif shape != expected_shape:
         raise ValueError(f"Element at index {i} has dimensions {shape}, expected {expected_shape}.")
    
    return np.vstack([np.ascontiguousarray(tensor.detach().numpy()).astype('float32') for tensor in encoded_embeddings])
  def calculate_similarity_matrix(self, sentences):
    """

    This finds the cosine_similarity matrix of a list of sentences. Recall that the formula for Cosine Simularity is the following: Dot Product of Vectors A and B Divided by the norm of A times B
    Source : https://en.wikipedia.org/wiki/Cosine_similarity 
    Since Our sentences are converted to a vector of numbers, we can use cosine simularity as a way to sind similar sentances. Keep in mind this formula is equal to cosine(theta), so this formula
    returns a value between 0 and 1, where 1 means you have the two exact same sentences, and 0 is a completely unsimilar pair of sentences
    To find the similarity between the i'th and j'th sentence, select the matrix at element (i, j). Diagonal elements are the same sentences,
    so will always have 1.   

        Args:
            sentence: A List of Strings that are the sentances you wish to find the similarity for.
            

        Returns:
            sentence numpy.nparray : A numpy array representing the similarity matrix. 

        Example:
            Example usage of encode_sentences:
            
            >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca") #default size of batch is 10
            >>> result = model.similarity_sentences([a list of sentences with length 3])
            >>> print(result) #just a example matrix
            [[1.         0.89068115 0.877669  ]
            [0.89068115 1.         0.9122398 ]
            [0.877669   0.9122398  1.0000001 ]]
            
        """ 
    if not isinstance(sentences, list) or len(sentences) < 2:
        raise ValueError("Input must be a list of at least two sentences")

        # Preprocess and encode all sentences
    sentence_embeddings = self.preprocess_for_faiss(self.encode_sentences(sentences))
    return cosine_similarity(sentence_embeddings)


  def find_most_similar_pairs(self, sentences, n):
      """

    This finds a specified number of the most similar pairs using cosine simularity.
    This uses the Faiss Library to optimize the speed and memory usage to find the most similar sentences
    For more information on the Faiss library, consult this : https://github.com/facebookresearch/faiss


        Args:
            sentence: A List of Strings that are the sentances you wish to find the similarity for.
            n : the number of pairs to return. so, n = 3 would return the 3 most similar sentance pairs
            

        Returns:
          list [Tuple[int, int, float]] : a list of the index's of the sentences, and the similarity score between them.
              
            

        Example:
            Example usage of encode_sentences:
            
            >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca") #default size of batch is 10
            >>> result = model.find_n_similar_pair(sentences, 2)
            >>> print(result)
            "[( لا يُتَوَضَّأُ مِنْ طَعَامٍ أَحَلَّ اللَّهُ أَكْلَهُ ,
   إِذَا تَوَضَّأَ الْعَبْدُ الْمُسْلِمُ فَغَسَلَ وَجْهَهُ خَرَجَ مِنْ وَجْهِهِ كُلُّ خَطِيئَةٍ نَظَرَ إِلَيْهَا بِعَيْنِهِ آخِرَ قَطْرِ الْمَاءِ ، فَإِذَا غَسَلَ يَدَيْهِ خَرَجَتْ مِنْهُمَا كُلُّ خَطِيئَةٍ بَطَشَهَا بِهِمَا ، ثُمَّ كَذَلِكَ حَتَّى يَخْرُجَ نَقِيًّا مِنَ الذُّنُوبِ ,
  0.61238515),
 ( الْغِيبَةُ تُفْطِرُ الصَّائِمَ وَتَنْقُضُ الْوُضُوءَ ,الْوُضُوءُ مِنَ الْمَذْيِ ، وَالْغُسْلُ مِنَ الْمَنِيِّ ,
  0.8229922)]"
        """ 
      if not isinstance(sentences, list) or len(sentences) < 2:
        raise ValueError("Input must be a list of at least two sentences")
      if n <= 0:
            raise ValueError("The value of 'n' must be greater than 0")

        # Preprocess and encode all sentences
      sentence_embeddings = self.preprocess_for_faiss(self.encode_sentences(sentences))

        # Calculate pairwise similarities
      index = faiss.IndexFlatIP(sentence_embeddings.shape[1])  # Use inner product (cosine similarity)
      faiss.normalize_L2(sentence_embeddings)

        # Add sentence embeddings to the index
      index.add(sentence_embeddings)

        # Find the two most similar sentences
      k = 2  # Number of similar sentences to retrieve
      D, I = index.search(sentence_embeddings, k)
      most_similar_pairs = []
      for i in range(len(sentences)):
        similar_indices = I[i]
        similarity_scores = D[i]
        top_n_indices = similar_indices[1:n + 1]  # Exclude the first index, which is the sentence itself
        top_n_scores = similarity_scores[1:n + 1]

        # Create a set to keep track of unique pairs
        unique_pairs = set()

        for j, sim_idx in enumerate(top_n_indices):
            # Check if the pair is unique by comparing sorted tuples
            pair = tuple(sorted([i, sim_idx]))
            if pair not in unique_pairs:
                most_similar_pairs.append((i, sim_idx, top_n_scores[j]))
                unique_pairs.add(pair)

    
      most_similar_pair = sorted(most_similar_pairs, key=lambda x: x[2])[:n]
      return most_similar_pair

  def find_most_similar_sentence(self, sentences, sentence):

    """

    This finds a the most similar sentence from a list of sentances. You input a sentance to co
    compare the list to. It returns the sentence, the score, and the index of the sentence. 
    That way you can keep track of which element it came from.



        Args:
            sentences: A List of Strings that are the sentances you wish to find the similarity for.
            sentence: a single string that is the sentance you want to compare the list to.
            

        Returns:
            most_similar_sentence, similarity_score, most_similar_index: string of the sentence, a float of the similarity score, and index as a integer

        Example:
            Example usage of encode_sentences:
            
            >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca") #default size of batch is 10
            >>> result = model.find_most_similar_sentence(sentence, list of sentences)
            >>> print(result)
            " فَسَمِعَ رَجُلا يَقْرَأُ : /4 قُلْ هُوَ اللَّهُ أَحَدٌ سورة الإخلاص آية 1 /4 ، إِلَى آخِرِهَا ، فَقَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ : "" وَجَبَتْ "" ، فَقُلْتُ : مَاذَا يَا رَسُولَ اللَّهِ ؟ ، فَقَالَ : "" الْجَنَّةُ "" ، قَالَ أَبُو هُرَيْرَةَ : فَأَرَدْتُ أَنْ أَذْهَبَ إِلَى الرَّجُلِ فَأُبَشِّرَهُ ، ثُمَّ خِفْتُ أَنْ يَفُوتَنِي الْغَدَاءُ مَعَ رَسُولِ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ فَآثَرْتُ الْغَدَاءَ مَعَ رَسُولِ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ ، ثُمَّ ذَهَبْتُ إِلَى الرَّجُلِ فَوَجَدْتُهُ قَدْ ذَهَبَ ,
1.0,
 0)"
        """ 
    if not isinstance(sentences, list) or len(sentences) < 2:
      raise ValueError("Input must be a list of at least two sentences")

        # Preprocess and encode all sentences
    sentences_embeddings = self.preprocess_for_faiss(self.encode_sentences(sentences))
    sentence_embedding = self.preprocess_for_faiss(self.encode_sentences(sentence))
    similarities = cosine_similarity(sentence_embedding, sentences_embeddings)[0]
    most_similar_index = np.argmax(similarities)
    most_similar_sentence = sentences[most_similar_index]
    similarity_score = similarities[most_similar_index]
    return most_similar_sentence, similarity_score, most_similar_index




  def find_most_similar_sentences(self, sentences, sentence, n = 2):
      """

    This finds a the most similar sentence from a list of sentances. You input a sentance to co
    compare the list to. It returns the sentence, the score, and the index of the sentence. 
    That way you can keep track of which element it came from.



        Args:
            sentences: A List of Strings that are the sentances you wish to find the similarity for.
            sentence: a single string that is the sentance you want to compare the list to.
            n: the number of the most similar_sentences to return
            

        Returns:
            most_similar_sentences, similarity_scores, most_similar_index's:  a list of string of the sentences, list of float's of the similarity scores, 
            and list of indexes as ints

        Example:
            Example usage of encode_sentences:
            
            >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca") #default size of batch is 10
            >>> strings, scores, idxs = model.find_most_similar_sentences(list of sentences, sentence , 2)
            >>> print(len(scores))
            >>> 2
            "
        """
      if not isinstance(sentences, list) or len(sentences) < 2:
        raise ValueError("Input must be a list of at least two sentences")
      if n > len(sentences):
        raise ValueError("'n' cannot be greater than the number of sentences in the list.")
  
      encoded_sentences = self.preprocess_for_faiss(self.encode_sentences(sentences))
      encoded_sentence = self.preprocess_for_faiss(self.encode_sentences(sentence)) 
      index = faiss.IndexFlatIP(encoded_sentences.shape[1])
      faiss.normalize_L2(encoded_sentences)
      faiss.normalize_L2(encoded_sentence)
      index.add(encoded_sentences)
      encoded_sentence = encoded_sentence.reshape(1, -1)
      distances, similar_indices = index.search(encoded_sentence, n)
      top_similar_sentences = [sentences[idx] for idx in similar_indices[0]]
      idcx = similar_indices[0].tolist()
      similar_indices_int = [int(idx) for idx in idcx]
      return top_similar_sentences , distances[0].tolist(), similar_indices_int

      
  def find_most_similar_pair(self, sentences):
      """

    This finds the most similar pairs using cosine simularity.
    This uses the Faiss Library to optimize the speed and memory usage to find the most similar sentences
    For more information on the Faiss library, consult this : https://github.com/facebookresearch/faiss
    Faiss can return a number of the most similar sentences, not just a pair. 


        Args:
            sentence: A List of Strings that are the sentances you wish to find the similarity for.
            
            

        Returns:
            idx1, id2, score : the index's of the sentences, and the similarity score between them.
        Example:
            Example usage of encode_sentences:
            
            >>> model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca") #default size of batch is 10
            >>> result = model.find_most_similar_pair(sentences, 2)
            >>> print(result)
            "( لا يُتَوَضَّأُ مِنْ طَعَامٍ أَحَلَّ اللَّهُ أَكْلَهُ ,
   إِذَا تَوَضَّأَ الْعَبْدُ الْمُسْلِمُ فَغَسَلَ وَجْهَهُ خَرَجَ مِنْ وَجْهِهِ كُلُّ خَطِيئَةٍ نَظَرَ إِلَيْهَا بِعَيْنِهِ آخِرَ قَطْرِ الْمَاءِ ، فَإِذَا غَسَلَ يَدَيْهِ خَرَجَتْ مِنْهُمَا كُلُّ خَطِيئَةٍ بَطَشَهَا بِهِمَا ، ثُمَّ كَذَلِكَ حَتَّى يَخْرُجَ نَقِيًّا مِنَ الذُّنُوبِ ,
  0.61238515,
 )"
        """ 
      if not isinstance(sentences, list) or len(sentences) < 2:
        raise ValueError("Input must be a list of at least two sentences")

        # Preprocess and encode all sentences
      sentence_embeddings = self.preprocess_for_faiss(self.encode_sentences(sentences))

        # Calculate pairwise similarities
      index = faiss.IndexFlatIP(sentence_embeddings.shape[1])  # Use inner product (cosine similarity)
      faiss.normalize_L2(sentence_embeddings)

        # Add sentence embeddings to the index
      index.add(sentence_embeddings)

        # Find the two most similar sentences
      k = 2  # Number of similar sentences to retrieve
      D, I = index.search(sentence_embeddings, k)
      most_similar_pair = None
      max_similarity_score = -1
      for i in range(len(sentences)):
        for j in range(1, k):
            similar_idx = I[i][j]
            similarity_score = D[i][j]

            if similarity_score > max_similarity_score and i!=similar_idx:
                max_similarity_score = similarity_score
                most_similar_pair = (i, similar_idx, max_similarity_score)

      return most_similar_pair
  

