

# -*- coding: utf-8 -*-
"""
    Author: Ferdaws Rashid
    email: frashid@berkeley.edu
"""
from semanticsimilarityarabic.semantic_similarity_arabic import SemanticSimilarityArabic
import numpy as np
import os


from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
import qalsadi.lemmatizer 
import arabicstopwords.arabicstopwords as stp




class JaccardSimilarity(SemanticSimilarityArabic):
  """
    A class for processing and comparing the Jaccard similarity of sentences using Arabic  Models.
    Note: All preproccessing is done for Arabic, so Please use only Arab Texts to use this model.

    Args:
        batch size (int) : the size of the batch you want to manage. 
    Dependencies:
      -transformers
      -Camel-tools
      -torch
      -numpy
      -sci-kit learn
      -qalsadi
      -Arabic-Stopwords

    Methods:
        preprocess(sentence: str) -> list[str]:
            Preprocesses a sentence before encoding.
            
        preprocess_batch(sentences: List[str]) -> List[str]:
          Preprocesses a  batch of sentences before encoding.


        jaccard_similarity(tokenized1: List[str], tokenized2: List[str]) -> float :
            finds the jaccard similarity of two tokenized sentences.

        find_most_similar_pair(sentences: List[str]) -> Tuple[str, str, float]:
            Finds the two most similar sentences from a list of sentences.

        find_n_similar_pair(sentences: List[str], n: int) -> List[Tuple[str, str, float]]:
            Finds the n most similar sentences from a list of sentences.

        find_most_similar_sentence(sentences: List[str], sentence: str) -> Tuple[str, float, int]:
            Finds the most similar sentence for a input sentence from a list of sentences.
            returns the sentence, the similarity score, and which index of the sentence it returns
            
        find_most_similar_sentences(sentences: List[str], sentence: str, n: int) -> Tuple[list[str], list[float], list[int]]:
            Finds the number n of the most similar sentence's for a input sentence from a list of sentences.
            returns a list of sentences, the similarity scores, and which number of the sentence's it returns

        calculate_similarity_matrix(sentences: List[str]) ->  np.nparray :
          calculates the similarity matrix of a list of sentences, doing pre-processing as well.
          to access the similarity score of the i'th and j'th sentences, find the matrix element at
          [i, j].

  """
  def __init__(self):
    try:
      self.lemmer = qalsadi.lemmatizer.Lemmatizer()
    except Exception as e:
      raise ValueError(f"Failed to initialize Class: {e}")
  def preprocess(self, sentence):
    """
        Preprocesses a sentence before encoding.
        First, normalize unicode, meaning all varients are a letter are returned to a canonical form. Then normalize alef, alef maksura, and tah marbuta, which
        is the same idea but for those letters.Then, vowel marks are removed. This is done becasue it decreases data sparsity. 
        For further information,
        consult https://camel-tools.readthedocs.io/en/latest/index.html
        Jaccard Similarity's preprocess function does more than the other preprocessing functions. It then tokenizes the sentence by splitting white space 
        and punctuation. then, it lemmatizes words of a sentence using qalsadi. finally, it removes stop words from the sentence list and returns the list of words. 

        Args:
            sentence: A string of Arabic Words you wish to pre-process for NLP
            

        Returns:
            sentence (list[string]), the preprocessed sentence as a list. 

        Example:
            Example usage of preprocess:
            
            >>> model = JaccardSimilarity() #default size of batch is 10
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
    sentence = simple_word_tokenize(sentence)
    sentence = [self.lemmer.lemmatize(i) for i in sentence]
    mask = [i for i in sentence if not stp.is_stop(i)]
    return mask


  def preprocess_batch(self, sentences):
    """
        Preprocesses a  a batch of sentences before encoding.
        First, normalize unicode, meaning all varients are a letter are returned to a canonical form. Then normalize alef, alef maksura, and tah marbuta, which
        is the same idea but for those letters.Then, vowel marks are removed. This is done becasue it decreases data sparsity. 
        For further information,
        consult https://camel-tools.readthedocs.io/en/latest/index.html
        Jaccard Similarity's preprocess function does more than the other preprocessing functions. It then tokenizes the sentence by splitting white space 
        and punctuation.
        then, it lemmatizes words of a sentence using qalsadi. 
        finally, it removes stop words from the sentence list and returns the list of words. 

        Args:
            sentences: A  List of type string of Arabic Sentences you wish to pre-process for NLP
            

        Returns:
            sentence (string), a string with proper formatting

        Example:
            Example usage of preprocess_batch:
            
            >>> model = JaccardSimilarity() #default size of batch is 10
            >>> result = model.preprocess(" فَسَمِعَ رَجُلا ")
            >>> print(result)
            " فسمع رجلا "
        """

    if not isinstance(sentences, list):
        raise ValueError("Input must be a list of sentences")

    preprocessed_sentences = [self.preprocess(i) for i in sentences]
    return preprocessed_sentences

  def jaccard_similarity(self, encoded_sentence1, encoded_sentence2):
     """

    This finds the Jaccard Similarity  of a two sentences. Jaccardian Similarity is defined by the size of the intersection of two sets divided by the size
    of the union of two sets.The closer the score is to 1, the more similar the sentence.
     

        Args:
            sentence: a string of the first sentence
            sentance2 : a string of the second sentence
            

        Returns:
            similarity : float of the jaccard similarity

        Example:
            Example usage of calculate_similarity_matrix:
            
            >>> model = JaccardSimilarity() #default size of batch is 10
            >>> result = model.jaccard_similarity(sentance1, sentence2)
            >>> print(result) 
            .4567

            
        """ 
     set1 = set(encoded_sentence1)
     set2 = set(encoded_sentence2)
     #make sets from the tokenized sentences

     intersection = len(set1.intersection(set2))
     union = len(set1.union(set2))
     similarity = intersection / union
     return similarity



  def calculate_similarity_matrix(self, sentences):
    """

    This finds the Jaccard Similarity matrix of a list of sentences. Jaccardian Similarity is defined by the size of the intersection of two sets divided by the size
    of the union of two sets.The closer the score is to 1, the more similar the sentence
    To find the similarity between the i'th and j'th sentence, select the matrix at element (i, j). Diagonal elements are the same sentences,
    so will always have 1.   

        Args:
            sentence: A List of Strings that are the sentances you wish to find the similarity for.
            tokenized: either True or false, wether you want to tokenize sentences or not.
            

        Returns:
            sentence numpy.nparray : A numpy array representing the similarity matrix. 

        Example:
            Example usage of calculate_similarity_matrix:
            
            >>> model = JaccardSimilarity() #default size of batch is 10
            >>> result = model.calculate_similarity_matrix([a list of sentences with length 3])
            >>> print(result) #just a example matrix
            [[ 0.          5.19615242 10.39230485]
            [ 5.19615242  0.          5.19615242]
            [10.39230485  5.19615242  0.        ]]

            
        """ 
    if not isinstance(sentences, list) or len(sentences) < 2:
        raise ValueError("Input must be a list of at least two sentences")

        # Preprocess and encode all sentences
    num_sentences = len(sentences)
    sentences = self.preprocess_batch(sentences)
    similarity_matrix = np.zeros((num_sentences, num_sentences))
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i == j:
                similarity_matrix[i][j] = 1.0  # Similarity of a sentence with itself is 1
            else:
                similarity_matrix[i][j] = self.jaccard_similarity(sentences[i], sentences[j])
    return similarity_matrix


  def find_most_similar_pairs(self, sentences, n):
      """

    This finds a specified number of the most similar pairs using Jaccardian Distance.
    This uses the similarity matrix from calculate_similarity_matrix() . It flattens the upper trangular part of the matrix, 
    then sorts in descending order and gets the number of sentences required. since its making a similarity matrix, i honestly dont
    recomend for large data sets. i kinda got lazy here. 


        Args:
            sentence: A List of Strings that are the sentances you wish to find the similarity for.
            n : the number of pairs to return. so, n = 3 would return the 3 most similar sentance pairs
            tokenized: either True or false, wether you want to tokenize sentences or not.
            

        Returns:
           list[Tuple[string,string,float]] : a list of the sentences, and the similarity score between them.
              
            

        Example:
            Example usage of find_most_similar_pairs:
            
            >>> model = JaccardSimilarity() #default size of batch is 10
            >>> result = model.find_most_similar_pairs(sentences, 2)
            >>> print(result)
            "[( لا يُتَوَضَّأُ مِنْ طَعَامٍ أَحَلَّ اللَّهُ أَكْلَهُ ,
   إِذَا تَوَضَّأَ الْعَبْدُ الْمُسْلِمُ فَغَسَلَ وَجْهَهُ خَرَجَ مِنْ وَجْهِهِ كُلُّ خَطِيئَةٍ نَظَرَ إِلَيْهَا بِعَيْنِهِ آخِرَ قَطْرِ الْمَاءِ ، فَإِذَا غَسَلَ يَدَيْهِ خَرَجَتْ مِنْهُمَا كُلُّ خَطِيئَةٍ بَطَشَهَا بِهِمَا ، ثُمَّ كَذَلِكَ حَتَّى يَخْرُجَ نَقِيًّا مِنَ الذُّنُوبِ ,
  0.61238515),
 ( الْغِيبَةُ تُفْطِرُ الصَّائِمَ وَتَنْقُضُ الْوُضُوءَ ,الْوُضُوءُ مِنَ الْمَذْيِ ، وَالْغُسْلُ مِنَ الْمَنِيِّ ,
  0.8229922)]"
        """ 
      if not isinstance(sentences, list) or len(sentences) < 2:
        raise ValueError("Input must be a list of at least two sentences")

        # Preprocess and encode all sentences
      similarity_matrix = self.calculate_similarity_matrix(sentences)
      num_sentences = len(sentences)

        # Flatten the upper triangular part of the matrix (excluding the diagonal)
      flat_similarity_scores = similarity_matrix[np.triu_indices(num_sentences, k=1)] 
      

      # Sort the similarity scores in descending order and get the indices
      sorted_indices = np.argsort(flat_similarity_scores)[::-1]
      top_n_indices = sorted_indices[:n]


      top_n_scores = [flat_similarity_scores[idx] for idx in top_n_indices]
      top_n_pairs = []
      for idx in top_n_indices:
         i, j = np.unravel_index(idx, (num_sentences, num_sentences))
         top_n_pairs.append((sentences[i], sentences[j], top_n_scores[idx]))

      
      return top_n_pairs

  def find_most_similar_sentence(self, sentences, sentence):

    """

    This finds a the most similar sentence from a list of sentances. You input a sentance to co
    compare the list to. It returns the sentence, the score, and the index of the sentence. 
    That way you can keep track of which element it came from.



        Args:
            sentences: A List of Strings that are the sentances you wish to find the similarity for.
            sentence: a single string that is the sentance you want to compare the list to.
            tokenized: either True or false, wether you want to tokenize sentences or not.
            

        Returns:
            most_similar_sentence, similarity_score, most_similar_index: string of the sentence, a float of the similarity score, and index as a integer

        Example:
            Example usage of find_most_similar_sentence:
            
            >>> model = JaccardSimilarity() #default size of batch is 10
            >>> result = model.find_most_similar_sentence(sentence, list of sentences)
            >>> print(result)
            " فَسَمِعَ رَجُلا يَقْرَأُ : /4 قُلْ هُوَ اللَّهُ أَحَدٌ سورة الإخلاص آية 1 /4 ، إِلَى آخِرِهَا ، فَقَالَ رَسُولُ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ : "" وَجَبَتْ "" ، فَقُلْتُ : مَاذَا يَا رَسُولَ اللَّهِ ؟ ، فَقَالَ : "" الْجَنَّةُ "" ، قَالَ أَبُو هُرَيْرَةَ : فَأَرَدْتُ أَنْ أَذْهَبَ إِلَى الرَّجُلِ فَأُبَشِّرَهُ ، ثُمَّ خِفْتُ أَنْ يَفُوتَنِي الْغَدَاءُ مَعَ رَسُولِ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ فَآثَرْتُ الْغَدَاءَ مَعَ رَسُولِ اللَّهِ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ ، ثُمَّ ذَهَبْتُ إِلَى الرَّجُلِ فَوَجَدْتُهُ قَدْ ذَهَبَ ,
1.0,
 0)"
        """ 
    if not isinstance(sentences, list) or len(sentences) < 2:
      raise ValueError("Input must be a list of at least two sentences")

        # Preprocess and encode all sentences
    sentence = self.preprocess(sentence)
    sentences1 = self.preprocess_batch(sentences)
    # Calculate Jaccard similarity between input sentence and all sentences in the list
    similarity_scores = [self.jaccard_similarity(sentence, i) for i in sentences1]
    most_similar_index = np.argmax(similarity_scores)
    most_similar_score = similarity_scores[most_similar_index]

    return (sentences[most_similar_index], most_similar_score, most_similar_index)



  def find_most_similar_sentences(self, sentences, sentence, n = 2):
      """

    This finds  the most similasr sentence from a list of sentances. You input a sentance to co
    compare the list to. It returns list , each elemenet has the sentence, the score, and the index of the sentence. 
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
            
            >>> model = JaccardSimilarity() #default size of batch is 10
            >>> strings, scores, idxs = model.find_most_similar_sentences(list of sentences, sentence , 2)
            >>> print(len(scores))
            >>> 2
            "
        """
      if len(sentences) < 2:
          raise ValueError('List of Sentences needs to be at least 2!')
      sentence = self.preprocess(sentence)
      sentences1 = self.preprocess_batch(sentences)
      similarity_scores = [self.jaccard_similarity(sentence, i) for i in sentences1]
      top_n_indices = np.argpartition(similarity_scores, n)[-n:]
      top_n_scores = [similarity_scores[idx] for idx in top_n_indices]

    # Retrieve the top n most similar sentences
      top_n_sentences = [sentences[idx] for idx in top_n_indices]
      
      return top_n_sentences, top_n_scores, top_n_indices

    
  def find_most_similar_pair(self, sentences):
      """

    This finds the most similar pairs using Jaccardian Similarity.
    Ngl i got really lazy i just called the other function and set n = 1.  


        Args:
            sentence: A List of Strings that are the sentances you wish to find the similarity for.
            tokenized: either True or false, wether you want to tokenize sentences or not.
            
            

        Returns:
            sentence tuple[str, str, float] : returns the two sentences and the similarity between them 

        Example:
            Example usage of encode_sentences:
            
            >>> model = JaccardSimilarity() #default size of batch is 10
            >>> result = model.find_most_similar_pair(sentences)
            >>> print(result)
            "( لا يُتَوَضَّأُ مِنْ طَعَامٍ أَحَلَّ اللَّهُ أَكْلَهُ ,
   إِذَا تَوَضَّأَ الْعَبْدُ الْمُسْلِمُ فَغَسَلَ وَجْهَهُ خَرَجَ مِنْ وَجْهِهِ كُلُّ خَطِيئَةٍ نَظَرَ إِلَيْهَا بِعَيْنِهِ آخِرَ قَطْرِ الْمَاءِ ، فَإِذَا غَسَلَ يَدَيْهِ خَرَجَتْ مِنْهُمَا كُلُّ خَطِيئَةٍ بَطَشَهَا بِهِمَا ، ثُمَّ كَذَلِكَ حَتَّى يَخْرُجَ نَقِيًّا مِنَ الذُّنُوبِ ,
  0.61238515,
 )"
        """ 
      
      sim_sentance = self.find_most_similar_pairs(sentences, 1)
      return sim_sentance[0]

