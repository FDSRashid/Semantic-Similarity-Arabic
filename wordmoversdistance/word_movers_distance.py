import os
import subprocess
from semanticsimilarityarabic.semantic_similarity_arabic import SemanticSimilarityArabic
import numpy as np
import zipfile
import gensim
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
import warnings









class WordMoversDistance(SemanticSimilarityArabic):
    """
    WordMoversDistance class for semantic similarity computation using Word Movers Distance.
    Important note: this class only uses models from the AraVec library. Consult this for more
    information : https://github.com/bakrianoo/aravec. This is because theres literally only one large Word 2 vector model with arabic.
    There are 15 different models, based on n-grams and unigrams, based on wikipedia or twitter from the
    data, vector size of 300 or 100. Please go to the link above for the full details. 
    Important note : the unigram wikipedia model for 100 vectors doesnt work. i took it off the available models. 

    Args:
        model_name (str): The name of the Word2Vec model to load.
        batch_size (int): The batch size for preprocessing a list of sentences (default is 10).
        model_dir (string) : the location to put the models. if the directory path doesnt exist, it makes one.
    
    """
    AVAILABLE_MODELS = [
        'full_grams_cbow_100_twitter',
        'full_grams_cbow_300_twitter',
        'full_uni_sg_300_twitter',
        'full_uni_sg_100_twitter',
        'full_uni_cbow_300_wiki',
        'full_uni_cbow_100_wiki',
        'full_uni_sg_300_wiki',
        'full_grams_sg_300_twitter',
        'full_grams_sg_100_twitter',
        'full_grams_cbow_300_wiki',
        'full_grams_cbow_100_wiki',
        'full_grams_sg_300_wiki',
        'full_grams_sg_100_wiki',
        'full_uni_cbow_300_twitter',
        'full_uni_cbow_100_twitter'
    ]

    def __init__(self, model_name, batch_size = 10, model_dir="./model_gensim"):
        """
        Initializes the WordMoversDistance class.

        Args:
            model_name (str): The name of the Word2Vec model to load.
            batch_size (int): The batch size for preprocessing a list of sentences.
            model_dir (string) : the location to put the models
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model '{model_name}' is not available. Choose from: {', '.join(self.AVAILABLE_MODELS)}")
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, f"{model_name}.mdl")
        self.batch_size = batch_size
        
        
        
        # Check if the model file exists; if not, download and unzip it
        if not os.path.exists(self.model_path):
            self.download_model(model_name)

        
        
        # Load the model
        self.model = gensim.models.Word2Vec.load(self.model_path)
        self.word_vectors = self.model.wv
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax._src.xla_bridge")

    def download_model(self, model_name):
        """
        Downloads and unzips a Word2Vec model.

        Args:
            model_name (str): The name of the Word2Vec model to download.
        """
        # Define the URL and file names
        url = f"https://bakrianoo.ewr1.vultrobjects.com/aravec/{model_name}.zip"
        
        # Download the model zip file
        subprocess.run(["wget", url, "-P", self.model_dir])
        zip_file = os.path.abspath(os.path.join(self.model_dir, f"{model_name}.zip"))
        # Unzip the model
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.model_dir)
        
        # Remove the zip file
        os.remove(zip_file)

    def get_word_vector(self, word):
        if word in self.word_vectors:
            return self.word_vectors[word]
        else:
            return None
    
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
            
            >>> model = WordMoversDistance("full_grams_cbow_100_twitter") #default size of batch is 10
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
            
            >>> model = WordMoversDistance("full_grams_cbow_100_twitter") #default size of batch is 10
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
    def tokenize(self, sentence):
        """
        Tokenizes a sentence into words.

        Args:
            sentence (str): A string containing Arabic text to tokenize.

        Returns:
            tokens (list): A list of Arabic word tokens.
        """
        return simple_word_tokenize(sentence)
    def word_movers_distance(self, text1, text2):
        """
        Calculates the semantic similarity between two Arabic texts using Word Movers Distance (WMD).

        Args:
            text1 (str): The first Arabic text.
            text2 (str): The second Arabic text.

        Returns:
            similarity (float): The semantic similarity score between the two texts.
        """
        tokens1 = self.tokenize(self.preprocess(text1))  
        tokens2 = self.tokenize(self.preprocess(text2)) 

        # Calculate the WMD using WordEmbeddingsKeyedVectors
        try:
            similarity = self.word_vectors.wmdistance(tokens1, tokens2)
            return similarity
        except Exception as e:
            print(f"Error calculating WMD: {e}")
            return None
    
    def calculate_similarity_matrix(self, sentences):
        """
        Calculates the similarity matrix for a list of Arabic sentences.

        Args:
            sentences (list): A list of Arabic sentences for which to calculate the similarity matrix.

        Returns:
            similarity_matrix (np.ndarray): A matrix of semantic similarity scores between all pairs of sentences.
        """
        if not isinstance(sentences, list) or len(sentences) < 2:
            raise ValueError("Input must be a list of at least two sentences")
        num_sentences = len(sentences)
        similarity_matrix = np.zeros((num_sentences, num_sentences), dtype=float)

        for i in range(num_sentences):
            for j in range(i, num_sentences):
                similarity = self.word_movers_distance(sentences[i], sentences[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity

        return similarity_matrix
    
    def find_most_similar_pair(self, sentences):
        """
        Returns the most similar pair of texts among a list of input texts using a similarity matrix.

        Args:
            sentences (list): A list of Arabic sentences for which to find the most similar pair.

        Returns:
            most_similar_score (tuple): the score, and the sentenc indices
        """
        
        similarity_matrix = self.calculate_similarity_matrix(sentences)

        upper_triangular_matrix = np.triu(similarity_matrix, k=1)

        # Find the maximum similarity score and its indices
        max_similarity = np.max(upper_triangular_matrix)
        max_similarity_indices = np.unravel_index(np.argmax(upper_triangular_matrix), upper_triangular_matrix.shape)

        return max_similarity, max_similarity_indices
    
    def find_most_similar_sentence(self, sentences, sentence):
        if not isinstance(sentences, list) or len(sentences) < 2:
            raise ValueError("Input must be a list of at least two sentences")

       
        similarity_scores = [self.word_movers_distance(sentence, i) for i in sentences]
        most_similar_index = np.argmax(similarity_scores)
        most_similar_score = similarity_scores[most_similar_index]

        return (sentences[most_similar_index], most_similar_score, most_similar_index)
    
    def find_most_similar_pairs(self, sentences, n, tokenize = True):
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
            
            >>> model = WordMoversDistance("full_grams_cbow_100_twitter") #default size of batch is 10
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
      similarity_matrix = self.calculate_similarity_matrix(sentences, tokenize)
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
            
            >>> model = WordMoversDistance("full_grams_cbow_100_twitter") #default size of batch is 10
            >>> strings, scores, idxs = model.find_most_similar_sentences(list of sentences, sentence , 2)
            >>> print(len(scores))
            >>> 2
            "
        """
      if not isinstance(sentences, list) or len(sentences) < 2:
        raise ValueError("Input must be a list of at least two sentences")
      
      similarity_scores = [self.word_movers_distance(sentence, i) for i in sentences]
      top_n_indices = np.argpartition(similarity_scores, n)[-n:]
      top_n_scores = [similarity_scores[idx] for idx in top_n_indices]

    # Retrieve the top n most similar sentences
      top_n_sentences = [sentences[idx] for idx in top_n_indices]
      
      return top_n_sentences, top_n_scores, top_n_indices
    
    
