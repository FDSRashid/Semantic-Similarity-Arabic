import pytest
import torch
import numpy as np
from euclideandistance.euclidean_distance import EuclideanDistance  

# Define test cases for EuclideanDistance methods
class TestEuclideanDistance:
    @classmethod
    def setup_class(cls):
        # Initialize  EuclideanDistance instance for testing
        cls.euclidean_distance = EuclideanDistance("CAMeL-Lab/bert-base-arabic-camelbert-ca")

    def test_preprocess(self):
        # Test the preprocess method
        result = self.euclidean_distance.preprocess(" فَسَمِعَ رَجُلا ")
        assert isinstance(result, str)
        assert result == " فسمع رجلا "

    def test_encode_sentences(self):
        # Test the encode_sentences method
        sentences = ["Sentence 1", "Sentence 2"]
        embeddings = self.euclidean_distance.encode_sentences(sentences)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sentences)

    def test_find_most_similar_pair(self):
        # Test the find_most_similar_pair method
        sentences = ["Sentence 1", "Sentence 2", "Sentence 3"]
        result = self.euclidean_distance.find_most_similar_pair(sentences)
        assert isinstance(result, tuple)
        assert len(result) == 3
        

    

# Run the tests with pytest
if __name__ == "__main__":
    pytest.main()



