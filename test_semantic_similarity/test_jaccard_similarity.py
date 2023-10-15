import pytest
from jaccardsimilarity.jaccard_similarity  import JaccardSimilarity
import os

# Fixture to create an instance of JaccardSimilarity for testing
@pytest.fixture
def jaccard_similarity_instance():
    
    yield JaccardSimilarity()
    

# Test the preprocess method
def test_preprocess(jaccard_similarity_instance):
    input_sentence = "أَنَا أُحِبّ تَعْلِم اللُغَات."
    expected_output = ['انا', 'حبا', 'تعلم', 'لغة', '.']
    result = jaccard_similarity_instance.preprocess(input_sentence)
    assert expected_output == result



if __name__ == "__main__":
    pytest.main()
