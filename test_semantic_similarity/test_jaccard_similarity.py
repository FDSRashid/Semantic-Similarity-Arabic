import pytest
from jaccardsimilarity.jaccard_similarity  import JaccardSimilarity
import os

# Fixture to create an instance of JaccardSimilarity for testing
@pytest.fixture
def jaccard_similarity_instance():
    os.environ['CAMELTOOLS_DATA'] = 'placeholder_value'
    yield JaccardSimilarity('calima-msa-r13')
    del os.environ['CAMELTOOLS_DATA']

# Test the preprocess method
def test_preprocess(jaccard_similarity_instance):
    input_sentence = " فَسَمِعَ رَجُلا "
    expected_output = " فسمع رجلا "
    result = jaccard_similarity_instance.preprocess(input_sentence)
    assert result == expected_output

# Test the tokenize_sentence method
def test_tokenize_sentence(jaccard_similarity_instance):
    input_sentence = " فسمع رجلا "
    expected_output = ['ف+', 'سمع', 'رجلا']
    result = jaccard_similarity_instance.tokenize_sentence(input_sentence)
    assert result == expected_output


if __name__ == "__main__":
    pytest.main()
