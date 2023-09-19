import pytest
import torch
import numpy as np
from cosinesimilarity.cosine_similarity import CosineSimilarity






def test_constructor_required():
    #test 3 different type of tokenizers
    model1 = CosineSimilarity('CAMeL-Lab/bert-base-arabic-camelbert-ca')
    model2 = CosineSimilarity('gpt2')
    model3 = CosineSimilarity('jonatasgrosman/wav2vec2-large-xlsr-53-arabic')
    assert model1 is not None and model2 is not None and model3 is not None

def test_constructor_default():
    #test 3 different type of tokenizers
    expected_model_args = {"num_labels": 2}
    expected_tokenizer_args = {"model_max_length": 512}
    
    with pytest.raises(ValueError):
        CosineSimilarity('CAMeL-Lab/bert-base-arabic-camelbert-ca', gpu = True,tokenizer_args=expected_tokenizer_args, 
                             model_args=expected_model_args)
    model = CosineSimilarity('CAMeL-Lab/bert-base-arabic-camelbert-ca',tokenizer_args=expected_tokenizer_args, 
                             model_args=expected_model_args)
    assert model.tokenizer.model_max_length == expected_tokenizer_args['model_max_length']
    assert model.model.config.num_labels == expected_model_args['num_labels']

def test_preprocess():
    model = CosineSimilarity('CAMeL-Lab/bert-base-arabic-camelbert-ca')
    output = model.preprocess("هَلْ ذَهَبْتَ إِلَى المَكْتَبَةِ؟")
    assert type(output) == str
    assert output == 'هل ذهبت الي المكتبه؟'


def test_encode():
    model = CosineSimilarity('CAMeL-Lab/bert-base-arabic-camelbert-ca')
    encoded = model.encode_sentences('هل ذهبت الي المكتبه؟')
    assert type(encoded) == list
    assert torch.is_tensor(encoded[0])

def test_preprocess_faiss():
    model = CosineSimilarity('CAMeL-Lab/bert-base-arabic-camelbert-ca')
    arr = model.preprocess_for_faiss(model.encode_sentences('هل ذهبت الي المكتبه؟'))
    assert type(arr) == np.ndarray
    assert arr.flags['C_CONTIGUOUS']


def test_valid_input_pairs():
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    sentences = ["This is sentence 1.", "This is sentence 2.", "This is sentence 3."]
    n = 2
    result = model.find_most_similar_pairs(sentences, n)
    assert isinstance(result, list)
    assert len(result) == n
    for pair in result:
        assert isinstance(pair, tuple)
        assert len(pair) == 3
        assert isinstance(pair[0], str)
        assert isinstance(pair[1], str)
        assert isinstance(pair[2], float)

def test_invalid_input_pairs():
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    # Test with less than 2 sentences
    with pytest.raises(ValueError):
        model.find_most_similar_pairs(["Only one sentence"], 2)
    
    # Test with invalid 'n' value
    with pytest.raises(ValueError):
        model.find_most_similar_pairs(["Sentence 1", "Sentence 2"], 0)
    
    # Test with non-list input
    with pytest.raises(ValueError):
        model.find_most_similar_pairs("Not a list", 2)


def test_valid_input_sentence():
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    sentences = ["This is sentence 1.", "This is sentence 2.", "This is sentence 3."]
    sentence = "This is a test sentence."
    result = model.find_most_similar_sentence(sentences, sentence)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], str)
    assert isinstance(result[1], float)
    assert isinstance(result[2], int)

def test_invalid_input_sentences():
    # Test with less than 2 sentences in the list
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    with pytest.raises(ValueError):
        model.find_most_similar_sentence(["Only one sentence"], "Test sentence")

    # Test with non-list input for sentences
    with pytest.raises(ValueError):
        model.find_most_similar_sentence("Not a list", "Test sentence")

def test_similarity_score_sentence():
    # Ensure the similarity score is within the expected range [0, 1]
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    sentences = ["A", "B", "C", "D"]
    sentence = "E"
    result = model.find_most_similar_sentence(sentences, sentence)
    assert 0 <= result[1] <= 1


def test_most_similar_index_sentence():
    # Ensure the most similar index is within the valid range
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    sentences = ["A", "B", "C", "D"]
    sentence = "E"
    result = model.find_most_similar_sentence(sentences, sentence)
    assert 0 <= result[2] < len(sentences)
    
def test_valid_input_sentences():
    sentences = ["This is sentence 1.", "This is sentence 2.", "This is sentence 3."]
    sentence = "This is a test sentence."
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    n = 2
    result = model.find_most_similar_sentences(sentences, sentence, n)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)
    assert isinstance(result[2], list)
    assert len(result[0]) == n
    assert len(result[1]) == n
    assert len(result[2]) == n
    assert all(isinstance(sent, str) for sent in result[0])
    assert all(isinstance(score, float) for score in result[1])
    assert all(isinstance(idx, int) for idx in result[2])

def test_invalid_input_sentences():
    # Test with less than 2 sentences in the list
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    with pytest.raises(ValueError):
        model.find_most_similar_sentences(["Only one sentence"], "Test sentence")

    # Test with non-list input for sentences
    with pytest.raises(ValueError):
        model.find_most_similar_sentences("Not a list", "Test sentence")

def test_n_greater_than_sentences():
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    # Test when n is greater than the number of sentences in the list
    sentences = ["A", "B"]
    sentence = "C"
    n = 3
    result = model.find_most_similar_sentences(sentences, sentence, n)
    assert len(result[0]) == len(result[1]) == len(result[2]) == 2  # Should return all sentences



def test_valid_input():
    sentences = ["This is sentence 1.", "This is sentence 2.", "This is sentence 3."]
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    result = model.find_most_similar_pair(sentences)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)
    assert isinstance(result[2], float)

def test_invalid_input():
    # Test with less than 2 sentences in the list
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    with pytest.raises(ValueError):
        model.find_most_similar_pair(["Only one sentence"])
    
    # Test with non-list input for sentences
    with pytest.raises(ValueError):
        model.find_most_similar_pair("Not a list")

def test_similarity_score():
    model = CosineSimilarity("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    # Ensure the similarity score is within the expected range [0, 1]
    sentences = ["A", "B", "C", "D"]
    result = model.find_most_similar_pair(sentences)
    assert 0 <= result[2] <= 1


