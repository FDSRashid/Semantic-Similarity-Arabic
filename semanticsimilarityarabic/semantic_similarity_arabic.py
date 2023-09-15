# -*- coding: utf-8 -*-
"""
# Author : Ferdaws Rashid
# email: frashid.berkeley.edu
# file semanticsimilarityarabic.py


"""

from abc import ABC, abstractmethod
#the interface of the Package

class SemanticSimilarityArabic(ABC):
    @abstractmethod
    def preprocess(self, sentences):
        pass

    @abstractmethod
    def calculate_similarity_matrix(self, sentences):
        pass

    @abstractmethod
    def find_most_similar_pairs(self, sentences, n):
        pass

    @abstractmethod
    def find_most_similar_sentence(self, sentences, target_sentence):
        pass
        
    @abtractmethod
    def find_most_similar_sentences(self, sentences, target_sentence, n = 2):
        pass

    @abstractmethod
    def find_most_similar_pair(self, sentences):
        pass

