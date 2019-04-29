from typing import Dict, List

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.vocabulary import Vocabulary
import numpy as np
import torch

class FastTextVocab():

    def __init__(self, mapping, shared_tensor,max_subwords) -> None:
        self.mapping = mapping
        self.data = shared_tensor
        self.max_subwords = max_subwords
        self.default = torch.zeros(max_subwords, dtype=torch.int)

    def get_subword_ids(self, word):
        #print(word)
        if word not in self.mapping:
            return self.default

        #print(self.mapping[word],self.data[self.mapping[word]])

        return self.data[self.mapping[word]]

    @staticmethod
    def load_ids(file,max_subwords):
        mapping = {}
        data = []
        with open(file,"r",encoding="utf8") as in_file:
            for i,l in enumerate(in_file):
                l = l.rstrip().split()
                
                ids = [0] * max_subwords
                for k, val in enumerate(l[1:max_subwords]):
                    ids[k] = int(val)

                mapping[l[0]] = i
                data.append(ids)

        return mapping, torch.IntTensor(data)


class FastTextNGramIndexer(TokenIndexer[List[int]]):
    """
    Convert a token to an array of n-gram wordpiece ids to compute FastText representations.

    Parameters
    ----------
    namespace : ``str``, optional (default=``fasttext_grams``)
    """
    

    def __init__(self,
                 max_subwords,
                 namespace: str = 'fasttext_grams') -> None:
        self._namespace = namespace
        self.def_padding = torch.zeros(max_subwords, dtype=torch.int)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: FastTextVocab,
                          index_name: str) -> Dict[str, List[List[int]]]:
        # pylint: disable=unused-argument
        texts = [token.text.lower() for token in tokens]

        if any(text is None for text in texts):
            raise ConfigurationError('FastTextNGramIndexer needs a tokenizer '
                                     'that retains text')
        return {index_name: [vocabulary.get_subword_ids(text) for text in texts]}

    @overrides
    def get_padding_lengths(self, token: List[int]) -> Dict[str, int]:
        # pylint: disable=unused-argument
        return {}

    @overrides
    def get_padding_token(self) -> List[int]:
        return []

    def _default_value_for_padding(self):
        return self.def_padding

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[List[int]]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[List[int]]]:
        # pylint: disable=unused-argument
        return {key: torch.stack(pad_sequence_to_length(val, desired_num_tokens[key],
                                                        default_value=self._default_value_for_padding)).long()
                for key, val in tokens.items()}
