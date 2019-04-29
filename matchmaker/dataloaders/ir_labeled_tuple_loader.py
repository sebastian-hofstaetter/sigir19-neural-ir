# based on: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/seq2seq.py

from typing import Dict
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField,LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("ir_labeled_tuple_loader")
class IrLabeledTupleDatasetReader(DatasetReader):
    """
    Read a tsv file containing triple sequences, and create a dataset suitable for a
    neural IR model, or any model with a matching API.
    Expected format for each input line: <query_sequence_string>\t<pos_doc_sequence_string>\t<neg_doc_sequence_string>
    The output of ``read`` is a list of ``Instance`` s with the fields:
        query_tokens: ``TextField`` and
        doc_pos_tokens: ``TextField`` and
        doc_neg_tokens: ``TextField``
    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    source_add_start_token : bool, (optional, default=True)
        Whether or not to add `START_SYMBOL` to the beginning of the source sequence.
    """
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 max_doc_length:int = -1,
                 max_query_length:int = -1,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer() # little bit faster, useful for multicore proc. word_splitter=SimpleWordSplitter()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        self._source_add_start_token = source_add_start_token
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r", encoding="utf8") as data_file:
            #logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")

                if not line:
                    continue

                line_parts = line.split('\t')
                if len(line_parts) != 4:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (line, line_num + 1))
                query_id, doc_id, query_sequence, doc_sequence = line_parts
                yield self.text_to_instance(query_id, doc_id, query_sequence, doc_sequence)

    @overrides
    def text_to_instance(self, query_id:str, doc_id:str, query_sequence: str, doc_sequence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ

        query_id_field = LabelField(int(query_id), skip_indexing=True)
        doc_id_field = LabelField(int(doc_id), skip_indexing=True)

        query_tokenized = self._tokenizer.tokenize(query_sequence)
        #if self._source_add_start_token:
        #    query_tokenized.insert(0, Token(START_SYMBOL))
        #query_tokenized.append(Token(END_SYMBOL))
        if self.max_query_length > -1:
            query_tokenized = query_tokenized[:self.max_query_length]

        query_field = TextField(query_tokenized, self._token_indexers)
        
        doc_tokenized = self._tokenizer.tokenize(doc_sequence)
        #doc_tokenized.insert(0, Token(START_SYMBOL))
        #doc_tokenized.append(Token(END_SYMBOL))
        if self.max_doc_length > -1:
            doc_tokenized = doc_tokenized[:self.max_doc_length]

        doc_field = TextField(doc_tokenized, self._token_indexers)

        query_length = LabelField(len(query_tokenized), skip_indexing=True)
        doc_length = LabelField(len(doc_tokenized), skip_indexing=True)

        return Instance({
            "query_id":query_id_field,
            "doc_id":doc_id_field,
            "query_tokens":query_field,
            "doc_tokens":doc_field,
            "query_length":query_length,
            "doc_length":doc_length})
