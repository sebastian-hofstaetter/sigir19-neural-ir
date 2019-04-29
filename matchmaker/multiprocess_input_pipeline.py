import torch.multiprocessing as mp

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from dataloaders.fasttext_token_indexer import *
from dataloaders.ir_triple_loader import *
from dataloaders.ir_labeled_tuple_loader import IrLabeledTupleDatasetReader
from typing import Dict, Tuple, List

#
# Multiprocess input pipeline
# -------------------------------
#
# single epoch batch generators with multiple subprocesses, each subprocess works on its own file until the file is parsed completely
#
# - the processes have as little communication as possible (because it is prohibitly expensive in python)
# - the finished batches go into shared memory and then the queue to be picked up by the train/validaton loops
#

mp.get_logger().setLevel(logging.WARNING)  # ignore useless process start console logs
mp.set_sharing_strategy("file_system") # VERY MUCH needed for linux !! makes everything MUCH faster -> from 10 to 30+ batches/s

fasttext_vocab_cached_mapping = None
fasttext_vocab_cached_data = None

#
# process & queue starter, returns a queue which gets the batches put into ready to go into the model.forward pass
#
def get_multiprocess_batch_queue(name_prefix: str, target_function, files, conf, _logger, queue_size=100) -> Tuple[mp.Queue, List[mp.Process], mp.Event]:
    ctx = mp.get_context('spawn') # also set so that windows & linux behave the same 
    _queue = ctx.Queue(queue_size)
    _processes = []
    _finish_notification = ctx.Event()

    if len(files) == 0:
        _logger.error("No files for multiprocess loading specified, for: " + name_prefix)
        exit(1)
    else:
        _logger.info("Starting "+str(len(files))+" data loader processes, for:" + name_prefix)

    if conf["token_embedder_type"] == "fasttext":
        global fasttext_vocab_cached_mapping
        global fasttext_vocab_cached_data
        if fasttext_vocab_cached_data is None:
            fasttext_vocab_cached_mapping, fasttext_vocab_cached_data = FastTextVocab.load_ids(conf["fasttext_vocab_mapping"],conf["fasttext_max_subwords"])
            fasttext_vocab_cached_data.share_memory_()

    for proc_number, file in enumerate(files):
        process = ctx.Process(name=name_prefix + "-" + str(proc_number),
                             target=target_function,
                             args=(proc_number, conf, _queue, _finish_notification, file,fasttext_vocab_cached_mapping,fasttext_vocab_cached_data))
        process.start()
        _processes.append(process)
    return _queue, _processes, _finish_notification


#
# training instance generator
#   - filling the _queue with ready to run training batches
#   - everything is thread local
#
def multiprocess_training_loader(process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file,_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data):

    # workflow: we tokenize the data files with the costly spacy before training in a preprocessing step 
    # (and concat the tokens with single whitespaces), so here we only split on the whitepsaces
    _tokenizer = None
    if _config["preprocessed_tokenized"] == True:
        _tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())

    if _config["token_embedder_type"] == "embedding":
        _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        _vocab = Vocabulary.from_files(_config["vocab_directory"])

    elif _config["token_embedder_type"] == "fasttext":
        _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
        _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])

    elif _config["token_embedder_type"] == "elmo":
        _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
        _vocab = None

    _triple_loader = IrTripleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"])

    _iterator = BucketIterator(batch_size=int(_config["batch_size_train"]),
                               sorting_keys=[("doc_pos_tokens", "num_tokens"), ("doc_neg_tokens", "num_tokens")])

    _iterator.index_with(_vocab)

    for training_batch in _iterator(_triple_loader.read(_local_file), num_epochs=1):

        _queue.put(training_batch)  # this moves the tensors in to shared memory

    _queue.close()  # indicate this local thread is done
    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore

#
# validation instance generator
#   - filling the _queue with ready to run validation batches
#   - everything is defined thread local
#
def multiprocess_validation_loader(process_number: int, _config, _queue: mp.Queue, _wait_for_exit: mp.Event, _local_file,_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data):

    # workflow: we tokenize the data files with the costly spacy before training in a preprocessing step 
    # (and concat the tokens with single whitespaces), so here we only split on the whitepsaces
    _tokenizer = None
    if _config and _config["preprocessed_tokenized"] == True:
        _tokenizer = WordTokenizer(word_splitter=JustSpacesWordSplitter())

    if _config["token_embedder_type"] == "embedding":
        _token_indexers = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True)}
        _vocab = Vocabulary.from_files(_config["vocab_directory"])

    elif _config["token_embedder_type"] == "fasttext":
        _token_indexers = {"tokens": FastTextNGramIndexer(_config["fasttext_max_subwords"])}
        _vocab = FastTextVocab(_fasttext_vocab_cached_mapping,_fasttext_vocab_cached_data,_config["fasttext_max_subwords"])

    elif _config["token_embedder_type"] == "elmo":
        _token_indexers = {"tokens": ELMoTokenCharactersIndexer()}
        _vocab = None

    _tuple_loader = IrLabeledTupleDatasetReader(lazy=True, tokenizer=_tokenizer,token_indexers=_token_indexers, max_doc_length=_config["max_doc_length"],max_query_length=_config["max_query_length"])

    _iterator = BucketIterator(batch_size=int(_config["batch_size_eval"]),
                               sorting_keys=[("doc_tokens", "num_tokens"), ("query_tokens", "num_tokens")])

    _iterator.index_with(_vocab)

    for training_batch in _iterator(_tuple_loader.read(_local_file), num_epochs=1):

        _queue.put(training_batch)  # this moves the tensors in to shared memory

    _queue.close()  # indicate this local thread is done
    _wait_for_exit.wait()  # keep this process alive until all the shared memory is used and not needed anymore

