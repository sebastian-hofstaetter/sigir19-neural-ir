#
# train a neural-ir model
# -------------------------------
#
# features:
#
# * uses pytorch + allenNLP
# * tries to correctly encapsulate the data source (msmarco)
# * clear configuration with yaml files
#
# usage:
# python train.py --run-name experiment1 --config-file configs/knrm.yaml


import argparse
import copy
import os
import gc
import glob
import time
import sys
sys.path.append(os.getcwd())

# needs to be before torch import 
from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
prepare_environment(Params({}))

#import line_profiler
#import line_profiler_py35
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.multiprocessing as mp
import numpy

from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn.util import move_to_device
#from allennlp.training.trainer import Trainer

from utils import *
from models.knrm import KNRM
from models.knrm_ln import KNRM_LN
from models.conv_knrm import Conv_KNRM
from models.conv_knrm_ln import Conv_KNRM_LN
from models.conv_knrm_same_gram import Conv_KNRM_Same_Gram
from models.matchpyramid import MatchPyramid
from models.mv_lstm import MV_LSTM
from models.pacrr import PACRR

from matchmaker.modules.fasttext_token_embedder import *

from dataloaders.fasttext_token_indexer import *
from dataloaders.ir_triple_loader import *
from dataloaders.ir_labeled_tuple_loader import IrLabeledTupleDatasetReader
from evaluation.msmarco_eval import *
from typing import Dict, Tuple, List
from multiprocess_input_pipeline import *
from matchmaker.performance_monitor import * 
from matchmaker.eval import *

Tqdm.default_mininterval = 1

#
# main process
# -------------------------------
#
if __name__ == "__main__":

    #
    # config
    # -------------------------------
    #
    args = get_parser().parse_args()

    # the config object should only be used in this file, to keep an overview over the usage
    config = get_config(os.path.join(os.getcwd(), args.config_file), args.config_overwrites)
    run_folder = prepare_experiment(args, config)
    logger = get_logger_to_file(run_folder, "main")

    logger.info("Running: %s", str(sys.argv))
    # hardcode gpu usage
    cuda_device = int(args.cuda_device_id)
    perf_monitor = PerformanceMonitor()
    perf_monitor.start_block("startup")

    #torch.cuda.init() # just to make sure we can select a device
    #torch.cuda.set_device(cuda_device)
    #logger.info("Using cuda device id: %i - %s", cuda_device, torch.cuda.get_device_name(cuda_device))
    
    #
    # create (or load) model instance
    # -------------------------------
    #
    # * vocab (pre-built, to make the embedding matrix smaller, see generate_vocab.py)
    # * pre-trained embedding
    # * network
    # * optimizer & loss function
    #

    #
    # load candidate set for efficient cs@N validation 
    #
    validation_cont_candidate_set = None
    if "validation_cont_candidate_set_path" in config and "validation_cont_candidate_set_from_to" in config:
        validation_cont_candidate_set = parse_candidate_set(config["validation_cont_candidate_set_path"],config["validation_cont_candidate_set_from_to"][1])
    
    validation_end_candidate_set = None
    if "validation_end_candidate_set_path" in config and "validation_end_candidate_set_from_to" in config:
        validation_end_candidate_set = parse_candidate_set(config["validation_end_candidate_set_path"],config["validation_end_candidate_set_from_to"][1])

    test_candidate_set = None
    if "test_candidate_set_max" in config and "test_candidate_set_path" in config:
        test_candidate_set = parse_candidate_set(config["test_candidate_set_path"],config["test_candidate_set_max"])


    # embedding layer (use pre-trained, but make it trainable as well)
    if config["token_embedder_type"] == "embedding":
        vocab = Vocabulary.from_files(config["vocab_directory"])
        tokens_embedder = Embedding.from_params(vocab, Params({"pretrained_file": config["pre_trained_embedding"],
                                                              "embedding_dim": config["pre_trained_embedding_dim"],
                                                              "trainable": config["train_embedding"],
                                                              "padding_index":0,
                                                              "sparse":config["sparse_gradient_embedding"]}))
    elif config["token_embedder_type"] == "fasttext":
        vocab = None #FastTextVocab(config["fasttext_vocab_mapping"])
        tokens_embedder = FastTextEmbeddingBag(numpy.load(config["fasttext_weights"]),sparse=True)

    elif config["token_embedder_type"] == "elmo":
        vocab = None
        tokens_embedder = ElmoTokenEmbedder(config["elmo_options_file"],config["elmo_weights_file"])
    else:
        logger.error("token_embedder_type %s not known",config["token_embedder_type"])
        exit(1)

    word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

    if config["model"] == "knrm":
        model = KNRM(word_embedder, n_kernels=config["knrm_kernels"]).cuda(cuda_device)

    elif config["model"] == "conv_knrm":
        model = Conv_KNRM(word_embedder, n_grams=config["conv_knrm_ngrams"], n_kernels=config["conv_knrm_kernels"],conv_out_dim=config["conv_knrm_conv_out_dim"]).cuda(cuda_device)
    
    elif config["model"] == "match_pyramid":
        model = MatchPyramid(word_embedder, conv_output_size=config["match_pyramid_conv_output_size"], conv_kernel_size=config["match_pyramid_conv_kernel_size"],adaptive_pooling_size=config["match_pyramid_adaptive_pooling_size"]).cuda(cuda_device)
    
    else:
        logger.error("Model %s not known",config["model"])
        exit(1)

    logger.info('Model %s total parameters: %s', config["model"], sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)


    all_params = model.parameters()
    we_params = next(all_params) # we assume the word_embedding is the first thing defined per model !
    embedding_optimizer=None
    if config["train_embedding"]:
        if config["embedding_optimizer"] == "adam":
            embedding_optimizer = Adam([we_params], lr=config["embedding_optimizer_learning_rate"])

        elif config["embedding_optimizer"] == "sparse_adam":
            embedding_optimizer = SparseAdam([we_params], lr=config["embedding_optimizer_learning_rate"])

        elif config["embedding_optimizer"] == "sgd":
            embedding_optimizer = SGD([we_params], lr=config["embedding_optimizer_learning_rate"],momentum=config["embedding_optimizer_momentum"])

    if config["optimizer"] == "adam":
        optimizer = Adam(all_params, lr=config["optimizer_learning_rate"], weight_decay=config["optimizer_weight_decay"])

    elif config["optimizer"] == "sgd":
        optimizer = SGD(all_params, lr=config["optimizer_learning_rate"], momentum=0.5, weight_decay=config["optimizer_weight_decay"])

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="max",
                                     patience=config["learning_rate_scheduler_patience"],
                                     factor=config["learning_rate_scheduler_factor"],
                                     verbose=True)
    early_stopper = EarlyStopping(patience=config["early_stopping_patience"], mode="max")

    criterion = torch.nn.MarginRankingLoss(margin=1, reduction='elementwise_mean').cuda(cuda_device)

    loss_file_path = os.path.join(run_folder, "training-loss.csv")
    # write csv header once
    with open(loss_file_path, "w") as loss_file:
        loss_file.write("sep=,\nEpoch,After_Batch,Loss\n")
    
    # keep track of the best metric
    best_metric_info = {}
    best_metric_info["metrics"]={}
    best_metric_info["metrics"]["MRR"] = 0
    best_metric_info_file = os.path.join(run_folder, "best-info.csv")
    best_model_store_path = os.path.join(run_folder, "best-model.pytorch-state-dict")

    perf_monitor.stop_block("startup")

    #
    # training / saving / validation loop
    # -------------------------------
    #
    #@profile
    #def work():
    try:
        for epoch in range(0, int(config["epochs"])):
            if early_stopper.stop:
                break
            perf_monitor.start_block("train")
            perf_start_inst = 0
            #
            # data loading 
            # -------------------------------
            #
            training_queue, training_processes, train_exit = get_multiprocess_batch_queue("train-batches-" + str(epoch),
                                                                                          multiprocess_training_loader,
                                                                                          glob.glob(config.get("train_tsv")),
                                                                                          config,
                                                                                          logger)
            #time.sleep(len(training_processes))  # fill the queue
            logger.info("[Epoch "+str(epoch)+"] --- Start training with queue.size:" + str(training_queue.qsize()))

            #
            # vars we need for the training loop 
            # -------------------------------
            #
            model.train()  # only has an effect, if we use dropout & regularization layers in the model definition...
            loss_sum = torch.zeros(1).cuda(cuda_device)
            training_batch_size = int(config["batch_size_train"])
            # label is always set to 1 - indicating first input is pos (see criterion:MarginRankingLoss) + cache on gpu
            label = torch.ones(training_batch_size).cuda(cuda_device)

            # helper vars for quick checking if we should validate during the epoch
            validate_every_n_batches = config["validate_every_n_batches"]
            do_validate_every_n_batches = validate_every_n_batches > -1

            #s_pos = torch.cuda.Stream()
            #s_neg = torch.cuda.Stream()

            #
            # train loop 
            # -------------------------------
            #
            for i in Tqdm.tqdm(range(0, config["training_batch_count"]), disable=config["tqdm_disabled"]):

                batch = training_queue.get()

                current_batch_size = batch["query_tokens"]["tokens"].shape[0]

                batch = move_to_device(batch, cuda_device)

                optimizer.zero_grad()
                if embedding_optimizer:
                    embedding_optimizer.zero_grad()

                #with torch.cuda.stream(s_pos):
                output_pos = model.forward(batch["query_tokens"], batch["doc_pos_tokens"],batch["query_length"],batch["doc_pos_length"])
                #with torch.cuda.stream(s_neg):
                output_neg = model.forward(batch["query_tokens"], batch["doc_neg_tokens"],batch["query_length"],batch["doc_neg_length"])

                #torch.cuda.synchronize()

                # the last batches might (will) be smaller, so we need to check the batch size :(
                # but it should only affect the last n batches (n = # data loader processes) so we don't need a performant solution
                if current_batch_size != training_batch_size:
                    label = torch.ones(current_batch_size).cuda(cuda_device)

                loss = criterion(output_pos, output_neg, label)
                
                loss.backward()
                optimizer.step()
                if embedding_optimizer:
                    embedding_optimizer.step()
                
                # set the label back to a cached version (for next iterations)
                if current_batch_size != training_batch_size:
                    label = torch.ones(training_batch_size).cuda(cuda_device)

                loss_sum = loss_sum.data + loss.detach().data
                #loss = loss.detach()
                #del loss, output_neg, output_pos, batch
                #torch.cuda.synchronize() # only needed for profiling to get the remainder of the cuda work done and not put into another line

                if i > 0 and i % 100 == 0:
                    # append loss to loss file
                    with open(loss_file_path, "a") as loss_file:
                        loss_file.write(str(epoch) + "," +str(i) + "," + str(loss_sum.item()/100) +"\n")

                    # reset sum (on gpu)
                    loss_sum = torch.zeros(1).cuda(cuda_device)

                    # make sure that the perf of the queue is sustained
                    if training_queue.qsize() < 10:
                        logger.warning("training_queue.qsize() < 10")

                #
                # validation (inside epoch) - if so configured
                #
                if do_validate_every_n_batches:
                    if i > 0 and i % validate_every_n_batches == 0:
                        
                        perf_monitor.stop_block("train",(i - perf_start_inst) * training_batch_size)
                        perf_start_inst = i
                        perf_monitor.start_block("cont_val")

                        best_metric, _, validated_count = validate_model("cont",model, config, run_folder, logger, cuda_device, epoch, i,best_metric_info,validation_cont_candidate_set,use_cache=True)
                        if best_metric["metrics"]["MRR"] > best_metric_info["metrics"]["MRR"]:
                            best_metric_info = best_metric
                            save_best_info(best_metric_info_file,best_metric_info)
                            torch.save(model.state_dict(), best_model_store_path)

                        if config["model"] == "knrm" or config["model"] == "knrm_ln" or config["model"] == "conv_knrm" or config["model"] == "conv_knrm_same_gram":
                            #logger.info("KNRM-dense layer: b: %s , weight: %s",str(model.dense.bias.data), str(model.dense.weight.data))
                            logger.info("KNRM-dense layer: weight: %s",str(model.dense.weight.data))
                        if config["model"] in ["knrm_ln","conv_knrm_ln"]:
                            logger.info("KNRM_LN-length_norm_factor: weight: %s",str(model.length_norm_factor.data))

                        model.train()
                        lr_scheduler.step(best_metric["metrics"]["MRR"])
                        if early_stopper.step(best_metric["metrics"]["MRR"]):
                            logger.info("early stopping epoch %i batch count %i",epoch,i)
                            break

                        perf_monitor.stop_block("cont_val",validated_count)
                        perf_monitor.start_block("train")
                        perf_monitor.print_summary()

            # make sure we didn't make a mistake in the configuration / data preparation
            if training_queue.qsize() != 0:
                logger.error("training_queue.qsize() is not empty after epoch "+str(epoch))

            train_exit.set()  # allow sub-processes to exit
            perf_monitor.stop_block("train",i - perf_start_inst)

            #
            # validation (after epoch)
            #
            best_metric, _, validated_count = validate_model("cont",model, config, run_folder, logger, cuda_device, epoch,-1,best_metric_info,validation_cont_candidate_set,use_cache=True)
            if best_metric["metrics"]["MRR"] > best_metric_info["metrics"]["MRR"]:
                best_metric_info = best_metric
                save_best_info(best_metric_info_file,best_metric_info)
                torch.save(model.state_dict(), best_model_store_path)

            if config["model"] == "knrm" or config["model"] == "conv_knrm" or config["model"] == "conv_knrm_same_gram":
                #logger.info("KNRM-dense layer: b: %s , weight: %s",str(model.dense.bias.data), str(model.dense.weight.data))
                logger.info("KNRM-dense layer: weight: %s",str(model.dense.weight.data))

            lr_scheduler.step(best_metric["metrics"]["MRR"])
            if early_stopper.step(best_metric["metrics"]["MRR"]):
                logger.info("early stopping epoch %i",epoch)
                break

        #
        # evaluate the 2nd validation (larger) & test set with the best model
        #
        print("Mem allocated:",torch.cuda.memory_allocated())
        model_cpu = model.cpu() # we need this strange back and forth copy for models > 1/2 gpu memory, because load_state copies the state dict temporarily
        del model, loss, optimizer, all_params, we_params, lr_scheduler
        if embedding_optimizer:
            del embedding_optimizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(10) # just in case the gpu has not cleaned up the memory
        model_cpu.load_state_dict(torch.load(best_model_store_path,map_location="cpu"))
        model = model_cpu.cuda(cuda_device)


        if "validation_end_tsv" in config:
            best_metric, _, validated_count = validate_model("end",model, config, run_folder, logger, cuda_device, best_metric_info["epoch"], best_metric_info["batch_number"],best_metric_info,validation_end_candidate_set)
            if best_metric["metrics"]["MRR"] > best_metric_info["metrics"]["MRR"]:
                logger.info("got new best metric from end validation")
                best_metric_info = best_metric
                save_best_info(best_metric_info_file,best_metric_info)

        if "test_tsv" in config:
            test_result = test_model(model, config, run_folder, logger, cuda_device, test_candidate_set, best_metric_info["cs@n"])

        perf_monitor.save_summary(os.path.join(run_folder,"perf-monitor.txt"))

    except:
        logger.info('-' * 89)
        logger.exception('[train] Got exception: ')
        logger.info('Exiting from training early')
        print("----- Attention! - something went wrong in the train loop (see logger) ----- ")

        for proc in training_processes:
            if proc.is_alive():
                proc.terminate()
        exit(1)

    # make sure to exit processes (for early stopping)
    for proc in training_processes:
        if proc.is_alive():
            proc.terminate()
    exit(0)
    
    #prof = line_profiler.LineProfiler()
#
    #prof.add_function(work)
    #prof.runcall(work)
#
    #prof.dump_stats("line_prof_inline.prof")
#
    #prof.print_stats()