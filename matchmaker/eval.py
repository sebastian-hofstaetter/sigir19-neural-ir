import os
import copy
import time
import glob
from typing import Dict, Tuple, List

import torch

from allennlp.nn.util import move_to_device
from allennlp.common import Params, Tqdm

from evaluation.msmarco_eval import *
from multiprocess_input_pipeline import *

#
# run eval of a neural-ir model
# -------------------------------
#
# validate_model(...) = during training validation with parameter searches (saves all info in csv files)
# test_model(...) = after training do only inference on fixed params 

evaluate_cache={}

#
# raw model evaluation, returns model results as python dict, does not save anything / no metrics
#
def evaluate_model(model, config, _logger, cuda_device, eval_tsv, eval_batch_count, use_cache=False):

    model.eval()  # turning off training
    validation_results = {}
    fill_cache=False
    cached_batches = None

    try:
        if use_cache:
            global evaluate_cache
            if eval_tsv not in evaluate_cache:
                fill_cache=True
                evaluate_cache[eval_tsv] = []
            cached_batches = evaluate_cache[eval_tsv]
        
        if not use_cache or fill_cache:
            validation_queue, validation_processes, validation_exit = get_multiprocess_batch_queue("eval-batches",
                                                                                                   multiprocess_validation_loader,
                                                                                                   glob.glob(eval_tsv),
                                                                                                   config,
                                                                                                   _logger,
                                                                                                   queue_size=200)
            #time.sleep(len(validation_processes))  # fill the queue
            _logger.info("[eval_model] --- Start validation with queue.size:" + str(validation_queue.qsize()))
        else:
            _logger.info("[eval_model] --- Start validation with cache size:" + str(len(cached_batches)))

        with torch.no_grad():
            for i in Tqdm.tqdm(range(0, eval_batch_count), disable=config["tqdm_disabled"]):
                
                if not use_cache or fill_cache:
                    batch_orig = validation_queue.get()
                    if fill_cache:
                        cached_batches.append(batch_orig)
                else:
                    batch_orig = cached_batches[i]

                batch = move_to_device(copy.deepcopy(batch_orig), cuda_device)

                output = model.forward(batch["query_tokens"], batch["doc_tokens"], batch["query_length"], batch["doc_length"])
                output = output.cpu()  # get the output back to the cpu - in one piece

                for sample_i, sample_query_id in enumerate(batch_orig["query_id"]):  # operate on cpu memory

                    sample_query_id = int(sample_query_id)
                    sample_doc_id = int(batch_orig["doc_id"][sample_i])  # again operate on cpu memory

                    if sample_query_id not in validation_results:
                        validation_results[sample_query_id] = []

                    validation_results[sample_query_id].append((sample_doc_id, float(output[sample_i])))

                #if not use_cache or fill_cache and i % 100 == 0: # only to check for performance regresion
                #    if validation_queue.qsize() < 10:
                #        _logger.warning("validation_queue.qsize() < 10")

        if not use_cache or fill_cache:
            # make sure we didn't make a mistake in the configuration / data preparation
            if validation_queue.qsize() != 0:
                _logger.error("validation_queue.qsize() is not empty after evaluation")

            validation_exit.set()  # allow sub-processes to exit

    except BaseException as e:
        _logger.info('-' * 89)
        _logger.exception('[eval_model] Got exception: ')
        print("----- Attention! - something went wrong in eval_model (see logger) ----- ")
        
        if not use_cache or fill_cache:
            for proc in validation_processes:
                if proc.is_alive():
                    proc.terminate()
        raise e

    return validation_results

#
# validate a model during training + save results, metrics 
#
def validate_model(val_type, model, config, run_folder, _logger, cuda_device, epoch_number, batch_number=-1, global_best_info=None, candidate_set = None, use_cache=False):

    # this means we currently after a completed batch
    if batch_number == -1:
        evaluation_id = str(epoch_number)

    # this means we currently are in an inter-batch evaluation
    else:
        evaluation_id = str(epoch_number) + "-" +str(batch_number)
    
    if val_type == "end":
        evaluation_id = "end-" + evaluation_id
        config_prefix = "validation_end"
    else:
        config_prefix = "validation_cont"

    validation_results = evaluate_model(model,config,_logger,cuda_device,config[config_prefix+"_tsv"],config[config_prefix+"_batch_count"], use_cache)

    #
    # save sorted results
    #
    validation_file_path = os.path.join(run_folder, "validation-output-"+evaluation_id+".txt")
    lines = save_sorted_results(validation_results, validation_file_path)

    #
    # compute ir metrics (for ms_marco) and output them (to the logger + own csv file)
    # ---------------------------------
    #
    best_metric_info = {}
    best_metric_info["epoch"] = epoch_number
    best_metric_info["batch_number"] = batch_number
  
    #
    # do a cs@n over multiple n evaluation
    #
    if candidate_set != None:
        metrics = compute_metrics_with_cs_at_n(config[config_prefix+"_qrels"],validation_file_path,candidate_set,config[config_prefix+"_candidate_set_from_to"])

        # save mrr overview
        metric_file_path = os.path.join(run_folder, "validation-mrr-all.csv")
        save_one_metric_multiN(metric_file_path,metrics,
                               "MRR",range(config[config_prefix+"_candidate_set_from_to"][0],config[config_prefix+"_candidate_set_from_to"][1] + 1),
                               epoch_number,batch_number)

        # save all info + get best mrr
        best_mrr = 0
        for current_cs_n in range(config[config_prefix+"_candidate_set_from_to"][0],config[config_prefix+"_candidate_set_from_to"][1] + 1):
            metric_file_path = os.path.join(run_folder, "validation-metrics-cs_"+str(current_cs_n)+".csv")
            if val_type != "end":
                save_fullmetrics_oneN(metric_file_path,metrics[current_cs_n],epoch_number,batch_number)
            if metrics[current_cs_n]["MRR"] > best_mrr:
                best_mrr = metrics[current_cs_n]["MRR"]
                best_metric_info["metrics"] = metrics[current_cs_n]
                best_metric_info["cs@n"] = current_cs_n

        # save at the end all in one file
        if val_type == "end":
            save_fullmetrics_rangeN(os.path.join(run_folder, "validation-metrics-end.csv"),metrics,range(config[config_prefix+"_candidate_set_from_to"][0],config[config_prefix+"_candidate_set_from_to"][1] + 1))

    #
    # do a 1x evaluation over the full given validation set
    #
    else:
        metrics = compute_metrics_from_files(config[config_prefix+"_qrels"], validation_file_path)
        metric_file_path = os.path.join(run_folder, "validation-metrics.csv")
        save_fullmetrics_oneN(metric_file_path,metrics,epoch_number,batch_number)
        best_metric_info["metrics"] = metrics
        best_metric_info["cs@n"] = "-"
        best_mrr = metrics["MRR"]
    #
    # save best results
    #
    if config[config_prefix+"_save_only_best"] == True and global_best_info != None:
        os.remove(validation_file_path)
        if best_mrr > global_best_info["metrics"]["MRR"]:
            validation_file_path = os.path.join(run_folder, "best-validation-output.txt")
            save_sorted_results(validation_results, validation_file_path)

    #_logger.info('Validation for' + evaluation_id)
    #for metric in sorted(best_metric_info):
    #    _logger.info('{}: {}'.format(metric, best_metric_info[metric]))

    return best_metric_info, metrics, lines

#
# test a model after training + save results, metrics 
#
def test_model(model, config, run_folder, _logger, cuda_device, candidate_set = None, candidate_set_n = None):

    test_results = evaluate_model(model,config,_logger,cuda_device,config["test_tsv"],config["test_batch_count"])

    #
    # save sorted results
    #
    test_file_path = os.path.join(run_folder, "test-output.txt")
    save_sorted_results(test_results, test_file_path) 

    #
    # compute ir metrics (for ms_marco) and output them (to the logger + own csv file)
    # ---------------------------------
    #
    metrics = None
    if "test_qrels" in config:

        if candidate_set != None:
            metrics = compute_metrics_with_cs_at_n(config["test_qrels"], test_file_path, candidate_set, candidate_set_n)

        else:
            metrics = compute_metrics_from_files(config["test_qrels"], test_file_path)

        metric_file_path = os.path.join(run_folder, "test-metrics.csv")
        save_fullmetrics_oneN(metric_file_path, metrics, -1, -1)

    return metrics


def save_sorted_results(results, file, until_rank=-1):
    with open(file, "w") as val_file:
        lines = 0
        for query_id, query_data in results.items():
            
            # sort the results per query based on the output
            for rank_i, (doc_id, output_value) in enumerate(sorted(query_data, key=lambda x: x[1], reverse=True)):
                val_file.write("\t".join(str(x) for x in [query_id, doc_id, rank_i + 1, output_value])+"\n")
                lines += 1
                if until_rank > -1 and rank_i == until_rank + 1:
                    break
    return lines

def save_fullmetrics_oneN(file, metrics, epoch_number, batch_number):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("sep=,\nEpoch,After_Batch," + ",".join(k for k, v in metrics.items())+"\n")
    # append single row
    with open(file, "a") as metric_file:
        metric_file.write(str(epoch_number) + "," +str(batch_number) + "," + ",".join(str(v) for k, v in metrics.items())+"\n")

def save_fullmetrics_rangeN(file, metrics, m_range):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("sep=,\ncs@n," + ",".join(k for k, v in metrics[m_range.start].items())+"\n")
    # append single row
    with open(file, "a") as metric_file:
        for cs_n in m_range:
            metric_file.write(str(cs_n) + "," + ",".join(str(v) for k, v in metrics[cs_n].items())+"\n")


def save_best_info(file, best_info):
    with open(file, "w") as metric_file:
        metric_file.write("sep=,\nEpoch,batch_number,cs@n," + ",".join(k for k, v in best_info["metrics"].items())+"\n")
        metric_file.write(str(best_info["epoch"]) + "," +str(best_info["batch_number"])+ "," +str(best_info["cs@n"]) + "," + ",".join(str(v) for k, v in best_info["metrics"].items())+"\n")


def save_one_metric_multiN(file, metrics, selected_metric, over_range, epoch_number, batch_number):
    # write csv header once
    if not os.path.isfile(file):
        with open(file, "w") as metric_file:
            metric_file.write("sep=,\nEpoch,After_Batch," + ",".join(str(k) for k in over_range)+"\n")

    # append single row
    with open(file, "a") as metric_file:
        metric_file.write(str(epoch_number) + "," +str(batch_number) + "," + ",".join(str(v[selected_metric]) for cs_at_n, v in metrics.items())+"\n")
