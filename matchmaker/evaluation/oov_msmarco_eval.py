import argparse
import os
import sys
sys.path.append(os.getcwd())
import glob

from matchmaker.evaluation.msmarco_eval import *
from matchmaker.utils import *
#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--bm25', action='store', dest='bm25_file',
                    help='bm25 output file', required=True)

parser.add_argument('--neural-output', action='store', dest='neural_file',
                    help='neural-output file', required=True)

parser.add_argument('--queries-to-check', action='store', dest='subset_file',
                    help='subset of queries to check', required=False)

parser.add_argument('--cs-n', action='store', dest='cs_at_n',type=int,
                    help='candidate set size (bm25)', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to only check judged queries', required=True)

args = parser.parse_args()

#
# work
#

qids_to_ranked_candidate_passages = load_candidate(args.neural_file)

if args.subset_file:
    select_queries = {}
    with open(args.subset_file,'r') as f:
        for line in f:
            select_queries[int(line.split("\t")[0])] = 1
    for q in list(qids_to_ranked_candidate_passages.keys()):
        if q not in select_queries:
            qids_to_ranked_candidate_passages.pop(q,None)

qids_to_relevant_passageids = load_reference(args.qrel)

candidate_set = parse_candidate_set(args.bm25_file, args.cs_at_n)

metrics = compute_metrics_with_cs_at_n_memory(qids_to_relevant_passageids, qids_to_ranked_candidate_passages,candidate_set,args.cs_at_n)

print(metrics)