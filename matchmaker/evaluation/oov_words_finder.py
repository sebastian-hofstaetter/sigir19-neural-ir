#
# generate oov stats & with/without oov query list, for given query + vocab files
# -------------------------------
#

import argparse
import os
import sys
sys.path.append(os.getcwd())
from collections import defaultdict
import statistics

from matchmaker.dataloaders.ir_tuple_loader import *
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.data.fields.text_field import Token
from allennlp.common import Params, Tqdm
Tqdm.default_mininterval = 1
from msmarco_eval import *

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file-oov', action='store', dest='out_file_oov',
                    help='queries with oov out', required=True)

parser.add_argument('--out-file-no-oov', action='store', dest='out_file_no_oov',
                    help='queries without oov out', required=True)

parser.add_argument('--vocab-dir', action='store', dest='vocab',
                    help='vocab in', required=True)

parser.add_argument('--query-tsv', action='store', dest='query',
                    help='original query in', required=True)

parser.add_argument('--qrel', action='store', dest='qrel',
                    help='qrel, to only check judged queries', required=False)


args = parser.parse_args()


#
# load data & create vocab
# -------------------------------
#  

loader = IrTupleDatasetReader(lazy=True, lowercase=True)
vocab = Vocabulary.from_files(args.vocab)
if args.qrel:
    qrels = load_reference(args.qrel)

not_judged = 0
oov_queries = 0
non_oov_queries = 0
oov_count_list = []
instances = loader.read(args.query)

with open(args.out_file_oov,"w",encoding="utf8") as out_file_oov:
    with open(args.out_file_no_oov,"w",encoding="utf8") as out_file_non_oov:

        for i in Tqdm.tqdm(instances):
            id_str = i["source_tokens"].tokens[0].text
            if args.qrel and int(id_str) not in qrels:
                not_judged += 1
                continue

            i.index_fields(vocab)

            indexes =  i["target_tokens"]._indexed_tokens["tokens"]

            if 1 in i["target_tokens"]._indexed_tokens["tokens"]:
                # we have a oov query
                oov_queries += 1
                oov_count_list.append(sum(1 for t in indexes if t == 1))

                out_file_oov.write(id_str + "\t" + " ".join(t.text for t in i["target_tokens"].tokens) +
                                  "\t("+  " ".join(t.text for indx, t in enumerate(i["target_tokens"].tokens) if indexes[indx] == 1 ) +")\n")
            else:
                non_oov_queries +=1
                oov_count_list.append(0)
                out_file_non_oov.write(id_str + "\n")

total = oov_queries + non_oov_queries
print("vocab size:",vocab.get_vocab_size())
print("emb params @ 300 dim size:",vocab.get_vocab_size() * 300)
print("---------------")

print("not judged (ignore for rest of stats)",not_judged)
print("total (only judged)",total)
print("---------------")
print("oov_queries",oov_queries,"(",oov_queries/total*100,"%)")
print("non_oov_queries",non_oov_queries,"(",non_oov_queries/total*100,"%)")

print("sum oov query terms",sum(oov_count_list))
print("avg oov query terms",statistics.mean(oov_count_list))
print("avg oov query terms (only oov)",statistics.mean(t for t in oov_count_list if t > 0))
