#
# extract the input matrix from a fasttext bin model into a .npy numpy file (for faster loading )
# -------------------------------
#

import argparse
import os
import sys
from tqdm import tqdm
sys.path.append(os.getcwd())

from allennlp.common import  Tqdm
Tqdm.default_mininterval = 1

import fastText
import numpy

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--out-file', action='store', dest='out_file',
                    help='path to out file', required=True)

parser.add_argument('--fasttext-model', action='store', dest='model',
                    help='.bin model of fasttext', required=True)

args = parser.parse_args()


#
# work
# -------------------------------
# 
    
model = fastText.load_model(args.model)

in_matrix = model.get_input_matrix()
del model

print("shape:",in_matrix.shape)

z = numpy.zeros((1,in_matrix.shape[1]), dtype=numpy.float32)

in_matrix = numpy.append(z,in_matrix,axis=0)


print("shape (with padding):",in_matrix.shape)

numpy.save(args.out_file,in_matrix)
