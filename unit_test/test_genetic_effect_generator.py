import sys
import os
import numpy as np
import pytest

curr_dir = os.path.dirname(__file__)
e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
if e3SIM_dir not in sys.path:
	sys.path.insert(0, e3SIM_dir)

from genetic_effect_generator import *
from genetic_effect_generator import _count_gff_genes


def test_count_gff_genes():
	test_val = _count_gff_genes(os.path.join(curr_dir, '../test/data/TB/GCF_000195955.2_ASM19595v2_genomic.overlap.gff'))
	assert(test_val==1212)


def test_read_effvals():
	test_dir = os.path.join(curr_dir, '../test/manual_tests/test_drugresist')
	eff_size_true = read_effvals(test_dir, os.path.join(test_dir, "causal_gene_info.csv"), [1,2], 5)
	print(eff_size_true)


print(_count_gff_genes(os.path.join(curr_dir, '../test/data/TB/GCF_000195955.2_ASM19595v2_genomic.overlap.gff')))

