import sys
import os
import numpy as np
import pytest

from base_func import *

curr_dir = os.path.dirname(__file__)

def test_str2bool:
	assert(test_str2bool("YES")==True)
	assert(test_str2bool(1)==True)
	assert(test_str2bool("f")==False)
	assert(test_str2bool("yes  ")==True)


def test_check_ref_format:
	testfile = os.path.join(curr_dir, "test1.fasta")
	with open(testfile, "w") as file:
	    file.write(">Test123\n")
	    file.write("AACGATttgtACccgG\n")
	assert(check_ref_format(testfile)==[0.25, 0.25, 0.25, 0.25])
	os.remove(testfile)



