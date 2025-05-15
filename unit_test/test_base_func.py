import sys
import os
import numpy as np
import pytest

curr_dir = os.path.dirname(__file__)
e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
if e3SIM_dir not in sys.path:
	sys.path.insert(0, e3SIM_dir)

from base_func import *

def test_str2bool():
	assert(str2bool("YES")==True)
	assert(str2bool(1)==True)
	assert(str2bool("f")==False)
	assert(str2bool("yes  ")==True)


def test_check_ref_format():
	testfile = os.path.join(curr_dir, "test1.fasta")
	with open(testfile, "w") as file:
	    file.write(">Test123\n")
	    file.write("AACGATttgtACccgG\n")
	    file.write("AACGATttgtACccgG\n")
	assert(check_ref_format(testfile)==[0.25, 0.25, 0.25, 0.25])
	os.remove(testfile)



