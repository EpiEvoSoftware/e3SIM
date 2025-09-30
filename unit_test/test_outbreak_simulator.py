# import sys
# import os
# import shutil
# import networkx as nx
# import numpy as np
# import pytest
# import json


# curr_dir = os.path.dirname(__file__)
# e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
# if e3SIM_dir not in sys.path:
# 	sys.path.insert(0, e3SIM_dir)
# from outbreak_simulator import *


# def test_sigmoid():
# 	if os.path.exists(os.path.join(curr_dir, 'test_minimal_model')):
# 		shutil.rmtree(os.path.join(curr_dir, 'test_minimal_model'))
# 	shutil.copytree(os.path.join(curr_dir, '../test/manual_tests/test_minimal_model'), os.path.join(curr_dir, 'test_minimal_model'))
# 	os.remove(os.path.join(curr_dir, 'test_minimal_model/slim.params'))

# 	config_test = {
# 	  "BasicRunConfiguration": {
# 	    "cwdir": os.path.join(curr_dir, 'test_minimal_model'),
# 	    "n_replicates": 1
# 	  },
# 	  "EvolutionModel": {
# 	    "subst_model_parameterization": "mut_rate",
# 	    "n_generation": 3650,
# 	    "mut_rate": 3.12e-10,
# 	    "within_host_reproduction": False,
# 	    "within_host_reproduction_rate": 0,
# 	    "cap_withinhost": 1
# 	  },
# 	  "SeedsConfiguration": {
# 	    "seed_size": 5,
# 	    "use_reference": False
# 	  },
# 	  "GenomeElement": {
# 	    "use_genetic_model": True,
# 	    "ref_path": os.path.join(curr_dir, '../test/data/TB/GCF_000195955.2_ASM19595v2_genomic.fna'),
# 	    "traits_num": {
# 	      "transmissibility": 1,
# 	      "drug_resistance": 0
# 	    },
# 	    "sigmoid_prob": True
# 	  },
# 	  "NetworkModelParameters": {
# 	    "host_size": 10000
# 	  },
# 	  "EpidemiologyModel": {
# 	    "model": "SEIR",
# 	    "epoch_changing": {
# 	      "n_epoch": 1,
# 	      "epoch_changing_generation": []
# 	    },
# 	    "genetic_architecture": {
# 	      "transmissibility": [1],
# 	      "cap_transmissibility": [10],
# 	      "drug_resistance": [0],
# 	      "cap_drugresist": [0]
# 	    },
# 	    "transition_prob": {
# 	      "S_IE_prob": [1.0],
# 	      "I_R_prob": [0.008],
# 	      "R_S_prob": [0.05],
# 	      "latency_prob": [1],
# 	      "E_I_prob": [0.01],
# 	      "I_E_prob": [0],
# 	      "E_R_prob": [0],
# 	      "sample_prob": [0.00001],
# 	      "recovery_prob_after_sampling": [0]
# 	    },
# 	    "massive_sampling": {
# 	      "event_num": 0,
# 	      "generation": [],
# 	      "sampling_prob": [],
# 	      "recovery_prob_after_sampling": []
# 	    },
# 	    "super_infection": False
# 	  },
# 	  "Postprocessing_options": {
# 	    "do_postprocess": True,
# 	    "tree_plotting": {
# 	      "branch_color_trait": 1,
# 	      "heatmap": "drug_resistance"
# 	    },
# 	    "sequence_output": {
# 	      "vcf": True,
# 	      "fasta": False
# 	    }
# 	  }
# 	}

# 	all_slim_simulation_by_config(config_test)
# 	with open(os.path.join(curr_dir, 'test_minimal_model/slim.params'), 'r') as file:
# 		for line in file:
# 			if line.startswith("sigmoid_prob"):
# 				assert(line == "sigmoid_prob:T\n")
# 				break

# 	shutil.rmtree(os.path.join(curr_dir, 'test_minimal_model'))
















