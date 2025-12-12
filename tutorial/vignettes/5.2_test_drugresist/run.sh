#!/bin/bash


# This is the command-line codes for the second tutorial of e3SIM.
# Please refer to Chapter 5.2 of the manual for more information of this example.

# Before executing the code, make sure to define your ${e3SIM} as environmental variable
# (`e3SIM-main/e3SIM_codes`)
# and set your working directory ${WKDIR} to where this `run.sh` resides.
# (`tutorial/vignettes/5.2_test_drugresist`)
# and make sure the configuration file (test_config_drugresist.json) 
# and the provided genetic architecture (causal_gene_info.csv) are in your working directory.
# It is recommended to do `chmod -R +x run.sh` first to get execution access of this file.

e3SIM=YOURPATH_TO_E3SIM
WKDIR=YOUR_WORKING_DIRECTORY # WKDIR=${PWD}
RANDOMSEED=1100

# Run this command to update the absolute path in the provided configuration file to match your local directory
cd ${WKDIR}
python update_config.py


# NetworkGenerator
python ${e3SIM}/network_generator.py \
        -wkdir ${WKDIR} \
        -popsize 10000 \
        -method randomly_generate \
        -model BA \
        -m 2 \
        -random_seed ${RANDOMSEED}


# SeedGenerator is skipped in this example.


# GeneticEffectGenerator
python ${e3SIM}/genetic_effect_generator.py \
        -wkdir ${WKDIR} \
        -method user_input \
        -num_init_seq 1 \
        -csv ${WKDIR}/causal_gene_info.csv \
        -trait_n '{"transmissibility": 1, "drug_resistance": 2}' \
        -calibration F \
        -calibration_link T \
        -random_seed ${RANDOMSEED}


# HostSeedMatcher
python ${e3SIM}/seed_host_matcher.py \
	-wkdir ${WKDIR} \
	-method randomly_generate \
        -num_init_seq 1 \
        -match_scheme '{"0": "ranking"}' \
        -match_scheme_param '{"0": 5000}' \
        -random_seed ${RANDOMSEED}


# OutbreakSimulator
python -u ${e3SIM}/outbreak_simulator.py \
        -config ${WKDIR}/test_config_drugresist.json

