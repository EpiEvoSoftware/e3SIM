#!/bin/bash


# This is the command-line codes for the first tutorial of e3SIM.
# Please refer to Chapter 5.1 of the manual for more information of this example.

# Before executing the code, make sure to define your ${e3SIM} as environmental variable
# (`e3SIM-main/e3SIM_codes`)
# and set your working directory ${WKDIR} to where this `run.sh` resides.
# (`tutorial/vignettes/5.1_test_minimal_model`)
# Make sure the configuration file (test_config_minimal.json) is in your working directory.
# It is recommended to do `chmod -R +x run.sh` first to get execution access of this file.

e3SIM=YOURPATH_TO_E3SIM
WKDIR=YOUR_WORKING_DIRECTORY # WKDIR=${PWD}
RANDOMSEED=1100


# Run this command to update the absolute path in the provided configuration file to match your local directory
cd ${WKDIR}
python update_config.py


# NetworkGenerator
python ${e3SIM}/network_generator.py \
        -popsize 10000 \
        -wkdir ${WKDIR} \
        -method randomly_generate \
        -model ER \
        -p_ER 0.001 \
        -random_seed ${RANDOMSEED}


# SeedGenerator
python ${e3SIM}/seed_generator.py \
        -wkdir ${WKDIR} \
        -num_init_seq 5 \
        -method SLiM_burnin_WF \
        -Ne 1000 \
        -ref_path ${e3SIM}/../test_installation/data/TB/GCF_000195955.2_ASM19595v2_genomic.fna \
        -mu 1.1e-7 \
        -n_gen 4000 \
        -random_seed ${RANDOMSEED}


# GeneticEffectGenerator
python ${e3SIM}/genetic_effect_generator.py \
        -wkdir ${WKDIR} \
        -method randomly_generate \
        -num_init_seq 5 \
        -trait '{"transmissibility": 1, "drug_resistance": 0}' \
        -csv ${WKDIR}/candidate_regions.csv \
        -func n \
        -taus 1.5 \
        -calibration_link T \
        -Rs 2 \
        -random_seed ${RANDOMSEED}


# HostSeedMatcher
python ${e3SIM}/seed_host_matcher.py \
	-wkdir ${WKDIR} \
        -num_init_seq 5 \
	-method randomly_generate \
        -random_seed ${RANDOMSEED}


# OutbreakSimulator
python -u ${e3SIM}/outbreak_simulator.py \
        -config ${WKDIR}/test_config_minimal.json

