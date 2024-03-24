import os, shutil, subprocess, argparse
# import tskit, pyslim
from base_func import *
from post_simulation_func import *
from error_handling import CustomizedError

def _writebinary(v):
	"""
	Convert str/int/float value to binary string representation.

	Parameters:
		v (str): String value to be converted.

	Returns:
		str: Binary string representation ('T' for True, '' for False).
	"""
	return "T" if v else ""

def _check_integer(value, name):
	if not isinstance(value, int) or value < 1:
		raise CustomizedError(f"{name} has to be a positive integer.")
	
def _check_float(value, name):
	if not isinstance(value, float):
		raise CustomizedError(f"{name} has to be a float number.")

def _check_boolean(value, name):
	if not isinstance(value, bool):
		raise CustomizedError(f"{name} has to be a boolean value.")
	
def append_files(file1_path, file2_path):
	"""
	Append contents of file1 to file2.

	Parameters:
		file1_path (str): Path to the source file to be appended.
		file2_path (str): Path to the target file where contents will be appended.
	"""
	with open(file1_path, 'r') as file1:
		with open(file2_path, 'a') as file2:
			shutil.copyfileobj(file1, file2)

def create_slim_config(all_config):
	"""
	Create a SLiM-readable config file that includes the necessary parameters from the big config file.

	Parameters:
		all_config (dict): Dictionary of the big config file, read by read_config().

	Returns:
		dict: Dictionary containing all the parameters for SLiM.
	"""
	slim_pars = {}
	post_processing_config = {}
	print("********************************************************************")
	print("                    CHECKING THE CONFIGURATION")
	print("********************************************************************")
	cwdir = all_config["BasicRunConfiguration"]["cwdir"]
	slim_pars["cwdir"] = cwdir

	print("Checking \"BasicRunConfiguration\"...... ")
	if not os.path.exists(cwdir):
		raise CustomizedError(f"The working directory specified {cwdir} doesn't exist.")
	
	out_config = open(os.path.join(cwdir, "slim.params"), "w")
	out_config.write("cwdir:" + cwdir + "\n")

	slim_pars["n_replicates"] = all_config["BasicRunConfiguration"]["n_replicates"]
	_check_integer(slim_pars["n_replicates"], "Number of replicates")
	out_config.write(f"n_replicates:{slim_pars['n_replicates']}\n")
	print("\"BasicRunConfiguration\" checked.")

	print("Checking \"EvolutionModel\"...... ")
	slim_pars["n_generation"] = all_config["EvolutionModel"]["n_generation"]
	_check_integer(slim_pars["n_generation"], "Number of ticks")
	out_config.write(f"n_generation:{slim_pars['n_generation']}\n")

	slim_pars["mut_rate"] = all_config["EvolutionModel"]["mut_rate"]
	try:
		_check_integer(slim_pars["mut_rate"], "Mutation rate")
	except CustomizedError:
		_check_float(slim_pars["mut_rate"], "Mutation rate")
	out_config.write(f"mut_rate:{slim_pars['mut_rate']}\n")

	slim_pars["trans_type"] = all_config["EvolutionModel"]["trans_type"]
	if slim_pars["trans_type"] not in ["additive", "bialleleic"]:
		raise CustomizedError("Model for transmissibility has to be one of the two models: additive / bialleleic.")
	out_config.write(f"trans_type:{slim_pars['trans_type']}\n")

	slim_pars["dr_type"] = all_config["EvolutionModel"]["dr_type"]
	if slim_pars["dr_type"] not in ["additive", "bialleleic"]:
		raise CustomizedError("Model for drug-resistance has to be one of the two models: additive / bialleleic.")
	out_config.write(f"dr_type:{slim_pars['dr_type']}\n")

	slim_pars["within_host_reproduction"] = all_config["EvolutionModel"]["within_host_reproduction"]
	_check_boolean(slim_pars["within_host_reproduction"], "Within-host reproduction")
	out_config.write(f"within_host_reproduction:{_writebinary(slim_pars['within_host_reproduction'])}\n")

	slim_pars["cap_withinhost"] = all_config["EvolutionModel"]["cap_withinhost"]
	_check_integer(slim_pars["cap_withinhost"], "Capacity within host")
	if slim_pars["within_host_reproduction"]:
		if slim_pars["cap_withinhost"] < 1:
			raise CustomizedError("Capacity within host has to be at least 1")
		elif slim_pars["cap_withinhost"] == 1:
			print("WARNING: Within-host reproduction is turned on, but the capacity within-host is 1, which will be no different from within-host evolution being turned off.")

		slim_pars["within_host_reproduction_rate"] = all_config["EvolutionModel"]["within_host_reproduction_rate"]
		_check_float(slim_pars["within_host_reproduction_rate"], "Within-host reproduction probability")
		if not 0 < slim_pars["within_host_reproduction_rate"] <= 1:
			raise CustomizedError("Within-host reproduction probability should be between 0 and 1.")
		out_config.write(f"within_host_reproduction_rate:{slim_pars['within_host_reproduction_rate']}\n")
	out_config.write(f"cap_withinhost:{slim_pars["cap_withinhost"]}\n")
	print("\"EvolutionModel\" Checked. ")
	
	print("Checking \"SeedsConfiguration\"...... ")
	slim_pars["seed_size"] = all_config["SeedsConfiguration"]["seed_size"]
	_check_integer(slim_pars["seed_size"], "Number of seeds")
	out_config.write(f"seed_size:{slim_pars["seed_size"]}\n")

	slim_pars["use_reference"] = all_config["SeedsConfiguration"]["use_reference"]
	_check_boolean(slim_pars["use_reference"], "Whether to use reference genome for seeds")
	out_config.write(f"use_reference:{_writebinary(slim_pars["use_reference"])}\n")

	if not slim_pars["use_reference"]:
		if not os.path.exists(os.path.join(slim_pars["cwdir"], "originalvcfs")):
			raise CustomizedError("Reference genome isn't used for seed sequences, but SeedGenerator hasn't been run. Please run SeedGenerator before running this program or specify use_reference=true.")
	if not os.path.exists(os.path.join(slim_pars["cwdir"], "seed_host_match.csv")):
		raise CustomizedError("HostSeedMatcher hasn't been run. Please run HostSeedMatcher before running this program.")
	else:
		slim_pars["seed_host_matching_path"] = os.path.join(slim_pars["cwdir"], "seed_host_match.csv")
		out_config.write(f"seed_host_matching_path:{slim_pars["seed_host_matching_path"]}\n")
	print("\"SeedsConfiguration\" Checked. ")

	print("Checking \"GenomeElement\"...... ")
	slim_pars["ref_path"] = all_config["GenomeElement"]["ref_path"]
	if not os.path.exists(slim_pars["ref_path"]):
		raise CustomizedError(f"The provided reference genome path {slim_pars["ref_path"]} doesn't exist.")
	out_config.write(f"ref_path:{slim_pars["ref_path"]}\n")

	slim_pars["use_genetic_model"] = all_config["GenomeElement"]["use_genetic_model"]
	_check_boolean(slim_pars["use_genetic_model"], "Whether to use genetic model")
	if slim_pars["use_genetic_model"]:
		if not os.path.exists(os.path.join(slim_pars["cwdir"], "causal_gene_info.csv")):
			raise CustomizedError("Genetic model for trait is used, but GeneticElementGenerator hasn't been run. Please run GeneticElementGenerator before running this program.")
		else:
			out_config.write("causal_gene_path:" + os.path.join(slim_pars["cwdir"], "causal_gene_info.csv") + "\n")
	else:
		out_config.write("causal_gene_path:" + "\n")
	out_config.write(f"use_genetic_model:{_writebinary(slim_pars["use_genetic_model"])}\n")

	if not isinstance(all_config["GenomeElement"]["traits_num"], list) or len(all_config["GenomeElement"]["traits_num"]) != 2 or any(not isinstance(i, int) for i in all_config["GenomeElement"]["traits_num"]):
		raise CustomizedError("Number of traits (\"traits_num\") has to be a list ([]) of length 2, containing integers, showing number of effect size sets (traits) for transmissibility and drug-resistance.")
	print("\"GenomeElement\" Checked.")

	print("Checking \"NetworkModelParameters\"...... ")
	slim_pars["use_network_model"] = all_config["NetworkModelParameters"]["use_network_model"]
	_check_boolean(slim_pars["use_network_model"], "Whether to use network model as contact network")
	if not slim_pars["use_network_model"]:
		raise CustomizedError("Please specify \"true\" for whether to use the network model (\"use_network_model\").")
	out_config.write(f"use_network_model:{_writebinary(slim_pars["use_network_model"])}\n")

	if not os.path.exists(os.path.join(slim_pars["cwdir"], "contact_network.adjlist")):
		raise CustomizedError("NetworkGenerator hasn't been run. Please run NetworkGenerator before running this program.")
	out_config.write(f"contact_network_path:{os.path.join(slim_pars["cwdir"], "contact_network.adjlist")}\n")

	slim_pars["host_size"] = all_config["NetworkModelParameters"]["host_size"]
	_check_integer(slim_pars["host_size"], "Number of hosts")
	out_config.write(f"host_size:{slim_pars["host_size"]}\n")

	print("\"NetworkModelParameters\" Checked. ")

	print("Checking \"EpidemiologyModel\"...... ")
	slim_pars["epi_model"] = all_config["EpidemiologyModel"]["model"]
	if slim_pars["epi_model"] not in ["SIR", "SEIR"]:
		raise CustomizedError("Compartmental model (\"model\") has to be SIR or SEIR.")
	out_config.write(f"epi_model:{slim_pars["epi_model"]}\n")
	slim_pars["n_epoch"] = all_config["EpidemiologyModel"]["epoch_changing"]["n_epoch"]
	_check_integer(slim_pars["n_epoch"], "Number of epochs")
	out_config.write("n_epoch:" + str(slim_pars["n_epoch"]) + "\n")

	slim_pars["epoch_changing_generation"] = all_config["EpidemiologyModel"]["epoch_changing"]["epoch_changing_generation"]
	if slim_pars["n_epoch"] > 1:
		if not isinstance(slim_pars["epoch_changing_generation"], list) or len(slim_pars["epoch_changing_generation"])!=slim_pars["n_epoch"]-1:
			raise CustomizedError("Ticks to change epochs (\"epoch_changing_generation\") has to be a list of length of the number of epochs - 1.")
		if any(not isinstance(i, int) or i < 1 or i > slim_pars["n_generation"] for i in slim_pars["epoch_changing_generation"]):
			raise CustomizedError(f"Ticks to change epoch (\"massive_sample_generation\") has to in a tick that is valid (1..{slim_pars["n_generation"]}).")
		out_config.write(f"epoch_changing_generation: {",".join([str(x) for x in slim_pars["epoch_changing_generation"]])}\n")

	
	slim_pars["transmissibility_effsize"] = all_config["EpidemiologyModel"]["genetic_architecture"]["transmissibility"]
	slim_pars["cap_transmissibility"] = all_config["EpidemiologyModel"]["genetic_architecture"]["cap_transmissibility"]
	slim_pars["drugresistance_effsize"] = all_config["EpidemiologyModel"]["genetic_architecture"]["drug_resistance"]
	slim_pars["cap_drugresist"] = all_config["EpidemiologyModel"]["genetic_architecture"]["cap_drugresist"]
	slim_pars["S_IE_rate"] = all_config["EpidemiologyModel"]["transiton_rate"]["S_IE_rate"]
	slim_pars["I_R_rate"] = all_config["EpidemiologyModel"]["transiton_rate"]["I_R_rate"]
	slim_pars["R_S_rate"] = all_config["EpidemiologyModel"]["transiton_rate"]["R_S_rate"]
	slim_pars["latency_prob"] = all_config["EpidemiologyModel"]["transiton_rate"]["latency_prob"]
	slim_pars["E_I_rate"] = all_config["EpidemiologyModel"]["transiton_rate"]["E_I_rate"]
	slim_pars["I_E_rate"] = all_config["EpidemiologyModel"]["transiton_rate"]["I_E_rate"]
	slim_pars["E_R_rate"] = all_config["EpidemiologyModel"]["transiton_rate"]["E_R_rate"]
	slim_pars["sample_rate"] = all_config["EpidemiologyModel"]["transiton_rate"]["sample_rate"]
	slim_pars["recovery_prob_after_sampling"] = all_config["EpidemiologyModel"]["transiton_rate"]["recovery_prob_after_sampling"]

	for param in ["S_IE_rate", "I_R_rate", "R_S_rate", "latency_prob", "E_I_rate", "I_E_rate", "E_R_rate", "sample_rate", "recovery_prob_after_sampling"]:
		if not isinstance(slim_pars[param], list):
			raise CustomizedError(f"({param}) has to be a list [].")
		if len(slim_pars[param]) != slim_pars["n_epoch"]:
			raise CustomizedError("Ticks to change epochs (\"epoch_changing_generation\") has to have length of the number of epochs.")
		if any(not isinstance(i, (float, int)) for i in slim_pars[param]):
			raise CustomizedError(f"The probability of event ({param}) has to be a list of floats.")
		if any(i > 1 or i < 0 for i in slim_pars[param]):
			raise CustomizedError(f"({param}) has to be a list of probabilities, thus between 0 and 1.")
		out_config.write(f"{param}:{','.join(str(x) for x in slim_pars[param])}\n")

	for param in ["cap_transmissibility", "cap_drugresist"]:
		if not isinstance(slim_pars[param], list):
			raise CustomizedError(f"({param}) has to be a list [].")
		if len(slim_pars[param]) != slim_pars["n_epoch"]:
			raise CustomizedError(f"Cap of the trait value for each epoch ({param}) has to have length of the number of epochs.")
		if any(not isinstance(i, (float, int)) for i in slim_pars[param]):
			raise CustomizedError(f"Cap of the trait value for each epoch ({param}) has to be a list of numerical values.")
		out_config.write(f"{param}:{','.join(str(x) for x in slim_pars[param])}\n")

	for param in ["transmissibility_effsize", "drugresistance_effsize"]:
		if not isinstance(slim_pars[param], list):
			raise CustomizedError(f"({param}) has to be a list [].")
		if len(slim_pars[param]) != slim_pars["n_epoch"]:
			raise CustomizedError(f"The effect size set being used in each epoch ({param}) has to have length of the number of epochs.")
		if any(not isinstance(i, int) for i in slim_pars[param]):
			raise CustomizedError(f"The effect size set being used in each epoch ({param}) has to be a list of integers.")
		out_config.write(f"{param}:{','.join(str(x) for x in slim_pars[param])}\n")


	def _check_effect_size(param, max_value):
		if any(i > max_value for i in slim_pars[param]):
			raise CustomizedError(f"({param}) has to be chosen from {list(range(max_value))}.")
		
	_check_effect_size("transmissibility_effsize", all_config["GenomeElement"]["traits_num"][0])
	_check_effect_size("drugresistance_effsize", all_config["GenomeElement"]["traits_num"][1])

	slim_pars["n_massive_sample"] = all_config["EpidemiologyModel"]["massive_sampling"]["event_num"]
	if not isinstance(slim_pars["n_massive_sample"], int):
		raise CustomizedError("The number of massive sampling events must be an integer.")
	out_config.write(f"n_massive_sample:{slim_pars["n_massive_sample"]}\n")

	if slim_pars["n_massive_sample"] > 0:
		slim_pars["massive_sample_generation"] = all_config["EpidemiologyModel"]["massive_sampling"]["generation"]
		if not isinstance(slim_pars["massive_sample_generation"], list) or len(slim_pars["massive_sample_generation"])!=slim_pars["n_massive_sample"]:
			raise CustomizedError("Ticks to do massive sampling (\"massive_sample_generation\") should be a list of length of the number of massive sampling events.")
		if any([i > slim_pars["n_generation"] or i < 1 for i in slim_pars["massive_sample_generation"]]):
			raise CustomizedError(f"Ticks to do massive sampling (\"massive_sample_generation\") has to be a tick that is valid (1..{slim_pars["n_generation"]}).")
		out_config.write(f"massive_sample_generation: {",".join([str(x) for x in slim_pars["massive_sample_generation"]])}\n")

		slim_pars["massive_sample_prob"] = all_config["EpidemiologyModel"]["massive_sampling"]["sampling_prob"]
		slim_pars["massive_sample_recover_prob"] = all_config["EpidemiologyModel"]["massive_sampling"]["recovery_prob_after_sampling"]
		for param in ["massive_sample_prob", "massive_sample_recover_prob"]:
			if not isinstance(slim_pars[param], list) or len(slim_pars[param])!=slim_pars["n_massive_sample"]:
				raise CustomizedError(f"(\"{param}\") has to be a list [].")
			if any([type(i) not in [float, int] or i >1 or i < 0 for i in slim_pars[param]]):
				raise CustomizedError(f"(\"{param}\") has to be a list of probability, thus between 0 and 1.")
			out_config.write(f"{param}: {",".join([str(x) for x in slim_pars[param]])}\n")
	
	slim_pars["super_infection"] = all_config["EpidemiologyModel"]["super_infection"]
	_check_boolean(slim_pars["super_infection"], "Whether to enable super infection")
	out_config.write(f"super_infection: {_writebinary(slim_pars["n_massive_sample"])}\n")
	if slim_pars["super_infection"] and slim_pars["cap_withinhost"]==1:
		print("WARNING: Though super-infection is activated, you specified the capacity within-host being only 1, thus super-infection actually cannot happen.l")
	
	print("\"EpidemiologyModel\" Checked. ")
		
	print("Checking \"Postprocessing_options\"...... ")
	_check_boolean(all_config["Postprocessing_options"]["do_postprocess"], "Whether to postprocess results")
	if all_config["Postprocessing_options"]["do_postprocess"]:
		post_processing_config = all_config["Postprocessing_options"]
		post_processing_config["n_trait"] = all_config["GenomeElement"]["traits_num"]
		print("Post-simulation data processing starts.")

		branch_color_trait = post_processing_config["tree_plotting"]["branch_color_trait"]
		if not isinstance(branch_color_trait, int) or branch_color_trait < 0 or branch_color_trait > sum(post_processing_config["n_trait"]):
			raise CustomizedError(f"How to color the branches of the tree (branch_color_trait) should be an integer chosen from (0: color by seed, 1..{sum(post_processing_config["n_trait"])}: trait id.")
		if post_processing_config["tree_plotting"]["branch_color_trait"]==0:
			print("The tree will be colored by seed.")
		else:
			print(f"The tree will be colored by its trait {post_processing_config["tree_plotting"]["branch_color_trait"]}")
	else:
		print("Post-simulation data processing is not enabled.")
		post_processing_config = {}
	print("\"Postprocessing_options\" Checked. ")

	return slim_pars, post_processing_config


def create_slim_script(slim_pars):
	"""
    Create a SLiM script in the working directory according to the config.
    
    Parameters:
        slim_pars (dict): SLiM parameters.
	"""
	print("********************************************************************")
	print("                       CREATING SLIM SCRIPT")
	print("********************************************************************")

	code_path = os.path.join(os.path.dirname(__file__), "slim_scripts")
	mainslim_path = os.path.join(slim_pars["cwdir"], "simulation.slim")

	if os.path.exists(mainslim_path):
		os.remove(mainslim_path)

	# Create a slim file in the working directory
	f = open(mainslim_path, "w")
	f.close()

	# Append trait calculation functions
	append_files(os.path.join(code_path, "trait_calc_function.slim"), mainslim_path)

	# Initialization
	append_files(os.path.join(code_path, "initialization_pt1.slim"), mainslim_path)
	append_files(os.path.join(code_path, "read_config.slim"), mainslim_path)
	append_files(os.path.join(code_path, "initialization_pt2.slim"), mainslim_path)
	if slim_pars["use_genetic_model"]:
		append_files(os.path.join(code_path, "genomic_element_init_effsize.slim"), mainslim_path)
	else:
		append_files(os.path.join(code_path, "genomic_element_init.slim"), mainslim_path)
	append_files(os.path.join(code_path, "initialization_pt3.slim"), mainslim_path)

	# Mutation block (for m1)
	if slim_pars["use_genetic_model"]:
		append_files(os.path.join(code_path, "mutation_effsize.slim"), mainslim_path)

	# Block control 1first()
	append_files(os.path.join(code_path, "block_control.slim"), mainslim_path)

	# Seedss read-in and network read=in
	if slim_pars["use_network_model"]:
		if slim_pars["use_reference"]:
			append_files(os.path.join(code_path, "seeds_read_in_noburnin.slim"), mainslim_path)
		else:
			append_files(os.path.join(code_path, "seeds_read_in_network.slim"), mainslim_path)
		append_files(os.path.join(code_path, "contact_network_read_in.slim"), mainslim_path)

	# Epoch changing
	if slim_pars["n_epoch"] > 1:
		append_files(os.path.join(code_path, "change_epoch.slim"), mainslim_path)

	## Self reproduction
	append_files(os.path.join(code_path, "self_reproduce.slim"), mainslim_path)

	# Transmission reproduction
	if slim_pars["use_genetic_model"] == False:
		append_files(os.path.join(code_path, "transmission_nogenetic.slim"), mainslim_path)
	else:
		if slim_pars["trans_type"] == "additive":
			append_files(os.path.join(code_path, "transmission_additive.slim"), mainslim_path)
		elif slim_pars["trans_type"] == "bialleleic":
			append_files(os.path.join(code_path, "transmission_bialleleic.slim"), mainslim_path)
		if any(idx == 0 for idx in slim_pars["transmissibility_effsize"]):
			append_files(os.path.join(code_path, "transmission_nogenetic.slim"), mainslim_path)

	# Within-host reproduction
	if slim_pars["within_host_reproduction"]:
		append_files(os.path.join(code_path, "within_host_reproduce.slim"), mainslim_path)

	# Kill old pathogens
	append_files(os.path.join(code_path, "kill_old_pathogens.slim"), mainslim_path)

	# State transition for exposed hosts
	if slim_pars["epi_model"]=="SEIR":
		append_files(os.path.join(code_path, "Exposed_process.slim"), mainslim_path)

	# State transition for infected hosts
	if any(idx != 0 for idx in slim_pars["drugresistance_effsize"]):
		if slim_pars["trans_type"] == "additive":
			append_files(os.path.join(code_path, "Infected_process_additive.slim"), mainslim_path)
		elif slim_pars["trans_type"] == "bialleleic":
			append_files(os.path.join(code_path, "Infected_process_additive.slim"), mainslim_path)
	if any(idx == 0 for idx in slim_pars["drugresistance_effsize"]):
		append_files(os.path.join(code_path, "Infected_process_nogenetic.slim"), mainslim_path)

	# Massive sampling events
	if slim_pars["n_massive_sample"] > 0:
		append_files(os.path.join(code_path, "massive_sampling.slim"), mainslim_path)

	# New infections
	if slim_pars["super_infection"] == False:
		append_files(os.path.join(code_path, "New_infection_process.slim"), mainslim_path)
	else:
		append_files(os.path.join(code_path, "New_infection_process_superinfection.slim"), mainslim_path)

	# Recovered individuals
	if any(rate != 0 for rate in slim_pars["R_S_rate"]):
		append_files(os.path.join(code_path, "Recovered_process.slim"), mainslim_path)

	# Logging all things
	append_files(os.path.join(code_path, "log.slim"), mainslim_path)

	# Finish simulation
	append_files(os.path.join(code_path, "finish_simulation.slim"), mainslim_path)


def run_per_slim_simulation(slim_config_path, wk_dir, runid):
	"""
    Run the SLiM script generated by create_slimscript.

    Parameters:
        slim_config_path (str): Path to the SLiM config file.
        wk_dir (str): Working directory where the simulation will be executed.
        runid (int): Run ID.
    """
	output_path = os.path.join(wk_dir, str(runid))
	if os.path.exists(output_path):
		shutil.rmtree(output_path)
	# Removes existing output directory and its contents
	os.makedirs(output_path)  # Creates output directory

	script_path = os.path.join(wk_dir, "simulation.slim")
	slim_stdout_path = os.path.join(output_path, "slim.stdout")

	# Run SLiM with subprocess
	with open(slim_stdout_path, 'w') as fd:
		subprocess.run(["slim", "-d", f"config_path=\"{slim_config_path}\"", "-d", f"runid={runid}", script_path], stdout=fd)
	

def run_all_slim_simulation(slim_config_path = "", slim_pars = {}, dataprocess_pars = {}):
	"""
    Run all SLiM simulations based on the provided configuration.

    Parameters:
        slim_config_path (str): Path to the SLiM configuration file.
        slim_pars (dict): Dictionary containing SLiM parameters.
        dataprocess_pars (dict): Dictionary containing data processing parameters.
	"""
	# Set default slim_config_path if not provided
	if slim_config_path == "":
		slim_config_path = os.path.join(slim_pars["cwdir"], "slim.params")

	print("********************************************************************")
	print("                     RUNNING THE SIMULATION")
	print("********************************************************************")

	# List to store successful run ids
	run_success = []

	# Iterate over replicates
	for runid in range(1,slim_pars["n_replicates"] + 1):
		print(f"Running simulation for replication {runid}......")
		# Run SLiM simulation
		run_per_slim_simulation(slim_config_path, slim_pars["cwdir"], runid)
		print(f"Replication {runid} simulation finished.")

		# Check if sampled genomes exits
		sampled_genomes_path = os.path.join(slim_pars["cwdir"], str(runid), "sampled_genomes.trees")
		if os.path.exists(sampled_genomes_path):
			if len(dataprocess_pars) > 0:
				each_wkdir = os.path.join(slim_pars["cwdir"], str(runid))
				print(f"Processing replication {runid} treesequence file...")
				# Run data processing for each replicate
				run_per_data_processing(
					slim_pars["cwdir"], slim_pars["use_genetic_model"], runid, dataprocess_pars["n_trait"], 
					slim_pars["seed_host_matching_path"], dataprocess_pars["tree_plotting"]["branch_color_trait"])
				if slim_pars["use_reference"]:
					seed_phylo = ""
				else:
					seed_phylo = os.path.join(slim_pars["cwdir"], "seeds.nwk")
				# Plot transmission tree, strain distribution trajectory, and SEIR trajectory
				print(f"Plotting transmission tree for replication {runid}...")
				plot_per_transmission_tree(each_wkdir, slim_pars["seed_size"], slim_config_path, dataprocess_pars["n_trait"], seed_phylo)
				print(f"Plotting strain distribution trajectory for replication {runid}...")
				plot_strain_distribution_trajectory(each_wkdir, slim_pars["seed_size"], slim_pars["n_generation"])
				print(f"Plotting SEIR trajectory for replication {runid}...")
				if os.path.exists(os.path.join(each_wkdir, "SEIR_trajectory.csv.gz")):
					plot_SEIR_trajectory(each_wkdir, slim_pars["seed_size"], slim_pars["host_size"], slim_pars["n_generation"])
					run_success.append(runid)
		else:
			print(f"There's no sampled genome in replicate {runid}. Either the simulation failed or the sampling rate is too low. \
		 Please check your config file and confirm those are your desired simulation parameters.")
	# Plot aggregated SEIR and strain distribution trajectories
	print(f"Plotting the aggregated SEIR trajectory...")
	plot_all_SEIR_trajectory(slim_pars["cwdir"], slim_pars["seed_size"], slim_pars["host_size"], slim_pars["n_generation"], run_success)
	print(f"Plotting the aggregated strain distribution trajectory...")
	plot_all_strain_trajectory(slim_pars["cwdir"], slim_pars["seed_size"], slim_pars["host_size"], slim_pars["n_generation"], run_success)

	print("********************************************************************")
	print("                    FINISHED. THANKS FOR USING.")
	print("********************************************************************")

def all_slim_simulation_by_config(all_config):
    """
    Run all SLiM simulations based on the provided configuration.

    Parameters:
        all_config (dict): Dictionary containing all SLiM configuration parameters.

    Returns:
        None
    """
    # Create SLiM configuration parameters and data processing parameters
    slim_pars, dataprocess_pars = create_slim_config(all_config)

    # Create SLiM script
    create_slim_script(slim_pars)

    # Set working directory
    wk_dir = slim_pars["cwdir"]

    # Run all SLiM simulations
    run_all_slim_simulation(
        slim_config_path=os.path.join(wk_dir, "slim.params"),
        slim_pars=slim_pars,
        dataprocess_pars=dataprocess_pars
    )

def main():
	parser = argparse.ArgumentParser(description='Generate or modify seeds.')
	parser.add_argument('-config', action='store',dest='config', type=str, required=True, help="Path to the config file")

	args = parser.parse_args()

	config_path = args.config

	all_slim_simulation_by_config(read_params(config_path))


if __name__ == "__main__":
    main()











