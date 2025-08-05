import os, tskit, pyslim, shutil, subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib

NUM_META_COLS = 3
POS_COL = 1
TRANS_INDEX = 0
DRUG_RES_INDEX = 1

def read_tseq(each_wk_dir_):
	"""
	Returns the tree sequence, sampled tree, sample size, 
	and generation simulated in specified working directory.

	Parameters:
		each_wk_dir (str): Full directory to a single simulation.

	Returns:
		ts (TreeSequence): Tree sequence object of the sampled tree.
		sampled_tree (TreeSequence): Tree sequence object of the sampled tree (hyploid).
		sample_size (list[int]): List of the number of samples [0, 1, 2, ..., sample_size - 1]
		n_gens (int): The number of generation used in simulation.
	"""
	ts = tskit.load(os.path.join(each_wk_dir_, "sampled_genomes.trees"))
	n_gens = ts.metadata["SLiM"]["tick"]
	sample_size = range(ts.tables.individuals.num_rows)
	sampled_tree = ts.simplify(samples = [2 * i for i in sample_size])
	return ts, sampled_tree, sample_size, n_gens


def find_label(tseq_smp, sim_gen, sample_size):
	"""
	Returns the labels for tips, labeled as ${tick}.${host_id}. 
	${tick} is the generation that pathogen got sampled.

	Parameters:
		tseq_smp (TreeSequence): The simplified tree sequence object.
		sim_gen (int): The number of generation used in simulation.
		sample_size (int): The sample size.
	"""
	all_tables = tseq_smp.tables
	### ATTENTION FOR TESTING: WHAT DOES 0 MEAN IN all_tables.nodes.time?
	leaf_time = sim_gen - all_tables.nodes.time[sample_size].astype(int)
	real_name = {}
	for l_id in sample_size:
		subpop_now = all_tables.individuals[l_id].metadata["subpopulation"]
		real_name[l_id] = str(leaf_time[l_id]) + "." + str(subpop_now)
	return real_name


def nwk_output(tseq_smp, real_name, each_wk_dir_, seed_host_match_path):
	"""
	Writes transmission tree newick file for each simulation.

	Parameters:
		tseq_smp (TreeSequence): The simplified tree sequence object.
		real_name (str): The true label.
		each_wk_dir_ (str): The working directory for each simulation.
		seed_host_match_path (str): Full path to seed host match file.
	"""
	roots_all = tseq_smp.first().roots
	output_path = os.path.join(each_wk_dir_, "transmission_tree")
	df_host_seed_match = pd.read_csv(seed_host_match_path)
	match_dict = df_host_seed_match.set_index('host_id')['seed'].to_dict()
	# Removes all the subdirectories
	if os.path.exists(output_path):
		shutil.rmtree(output_path) 
	os.makedirs(output_path)
	for root in roots_all:
		root_subpop = tseq_smp.tables.individuals[root].metadata["subpopulation"]
		with open(os.path.join(output_path, f"{match_dict[root_subpop]}.nwk"), "w") as nwk:
			nwk.write(tseq_smp.first().as_newick(root = root, node_labels = real_name) + "\n")


def trait_calc_tseq(wk_dir_, tseq_smp, n_trait, sigmoid=False):
	"""
	Compute the trait values for all nodes given a TreeSequence object.

	Parameters:
		wk_dir_ (str): Full path to the working directory of a simulation.
		tseq_smp (TreeSequence): The TreeSequence objects.
		n_trait (list[int]): A list of number of traits for transmissibility and drug resistance.

	Returns:
		real_traits_vals (list[dict[int (node_id), float]]): Dictionary storing the trait values of every node.
		trvs_order (list[int]): Pre-order traversal or nodes in the TreeSequence object.
	"""
	# Compute the total number of traits for both transmissibility and drug resistance.
	num_trait = sum(n_trait.values())

	node_size = tseq_smp.tables.nodes.num_rows
	muts_size = tseq_smp.tables.mutations.num_rows
	tree_first = tseq_smp.first()
	trvs_order = list(tree_first.nodes(order="preorder"))

	if n_trait=={}:
		return [], trvs_order
	else:
		eff_size = pd.read_csv(os.path.join(wk_dir_, "causal_gene_info.csv"))
		# Increment the end index so it is no longer inclusive
		eff_size["end"] += 1
		

		# if there are no traits to calculate, directly return empty list
		if num_trait==0:
			return [], trvs_order

		search_intvls = np.array(np.ravel(eff_size[["start", "end"]]), dtype="float")
		search_intvls[0::2] -= 0.1
		search_intvls[1::2] += 0.1

		# seem unncessary
		pos_values = [] # list of positions of mutations happened
		node_ids = [] # list of list of nodes with specific mutations

		for mut_idx in range(muts_size):
			mut = tseq_smp.mutation(mut_idx)
			pos_values.append(tseq_smp.site(mut.site).position + 1)
			node_ids.append(mut.node)
		
		# Get the indices of each mutation position if inserted into the search intervals
		intvs = np.searchsorted(search_intvls, pos_values)

		which_m2 = np.where(intvs % 2 == 1)[0]
		real_traits_vals = []

		if len(which_m2) == 0:
			# for _ in range(num_trait):
			trait_val_now = { i: 0 for i in range(node_size)}
			print("WARNING: There's no mutations related to any trait in samples from this replication.", flush = True)
			return [trait_val_now for _ in range(num_trait)], trvs_order
		
		for trait_idx in range(num_trait):
			node_plus = np.zeros(node_size)
			for mut in which_m2:
				# nodes_ids[mut] are the nodes with that specific mutation
				# intvs[mut] // 2 implies the gene that mutation belongs to
				# record the effect size caused by the new mutations given the parent node
				node_plus[node_ids[mut]] += eff_size.iloc[intvs[mut] // 2, trait_idx + NUM_META_COLS]
			trait_val = {j: 0 for j in range(-1, node_size)}
			# if sigmoid==True:
			# 	for node_id in trvs_order:
			# 		trait_val[node_id] = 1 / (1 + np.exp(np.negative(trait_val[tree_first.parent(node_id)] + node_plus[node_id])))
			# else:
			# 	for node_id in trvs_order:
			# 		trait_val[node_id] = trait_val[tree_first.parent(node_id)] + node_plus[node_id]
			for node_id in trvs_order:
		 		trait_val[node_id] = trait_val[tree_first.parent(node_id)] + node_plus[node_id]
			trait_val.pop(-1)
			real_traits_vals.append(trait_val)
		# order of traversal
		# if sigmoid==True:
		# 	print(trait_val)
		# 	print(type(real_traits_vals))
		# 	real_traits_vals = 1 / (1 + np.exp(np.negative(np.array(real_traits_vals))))
		return real_traits_vals, trvs_order

def floats_to_colors_via_matplotlib(float_values):
	"""
	Converts a list of float values to hexadecimal colors using Matplotlib's colormap.

	Parameters:
		float_values (list[float]): List of float values.

	Returns:
		hex_colors (list[str]): List of hexadecimal colors corresponding to the input float values.
	"""
	cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "red"])
	colors = [cmap(value) for value in float_values]
	hex_colors = [mcolors.to_hex(color) for color in colors]
	return hex_colors


def color_by_trait_normalized(trait_val, trvs_order, sigmoid=False):
	"""
	Colorizes nodes based on normalized trait values.

	Parameters:
		trait_val (dict): Dictionary containing trait values for nodes.
		trvs_order (list[int]): List of node IDs in the pre-order traversal.

	Returns:
		dict: Dictionary mapping node IDs to hexadecimal colors based on normalized trait values.
	"""
	all_traits = np.array(list(trait_val.values()))
	if np.max(all_traits) > np.min(all_traits):
		normalized_traits = (all_traits - np.min(all_traits)) / (np.max(all_traits) - np.min(all_traits))
		color_map_nodes = floats_to_colors_via_matplotlib(normalized_traits)
		color_map_dict = {i: color_map_nodes[i] for i in range(len(trvs_order))}
	else:
		color_map_dict = {i:"#000000" for i in range(len(trvs_order))}

	return color_map_dict

def color_by_seed(tseq_smp, trvs_order, seed_host_match_path):
	"""
	Colorizes nodes based on seed-host matches.

	Parameters:
		tseq_smp (TreeSequence): The TreeSequence object.
		trvs_order (list[int]): List of node IDs in the order of traversal.
		seed_host_match_path (str): Path to the seed-host match file.

	Returns:
		dict: Dictionary mapping node IDs to hexadecimal colors based on seed-host matches.
	"""
	roots_all = tseq_smp.first().roots
	table_ind = tseq_smp.tables.individuals
	node_size = tseq_smp.tables.nodes.num_rows
	trait_val = {node_id: 0 for node_id in range(node_size)}
	tree_first = tseq_smp.first()
	
	# Read seed-host match file
	match_dict = {}
	df_host_seed_match = pd.read_csv(seed_host_match_path)
	match_dict = df_host_seed_match.set_index('host_id')['seed'].to_dict()

	# Create colormap based on the number of seed-host matches
	# cmap = cm.get_cmap('viridis')
	cmap = matplotlib.colormaps["viridis"]
	values = np.linspace(0, 1, len(match_dict))
	colors = cmap(values)
	hex_codes = [mcolors.to_hex(color) for color in colors]

	# Assign colors to roots based on seed-host matches
	for root in roots_all:
		seed_id = match_dict.get(table_ind[root].metadata.get("subpopulation", -1), -1)
		if seed_id != -1:
			trait_val[root] = hex_codes[seed_id]

	# Propagate colors to non-root nodes
	for u in trvs_order:
		if u not in roots_all:
			trait_val[u] = trait_val[tree_first.parent(u)]

	return trait_val


def metadata_generate(sample_size, trvs_order, sampled_ts, sim_gen, traits_num_values, trait_color):
	"""
	Generates metadata for nodes based on various parameters.

	Parameters:
		sample_size (int): Number of sampled nodes.
		trvs_order (list[int]): List of node IDs in the order of traversal.
		sampled_ts (TreeSequence): The sampled TreeSequence object.
		sim_gen (int): Number of simulation generations.
		traits_num_values (list[dict[int, float]]): Trait values for each node.
		trait_color (dict): Dictionary mapping node IDs to trait colors.

	Returns:
		dict: Dictionary containing metadata for each node.
	"""
	a_big_df = {}
	sample_size_max = len(sample_size)
	nodes_table = sampled_ts.tables.nodes
	inds_table = sampled_ts.tables.individuals
	tree_first = sampled_ts.first()
	traits_num = len(traits_num_values)

	for u in trvs_order:
		node_id = str(u)
		node_time = str(sim_gen - nodes_table.time[u].astype(int))
		if u < sample_size_max:
			subpop_id = str(inds_table[u].metadata["subpopulation"])
			name = f"{node_time}.{subpop_id}"
		else:
			subpop_id = str(-1)
			name = "."
		parent_id = str(tree_first.parent(u))
		color_by_trait = str(trait_color[u])
		traits_values_str = [str(traits_num_values[trait_id][u]) for trait_id in range(traits_num)]
		a_big_df[u] = [node_id, name, node_time, subpop_id, parent_id, color_by_trait] + traits_values_str

	return a_big_df


def write_metadata(mtdata, each_wk_dir_, n_trait, color_trait):
	"""
	Writes metadata to a CSV file.

	Parameters:
		mtdata (dict): Dictionary containing metadata for each node.
		each_wk_dir_ (str): Full path to the directory where the CSV file will be written.
		n_trait (list[int]): A list containing the number of traits for transmissibility 
		and drug resistance.
		color_trait (int): Index of the trait to be used for color coding.
	"""

	with open(os.path.join(each_wk_dir_, "transmission_tree_metadata.csv"), "w") as csv:
		# Write header
		if color_trait <= n_trait["transmissibility"]:
			color_trait_id = color_trait
		else:
			color_trait_id = color_trait - n_trait["transmissibility"]
		header = "node_id,name,node_time,subpop_id,parent_id,color_trait_" + str(color_trait_id)
		for i in range(n_trait["transmissibility"]):
			header += f",transmissibility_{i + 1}"
		for i in range(n_trait["drug_resistance"]):
			header += f",drug_resistance_{i + 1}"
		csv.write(header + "\n")
		# Write node data
		for _, node_data in mtdata.items():
			csv.write(",".join(node_data) + "\n")


def output_tseq_vcf(wk_dir_, real_label, sampled_ts):
	"""
	Converts a TreeSequence object to VCF format and writes it to a file.

	Parameters:
		wk_dir_ (str): Full path to the working directory.
		real_label (dict): Dictionary mapping individual IDs to their labels.
		sampled_ts (TreeSequence): The sampled TreeSequence object.
	"""
	vcf_path = os.path.join(wk_dir_, "sampled_pathogen_sequences.vcf.tmp")

	# Write VCF file
	with open(vcf_path, "w") as f:
		nu_ts = pyslim.convert_alleles(sampled_ts)
		nu_ts.write_vcf(f, individual_names = real_label.values())

	# Modify position values and write to final VCF file
	with open(vcf_path, "r") as in_vcf, open(os.path.join(wk_dir_, "sampled_pathogen_sequences.vcf"), "w") as out_vcf:
		for line in in_vcf:
			if line.startswith("#"):
				out_vcf.write(line)
			else:
				fields = line.split("\t")
				new_line = "\t".join(fields[:POS_COL]+ [str(int(fields[POS_COL]) + 1)] + fields[POS_COL + 1:])
				out_vcf.write(new_line)
	
	# Remove the temporary VCF file
	os.remove(vcf_path)


def get_full_fasta(ref_path, wk_dir_):
	ref_seq = SeqIO.parse(open(ref_path), 'fasta')
	for fasta in ref_seq:
		refseq = str(fasta.seq)
	
	index_SNP = []
	with open(os.path.join(wk_dir_, "final_samples_snp_pos.csv")) as index_file:
		for line2 in index_file:
			line1 = line2.rstrip()
			line = line1.split(",")[1]
			if line.isdigit():
				num_index = int(line)
				index_SNP.append(num_index - 1)

	snp_path = os.path.join(wk_dir_, "sample.SNPs_only.fasta")
	snp_seq = SeqIO.parse(open(snp_path), 'fasta')
	with open(os.path.join(wk_dir_, "sample.wholegenome.fasta"), "w") as out_fa:
		for fasta in snp_seq:
			ref = list(refseq)
			name, sequence = fasta.id, str(fasta.seq)
			out_fa.write(">" + name + "\n")
			for i, index in enumerate(index_SNP):
				ref[index] = sequence[i]
			out_fa.write(''.join(ref) + "\n")



def output_fasta(ref_path, wk_dir_):
	rscript_path = os.path.join(os.path.dirname(__file__), "generate_fas.r")
	subprocess.run(["Rscript", rscript_path, wk_dir_])
	get_full_fasta(ref_path, wk_dir_)


def run_per_data_processing(ref_path, wk_dir_, gen_model, runid, n_trait, seed_host_match_path, seq_out, color_trait=1, sigmoid=False):
	"""
	Performs data processing tasks for a specific run.

	Parameters:
		wk_dir_ (str): Working directory path.
		gen_model (bool): Whether to calculate trait values.
		runid (int): Identifier for the current run.
		n_trait (list): List containing the number of traits for transmissibility and drug resistance.
		seed_host_match_path (str): Path to the seed-host match file.
		color_trait (int): Index of the trait to be used for coloring the tree (default is 1).
	"""
	# Set up directory for the current run
	each_wk_dir = os.path.join(wk_dir_, str(runid))
	# Read TreeSequence data
	_, sampled_ts, sample_size, sim_gen = read_tseq(each_wk_dir)
	# Find real labels
	real_label = find_label(sampled_ts, sim_gen, sample_size)
	# Write Newick output
	nwk_output(sampled_ts, real_label, each_wk_dir, seed_host_match_path)

	# Determine if trait calculation is needed
	# if color_trait == 0:
	# 	gen_model = False

	# CHECK W/ PERRY
	# if gen_model:
	# 	traits_num_values, trvs_order = trait_calc_tseq(wk_dir_, sampled_ts, n_trait)
	# 	trait_color = color_by_trait_normalized(traits_num_values[color_trait - 1], trvs_order)
	# 	mtdata = metadata_generate(sample_size, trvs_order, sampled_ts, sim_gen, traits_num_values, trait_color)
	# 	write_metadata(mtdata, each_wk_dir, n_trait, color_trait)

	# CHECK W/ PERRY
	if gen_model:
		traits_num_values, trvs_order = trait_calc_tseq(wk_dir_, sampled_ts, n_trait, sigmoid)
		if color_trait > 0:
			trait_color = color_by_trait_normalized(traits_num_values[color_trait - 1], trvs_order)
		else:
			trait_color = color_by_seed(sampled_ts, trvs_order, seed_host_match_path)
		# if color_trait > 0:
		# 	trait_color = color_by_trait_normalized(traits_num_values[color_trait - 1], trvs_order)
		# else:
		# 	trait_color = color_by_seed(sampled_ts, trvs_order, seed_host_match_path)
	else:
		traits_num_values, trvs_order = trait_calc_tseq(wk_dir_, sampled_ts, {})
		trait_color = color_by_seed(sampled_ts, trvs_order, seed_host_match_path)
		color_trait = 0
	mtdata = metadata_generate(sample_size, trvs_order, sampled_ts, sim_gen, traits_num_values, trait_color)
	write_metadata(mtdata, each_wk_dir, n_trait, color_trait)


	
	# Output VCF file
	if seq_out["vcf"] or seq_out["fasta"]:
		print("Writing VCF file of sampled pathogens...")
		output_tseq_vcf(each_wk_dir, real_label, sampled_ts)

	# OUtput FASTA file
		if seq_out["fasta"]:
			print("Writing FASTA file of sampled pathogens...")
			output_fasta(ref_path, each_wk_dir)



############################## PLOTTING #############################################################

def plot_per_transmission_tree(each_wk_dir_, seed_size, slim_config_path, n_traits, seed_phylo_path, heatmap_trait):
	"""
    Plots transmission trees per simulation run.

    Parameters:
        each_wk_dir_ (str): Full path to the working directory for each run.
        seed_size (int): Number of seeds in the simulation.
        slim_config_path (str): Path to the SLiM configuration file.
        n_traits (tuple): Tuple containing the number of traits for transmissibility and drug resistance.
        seed_phylo_path (str): Path to the seed phylogeny file.
	"""
	rscript_path = os.path.join(os.path.dirname(__file__), "plot_tree.r")
	subprocess.run(["Rscript", rscript_path, each_wk_dir_, str(seed_size), slim_config_path, \
				 str(n_traits["transmissibility"]), str(n_traits["drug_resistance"]), seed_phylo_path, \
				 heatmap_trait])


def plot_strain_distribution_trajectory(each_wk_dir_, seed_size, n_generation):
	"""
    Plots strain distribution trajectory.

    Parameters:
        each_wk_dir_ (str): Full path to the working directory for each run.
        seed_size (int): Number of seeds in the simulation.
        n_generation (int): Number of generations.
    """
	# Read CSV with seed_ids as columns and generations as rows
	eff_size = pd.read_csv(os.path.join(each_wk_dir_, "strain_trajectory.csv.gz"), header = None, \
						names = range(seed_size), compression = "gzip")
	# Attach information for the starting generation
	eff_size = pd.concat([pd.DataFrame({i: [1] for i in range(seed_size)}), eff_size]).reset_index(drop = True)

	# Compute proportion at each generation
	eff_size_normalized = eff_size.div(eff_size.sum(axis=1), axis = 0)

	# Plot
	ax = eff_size_normalized.plot(kind = 'area', stacked = True, figsize = (10, 6), cmap = 'viridis')

	plt.xlabel('Generations')
	plt.ylabel('Proportion of strains')
	plt.title('Proportion of Different strains Through Time')
	plt.ylim(0, 1)
	plt.xlim(0, n_generation)
	ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), title = 'Strains')

	plt.tight_layout()
	plt.savefig(os.path.join(each_wk_dir_, "strain_trajectory.png"))
	plt.close()


def plot_SEIR_trajectory(each_wk_dir_, seed_size, host_size, n_generation):
	"""
    Plots SEIR trajectory.

    Parameters:
        each_wk_dir_ (str): Full path to the working directory for each run.
        seed_size (int): Number of seeds in the simulation.
        host_size (int): Total number of hosts in the simulation.
        n_generation (int): Number of generations.
	"""
	seir = pd.read_csv(os.path.join(each_wk_dir_, "SEIR_trajectory.csv.gz"), \
					header = None, names = ["S", "E", "I", "R"])
	seir = pd.concat([pd.DataFrame({"S": [host_size - seed_size], "E": [0], \
								 "I": [seed_size], "R": [0]}), seir]).reset_index(drop = True)

	seir.plot(kind = 'line', figsize=(10, 6), cmap = 'viridis', linewidth = 3)

	plt.xlabel('Generations')
	plt.ylabel('Number of hosts (Size of compartments)')
	plt.title('SEIR Trajectory')
	plt.ylim(0, host_size)
	plt.xlim(0, n_generation)
	# Add legend
	plt.legend()

	plt.tight_layout()

	plt.savefig(os.path.join(each_wk_dir_, "SEIR_trajectory.png"))
	plt.close()


def plot_all_SEIR_trajectory(wk_dir_, seed_size, host_size, n_generation, run_success):
	"""
    Plot SEIR (Susceptible-Exposed-Infectious-Recovered) trajectories.

    Parameters:
    	wk_dir_ (str): The directory where the SEIR data is stored.
    	seed_size (int): The initial size of the infected population / 
   		host_size (int): The total population size.
		n_generation (int): The number of generations.
    	run_success (list): List of run IDs for successful simulations.
    	plot_type (str): Type of plot to generate. Can be either 'line' for line plot or 'bar' for stacked bar plot. Default is 'line'.
    """
	if len(run_success) <= 0: return

	# Initiate the "empty" frame
	seir_sum = pd.DataFrame({"S": [0], "E": [0], "I": [0], "R": [0]})
	init_df = pd.DataFrame({"S": [host_size - seed_size],"E": [0], "I": [seed_size], "R": [0]})

	# Iterate through all the trajectories
	for idx, run_id in enumerate(run_success):
		each_wk_dir = os.path.join(wk_dir_, str(run_id))
		seir = pd.read_csv(os.path.join(each_wk_dir, "SEIR_trajectory.csv.gz"), header = None, names = ["S", "E", "I", "R"])
		seir = pd.concat([init_df.copy(), seir]).reset_index(drop = True)
		# SUSPICIOUS
		if idx == 0:
			ax = seir.plot(kind = 'line', figsize = (10, 6), cmap = 'viridis', alpha = 0.4)
		else:
			ax = seir.plot(ax = ax, kind = 'line', legend = None, cmap = 'viridis', alpha = 0.4)
		seir_sum = seir_sum.add(seir, fill_value = 0)

	seir_avg = seir_sum.div(len(run_success))
	ax = seir_avg.plot(ax = ax, kind ='line', cmap = 'viridis', linewidth = 3, legend = None)

	plt.xlabel('Generations')
	plt.ylabel('Number of hosts')
	plt.title('SEIR Trajectory')
	plt.ylim(0, host_size)
	plt.xlim(0, n_generation)

	plt.tight_layout()

	plt.savefig(os.path.join(wk_dir_, "all_SEIR_trajectory.png"))
	plt.close()


def plot_all_strain_trajectory(wk_dir_, seed_size, host_size, n_generation, run_success):
	"""
    Plot the trajectory of different strains over time.

    Parameters:
    	wk_dir_ (str): The directory where the strain data is stored.
    	seed_size (int): The number of initial strains.
    	host_size (int): The total population size.
    	n_generation (int): The number of generations.
    	run_success (list): List of run IDs for successful simulations.
    """
	if len(run_success) <= 0: return

	# Initiate the "empty" frame
	traj_sum = pd.DataFrame({str(i): [0] for i in range(seed_size)})
	init_df = pd.DataFrame({str(i): [1] for i in range(seed_size)})
	col_names = [str(i) for i in range(seed_size)]

	# Iterate through all trajectoreis
	for idx, run_id in enumerate(run_success):
		each_wk_dir = os.path.join(wk_dir_, str(run_id))
		strain_df = pd.read_csv(os.path.join(each_wk_dir, "strain_trajectory.csv.gz"), header = None, names = col_names)
		strain_df = pd.concat([init_df.copy(), strain_df]).reset_index(drop = True)
		strain_df_normalized = strain_df.div(strain_df.sum(axis = 1), axis = 0)
		# SUSPICIOUS
		if idx == 0:
			ax = strain_df_normalized.plot(kind = 'line', figsize = (10, 6), cmap = 'viridis', alpha = 0.4)
		else:
			ax = strain_df_normalized.plot(ax = ax, kind = 'line', legend = None, cmap = 'viridis', alpha = 0.4)
		traj_sum = traj_sum.add(strain_df_normalized, fill_value = 0)

	traj_avg = traj_sum.div(len(run_success))
	ax = traj_avg.plot(ax = ax, kind = 'line', legend = None, cmap = 'viridis', linewidth = 3)

	# MAYBE STACKING IS BETTER
	plt.xlabel('Generations')
	plt.ylabel('Proportion of strain')
	plt.title('Proportion of Different strains Through Time')
	plt.xlim(0, n_generation)
	plt.ylim(0, 1)

	plt.tight_layout()

	plt.savefig(os.path.join(wk_dir_, "all_strains_trajectory.png"))
	plt.close()


		
