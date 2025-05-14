from base_func import *
from error_handling import CustomizedError
import numpy as np
import pandas as pd
import argparse, statistics, os, json
from Bio import SeqIO

START_IDX = 0
END_IDX = 1
E_SIZE = 2

GFF_START = 3
GFF_END = 4
GFF_INFO = 8

def _count_gff_genes(gff_path):
	"""
	Returns the number of genetic elements documented in a gff file.

	Parameters:
		gff_path (str): Full path to the GFF-like file.
	"""
	if not os.path.isfile(gff_path):
		raise CustomizedError(f"The provided path to gff file ({gff_path}) is not a valid file path")
	
	n_genes = 0
	with open(gff_path, "r") as gff:
		for line in gff:
			if not (line.startswith("\n") or line.startswith("#")):
				# count the line if it is not part of the header
				n_genes += 1

	return n_genes

def seeds_trait_calc(wk_dir, dict_c_g, num_seed = 0):
	"""
	Calculate trait values (sum of all mutation's effect sizes) of all seeds.
	Must be run after seed_generator.

	Parameters:
		wk_dir (str): Working directory.
		dict_c_g (dict): A dictionary that stores the genetic elements' effect sizes.
						 Keys are gene names, values are lists of the form [start_pos(int), 
						 end_pos(int), effect_size(float)].

	Returns:
		list: A list of trait values (length = seed size).
	"""

	seed_vals = []
	seeds_vcf_dir = os.path.join(wk_dir, "originalvcfs/")

	# raise exception if we do not have access to VCF of individual seeds
	if not os.path.exists(seeds_vcf_dir):
		print("WARNING: seed_generator.py hasn't been run. "
					"If you want to use seed sequence different than reference genome, "
					"you must run seed_generator first", flush = True)
		if num_seed == 0:
			raise CustomizedError("The number of seeds cannot be 0 when seeding with one reference genome.")
		return [0]*num_seed
	else:
		seeds = ["seed." + str(i) + ".vcf" for i in range(len(os.listdir(seeds_vcf_dir)))]
		for seed in seeds:
			# iterate through all seeds
			sum_trait = 0
			with open(os.path.join(seeds_vcf_dir, seed), "r") as seed_vcf:
				for line in seed_vcf:
					if not line.startswith("#"):
						fields = line.rstrip("\n").split("\t")
						mut_pos = int(fields[1])
						for _ , trait_info in dict_c_g.items():
							# iterate through all causal genes
							if trait_info[START_IDX] <= mut_pos <= trait_info[END_IDX]:
								sum_trait += trait_info[E_SIZE]
			seed_vals.append(sum_trait)

	return seed_vals

def normalization_by_mutscounts(wk_dir, dict_c_g, n_gen, mut_rate, pis_ref, matrix_, num_seed, use_subst_matrix=False, final_T=1):
	"""
	Normalize the effect sizes based on the expected trait values at the end of the simulation.
	
	Parameters:
		wk_dir (str): Working directory.
		dict_c_g (dict): A dictionary storing the genetic elements' effect sizes. 
						 Keys are gene names, values are lists of the form [start_pos(int), end_pos(int), effect_size(float)].
		n_gen (int): Number of generations to be simulated.
		mut_rate (float): Mutation rate.
		pis_ref (list): Allele frequencies of A, C, G, T in reference genome, [piA, piC, piG, piT]
		matrix_ (list os lists): mutation probability matrix
		use_subst_matrix: whether to use mutation probability matirx
		
	Returns:
		dict_c_g (dict): A normalized dictionary storing the genetic elements' effect sizes.
						 Same as input dict_c_g, but with the effect size values normalized.
	"""
	# Calculate the average trait value at the initialization
	avg_trait_value = statistics.mean(seeds_trait_calc(wk_dir, dict_c_g, num_seed))
	
	# Calculate the total sum for normalization
	if use_subst_matrix:
		r = (1-matrix_[0][0]) * pis_ref[0] + (1-matrix_[1][1]) * pis_ref[1] + (1-matrix_[2][2]) * pis_ref[2] + (1-matrix_[3][3]) * pis_ref[3]
		total_sum = avg_trait_value + sum((n_gen * r) * (end_pos - start_pos + 1) * effect_size
			for start_pos, end_pos, effect_size in dict_c_g.values())
	else:
		total_sum = avg_trait_value + sum(
			(n_gen * mut_rate) * (end_pos - start_pos + 1) * effect_size
			for start_pos, end_pos, effect_size in dict_c_g.values()
		)
	
	# Normalize the effect sizes
	k = final_T / total_sum
	for _ , gene_info in dict_c_g.items():
		gene_info[E_SIZE] *= k
	
	return dict_c_g

def generate_eff_vals(gff_, causal, es_low, es_high, wk_dir, n_gen, mut_rate, norm_or_not, pis_ref, matrix_, num_seed, use_subst_matrix=False, final_T=1):
	"""
	Randomly generate a set of effect sizes for some genes sampled from a GFF-like annotation file
	and computes the trait value of each seed according to the gff file.
	
	Parameters:
		gff_ (str): Full path to the GFF-like file.
		causal (int): Number of genetic elements to be generated.
		es_low (float): Lower bound of the effect size to be sampled.
		es_high (float): Upper bound of the effect size to be sampled.
		wk_dir (str): Working directory.
		n_gen (int): Number of generations to be simulated (used only when norm_or_not is True).
		mut_rate (float): Mutation rate (used only when norm_or_not is True).
		norm_or_not (bool): Specifies whether the generated effect sizes will be normalized.
		pis_ref (list): Allele frequencies of A, C, G, T in reference genome, [piA, piC, piG, piT]
		matrix_ (list os lists): mutation probability matrix
		use_subst_matrix: whether to use mutation probability matirx
		final_T: Expected average trait value by the end of the simulation
		
	Returns:
		dict_causal_genes (dict): A dictionary storing the genetic elements' effect sizes. 
								  Keys are gene names, values are lists of the form [start_pos(int), end_pos(int), effect_size(float)].
		seeds_trait_vals (list): A list of trait values for each seed given the genetic elements' effect sizes generated.
	"""

	# GFF: a tab-delimed file, with each row representing one genetic element,
	# the 4th column being the starting position, the 5th column being the ending position,
	# the 8th column being the info column, which is ,delimted by ";" and the 2nd term of which is 
	# the name of the gene (Index start from 0)

	# select genes from the gff file for causals
	n_genes = _count_gff_genes(gff_)
	# genes = sample(range(n_genes), causal)
	genes = np.random.choice(n_genes, causal, replace = False)

	dict_causal_genes = {}

	# read line by line
	with open(gff_, "r") as gff:
		index = 0
		for line in gff:
			# skip the file header
			if line.startswith("#") or line.startswith("\n") :
				continue		
			if index in genes:
				# parse the line to fill in the dict_causal_genes_dict
				fields = line.rstrip("\n").split("\t")
				info = dict(item.split("=") for item in fields[GFF_INFO].split(";"))
				dict_causal_genes[info['ID']] = [int(fields[GFF_START]), int(fields[GFF_END]), 
									 np.random.uniform(float(es_low), float(es_high))]
			index = index + 1
	
	if norm_or_not:
		# modify dict_causal_genes by the normalization factor k
		dict_causal_genes = normalization_by_mutscounts(wk_dir, dict_causal_genes, n_gen, mut_rate, pis_ref, matrix_, num_seed, use_subst_matrix, final_T)
	
	seeds_trait_vals = seeds_trait_calc(wk_dir, dict_causal_genes, num_seed)

	return (dict_causal_genes, seeds_trait_vals)


def write_eff_size_csv(trait_n, traits_dict, wk_dir):
	"""
	Writes effect sizes of genes into a CSV file.
	
	Parameters:
		trait_n (dict[str, int]): A dictionary containing the number of traits for transmissibility and drug resistance.
		traits_dict (list): A list containing effect sizes of genetic elements.
							Indices are trait IDs, elements are dictionaries where keys are gene names and values are lists
							containing start position, end position, and effect size.
		wk_dir (str): Working directory.
	"""
	# Initialize dictionaries to store information
	all_dict = {} # {gene: [start_idx, end_idx, effect size for each trait ...]}
	start_pos_dict = {} # {{start_idx: gene_id}}

	# Iterate over trait IDs
	for trait_id in range(sum(trait_n.values())):
		t_dict = traits_dict[trait_id] # Get the dictionary for the current trait
		for gene in t_dict: # Iterate over genes in the current trait dictionary
			if gene in all_dict:
				while (len(all_dict[gene]) < 2 + trait_id): # Ensure the effect size list has the correct length; 
					# 2 is for starting and ending position
					all_dict[gene].append(0) # 0 is place holder if the gene do not contribute to certain traits
				all_dict[gene].append(t_dict[gene][E_SIZE])
			else: # If the gene is not in the dictionary, initialize its effect size list
				all_dict[gene] = t_dict[gene][:2] # first attach the starting and ending position
				# Append zeros for traits before the current trait
				for i in range(trait_id):
					all_dict[gene].append(0)
				all_dict[gene].append(t_dict[gene][E_SIZE])
				start_pos_dict[all_dict[gene][START_IDX]] = gene
	
	# fill in 0s for genes irrelevant to a trait
	for gene in all_dict: # 2 is for starting and ending position
		while len(all_dict[gene]) < 2 + sum(trait_n.values()):
			all_dict[gene].append(0)


	sorted_start_pos = sorted(start_pos_dict.keys())
	# {ascending starting position: gene_id}
	sorted_dict = {i: start_pos_dict[i] for i in sorted_start_pos}
	sorted_all_dict = {gene: all_dict[gene] for _ , gene in sorted_dict.items()}

	# Write to the CSV file
	with open(os.path.join(wk_dir, "causal_gene_info.csv"), "w") as csv_file:
		# Write the header
		header_traits = [
			f"eff_size_transmissibility{i + 1}" for i in range(trait_n["transmissibility"])
		] + [
			f"eff_size_dr{i + 1}" for i in range(trait_n["drug_resistance"])
		]
		header_csv = "gene_name,start,end," + ",".join(header_traits)
		csv_file.write(header_csv + "\n")
		# Write gene information
		for gene in sorted_all_dict:
			gene_info = sorted_all_dict[gene]
			csv_file.write(f"{gene},{','.join(map(str, gene_info))}\n")


def write_seeds_trait(wk_dir, seeds_trait_vals, traits_num):
	"""
	Writes trait values of all seeds into a CSV file.
	
	Parameters:
		wk_dir (str): Working directory.
		seeds_trait_vals (list): A list of lists storing the trait value of each seed for each trait.
		traits_num (tuple/list): A tuple/list containing the number of traits for transmissibility and drug resistance.
	"""
	# n_trans_traits, n_drug_traits = traits_num
	n_trans_traits = traits_num["transmissibility"]
	n_drug_traits = traits_num["drug_resistance"]

	# Create the header string, traits are indexed from 1
	header = "seed_id,"
	if n_trans_traits > 0:
		header += ",".join(f"transmissibility_{i+1}" for i in range(n_trans_traits))
		if n_drug_traits > 0:
			header += ","
	header += ",".join(f"drugresist_{i+1}" for i in range(n_drug_traits))

	# Write to CSV file
	csv_file_path = os.path.join(wk_dir, "seeds_trait_values.csv")
	with open(csv_file_path, "w") as csv_file:
		csv_file.write(header + "\n")
		for i, seed_traits in enumerate(zip(*seeds_trait_vals)):
			csv_file.write(f"{i},{','.join(map(str, seed_traits))}\n")

def generate_effsize_csv(trait_n, causal_sizes, es_lows, es_highs, gff_, wk_dir, n_gen, mut_rate, norm_or_not, num_seed, use_subst_matrix, mu_matrix, ref, final_T):
	"""
	Generate a CSV file containing genetic element information for all traits and calculate seeds' trait values.

	Parameters:
		trait_n (list): Number of traits to generate for each category.
		causal_sizes (list): Number of genetic elements to generate for each trait.
		es_lows (list): Lower bounds of effect sizes to sample for each trait.
		es_highs (list): Upper bounds of effect sizes to sample for each trait.
		gff_ (str): Full path to the GFF-like file.
		wk_dir (str): Working directory.
		n_gen (int): Number of generations to simulate.
		mut_rate (float): Mutation rate.
		norm_or_not (bool): Whether to normalize the generated effect sizes.
		use_subst_matrix (bool): Whether to use mutation probability matrix
		mu_matrix (str): substitution probability matrix
		ref (str): path to the reference genome
		final_T: Expected average trait value by the end of the simulation

	"""
	total_n_traits = sum(trait_n.values())
	if len(causal_sizes) != total_n_traits:
		raise CustomizedError("The given length of the number of causal genetic elements "
						f"(-causal_size_each {len(causal_sizes)}) is not consistent with the "
						f"number of traits ({total_n_traits})")
	if len(es_lows) != total_n_traits:
		raise CustomizedError(f"The given length of the lower bounds (-es_low {len(es_lows)}) "
						f"is not consistent with the number of traits ({total_n_traits})")
	
	if len(es_highs) != total_n_traits:
		raise CustomizedError(f"The given length of the upper bounds (-es_high {len(es_highs)}) "
						f"is not consistent with the number of traits ({total_n_traits})")

	pis_ref = []
	matrix_ = []
	if norm_or_not:
		if type(n_gen) != int or n_gen <= 0:
			raise CustomizedError("Please specify a positive integer for generation "
						"(-sim_generation) in normalization mode")
		if use_subst_matrix==True:
			if os.path.exists(ref)==False:
				raise CustomizedError("Please provide a valid path to the reference genome "
							"(-ref) in normalization mode using substitution probability matrix")
			else:
				matrix_ = format_subst_mtx(mu_matrix, diag_zero=False)
				pis_ref = check_ref_format(ref)
		else:
			if (type(mut_rate) != float and type(mut_rate) != int) or mut_rate <= 0:
				raise CustomizedError("Please specify a positive number for mutation rate "
							"(-mut_rate) in normalization mode using single mutation rate")

	traits_dict =[]
	seeds_trait_vals = []
	# Generate effect sizes and calculate seeds' trait values for each trait
	for trait_id in range(sum(trait_n.values())):
		current_tdict, seed_vals = generate_eff_vals(gff_=gff_, causal=causal_sizes[trait_id], es_low=es_lows[trait_id], 
										 es_high=es_highs[trait_id], wk_dir=wk_dir, n_gen=n_gen, mut_rate=mut_rate, 
										 norm_or_not=norm_or_not, pis_ref=pis_ref, matrix_=matrix_, num_seed=num_seed, 
										 use_subst_matrix=use_subst_matrix, final_T=final_T)

		traits_dict.append(current_tdict)
		seeds_trait_vals.append(seed_vals)

	# Write seeds' trait values and genetic element information to CSV files
	write_seeds_trait(wk_dir, seeds_trait_vals, trait_n)
	write_eff_size_csv(trait_n, traits_dict, wk_dir)


def read_effvals(wk_dir, effsize_path, traits_num, num_seed = 0):
	"""
    Read a user-specified CSV file of effect size and return the seeds' trait values accordingly.

    Parameters:
        wk_dir (str): Working directory.
        effsize_path (str): Full path to the effect size file (CSV).
        traits_num (list): A list of two integers representing the number of traits in the file.
                           The first element represents the number of traits for the first category,
                           and the second element represents the number of traits for the second category.

    Returns:
        list: A list containing the trait values of seeds. Each element in the list represents
              the trait values of seeds for a specific trait.
	"""
	if effsize_path == "":
		raise CustomizedError("You need to specify a path to the effect size csv "
						"file (-effsize_path) in user_input mode.")
	
	if not os.path.exists(effsize_path):
		raise FileNotFoundError(f"Path to effect size file '{effsize_path}' not found")
	
	eff_df = pd.read_csv(effsize_path)
	num_cols = len(eff_df.columns)
	num_df_traits = num_cols - 3 # 0: gene_name, 1: start, 2: end

	traits_num_sum = traits_num["transmissibility"] + traits_num["drug_resistance"]
	if traits_num_sum != num_df_traits:
		raise ValueError(f"The sum of traits specified ({traits_num_sum}) does not match the number "
				   f"of traits in the file ({num_df_traits}).")
	
	gene_names = eff_df["gene_name"].tolist()
	start_pos = eff_df["start"].tolist()
	end_pos = eff_df["end"].tolist()

	seeds_trait_vals = []
	traits = []
	for idx in range(traits_num_sum):
		trait_col_idx = 3 + idx # The first three columns are meta information.
		trait_vals = eff_df.iloc[:, trait_col_idx].tolist()
		dict_gene_trait = {gene_names[j]: [start_pos[j], end_pos[j], trait_vals[j]] for j in range(len(gene_names))}
		seeds_trait_vals.append(seeds_trait_calc(wk_dir, dict_gene_trait, num_seed))
		traits.append(dict_gene_trait)
	# Question: I am a little confused by why we would want to write the CSV again; for column names?
	write_eff_size_csv(traits_num, traits, wk_dir)
	return seeds_trait_vals


def run_effsize_generation(method, wk_dir, effsize_path="", gff_in="", trait_n={}, causal_sizes=[], es_lows=[], es_highs=[], 
						   norm_or_not=False, n_gen=0, mut_rate=0, rand_seed = None, num_seed = 0, use_subst_matrix=False, 
						   mu_matrix="", ref="", final_T = 1):
	"""
	Generate effect sizes for genes and computes trait values for seeds' sequences.

	Parameters:
		method (str): The method used for generating effect sizes. Either "user_input" or "randomly_generate".
        wk_dir (str): Working directory.
        effsize_path (str, optional): Path to the effect size file (CSV)) for the user_input method.
        gff_in (str, optional): Path to the General Feature Format (GFF) file for randomly_generate method.
        trait_n (list[int], optional): A list containing two integers representing the number of traits for transmissibility and drug resistance.
        causal_sizes (list[int], optional): List of integers representing the number of causal genes for each trait.
        es_lows (list[int, float], optional): List of floats representing the lower bounds for effect sizes.
        es_highs (list[int, float], optional): List of floats representing the upper bounds for effect sizes.
        norm_or_not (bool, optional): Boolean indicating whether to normalize effect sizes.
        n_gen (int, optional): Number of generations for randomly_generate method.
        mut_rate (float or int, optional): Mutation rate for randomly_generate method.
		

	Returns:
		error_message (str): Error message.
	"""
	if rand_seed != None:
		np.random.seed(rand_seed)
	error_message = None
	try:
		if len(trait_n.keys()) != 2:
			raise CustomizedError("Please specify exactly 2 traits quantities in a list (-trait_n for transmissibility and drug resistance)")
		if sum(trait_n.values()) < 1:
			raise CustomizedError("Please provide a list of trait quantities (-trait_n) that sums up to at least 1")
		if method == "user_input":
			write_seeds_trait(wk_dir, read_effvals(wk_dir, effsize_path, trait_n, num_seed), trait_n)
		elif method == "randomly_generate":
			
			generate_effsize_csv(trait_n, causal_sizes, es_lows, es_highs, gff_in, wk_dir, n_gen, mut_rate, \
				norm_or_not, num_seed, use_subst_matrix, mu_matrix, ref, final_T)
		else:
			raise CustomizedError(f"{method} isn't a valid method. Please provide a permitted method. "
							"(user_input/randomly_generate)")
		print("******************************************************************** \n" +
				"                  GENETIC ARCHITECTURES GENERATED		            \n" +
				"******************************************************************** \n", flush = True)
	except Exception as e:
		print(f"Genetic effects generation - An error occured: {e}.", flush = True)
		error_message = e
	return error_message

def effsize_generation_byconfig(all_config):
	"""
    Generates effect size file and compute seeds' trait values based on a provided config file.

    Parameters:
        all_config (dict): A dictionary of the configuration (read with read_params()).
	"""

	genetic_config = all_config["GenomeElement"]
	wk_dir = all_config["BasicRunConfiguration"]["cwdir"]
	effsize_method = genetic_config["effect_size"]["method"]
	random_seed = all_config["BasicRunConfiguration"].get("random_number_seed", None)
	num_seed = all_config["SeedsConfiguration"]["seed_size"]
	subst_model_param = all_config["EvolutionModel"]["subst_model_parameterization"]
	if subst_model_param=="mut_rate":
		use_subst_matrix=True
	elif subst_model_param=="mut_rate_matrix":
		use_subst_matrix=False
	else:
		raise CustomizedError(f"The given subst_model_parameterization is NOT valid -- please input 'mut_rate' or 'mut_rate_matrix'.")
	mu_matrix_ori = all_config["EvolutionModel"]["burn_in_mutrate_matrix"]
	mu_matrix = {"A": mu_matrix_ori[0], "C": mu_matrix_ori[1], "G": mu_matrix_ori[2], "T": mu_matrix_ori[3]}

	eff_params_config = genetic_config["effect_size"]["randomly_generate"]
	error = run_effsize_generation(method=effsize_method, wk_dir=wk_dir, trait_n=genetic_config["traits_num"], 
						 effsize_path=genetic_config["effect_size"]["user_input"]["path_effsize_table"],
						 causal_sizes=eff_params_config["genes_num"], es_lows=eff_params_config["effsize_min"], 
						 es_highs=eff_params_config["effsize_max"], gff_in=eff_params_config["gff"], 
						 n_gen=all_config["EvolutionModel"]["n_generation"], mut_rate=all_config["EvolutionModel"]["mut_rate"], 
						 norm_or_not=eff_params_config["normalize"], rand_seed = random_seed, num_seed=num_seed,
						 use_subst_matrix=use_subst_matrix, mu_matrix=mu_matrix, ref=all_config["GenomeElement"]["ref_path"], 
						 final_T = eff_params_config["final_trait"])
	return error


def main():
	parser = argparse.ArgumentParser(description='Generate or modify seeds.')
	parser.add_argument('-method', action='store',dest='method', type=str, required=True, help="Method of the genetic element file generation")
	parser.add_argument('-wkdir', action='store',dest='wkdir', type=str, required=True, help="Working directory")
	parser.add_argument('-effsize_path', action='store',dest='effsize_path', type=str, required=False, help="Path to the user-provided effect size genetic element csv file", default="")
	parser.add_argument('-trait_n', action='store', dest='trait_n', type=str, required=True, help="Number of traits that user want to generate a genetic architecture for transmissibility and drug resistance, format: '{\"transmissibility\": x, \"drug-resistance\": y}'", default="")
	parser.add_argument('-causal_size_each','--causal_size_each', nargs='+', help='Size of causal genes for each trait', required=False, type=int, default=[])
	parser.add_argument('-es_low','--es_low', nargs='+', help='Lower bounds of effect size for each trait', required=False, type=float, default=[])
	parser.add_argument('-es_high','--es_high', nargs='+', help='Higher bounds of effect size for each trait', required=False, type=float, default=[])
	parser.add_argument('-gff', action='store',dest='gff', type=str, required=False, help='Path to the gff file', default="")
	parser.add_argument('-normalize','--normalize', default=False, required=False, type=str2bool, help='Whether to normalize the effect size based on sim_generations and mut_rate')
	parser.add_argument('-sim_generation', action='store',dest='sim_generation', required=False, type=int, default=0)
	parser.add_argument('-mut_rate', action='store',dest='mut_rate', required=False, type=float, default=0)
	parser.add_argument('-use_subst_matrix', action='store',dest='use_subst_matrix', type=str2bool, required=False, help="Whether to use a substitution probability matrix to parametrize mutations", default=False)
	parser.add_argument('-mu_matrix', action='store',dest='mu_matrix', type=str, required=False, help="JSON format string specifying the mutation probability matrix, required in SLiM burn-in", default="")
	parser.add_argument('-random_seed', action = 'store', dest = 'random_seed', required = False, type = int, default = None)
	parser.add_argument('-ref', action = 'store', dest = 'ref', required = False, type = str, default = "", help='Reference genome of the pathogen, required for normalization using substitution probability matrix.')
	parser.add_argument('-n_seed', action='store', dest = 'num_seed', required = False, type = int, default = None)
	parser.add_argument('-final_T', action='store', dest = 'final_T', required = False, type = float, default = 1, help='Expected average trait value at the end of the simulation, used in normalization mode. Default is 1.')


	args = parser.parse_args()
	method = args.method
	effsize_path = args.effsize_path
	gff_in = args.gff
	trait_n = args.trait_n
	trait_n = json.loads(trait_n)
	causal_sizes = args.causal_size_each
	es_lows = args.es_low
	es_highs = args.es_high
	wk_dir = args.wkdir
	n_gen = args.sim_generation
	mut_rate = args.mut_rate
	norm_or_not = args.normalize
	rand_seed = args.random_seed
	num_seed = args.num_seed
	use_subst_matrix = args.use_subst_matrix
	mu_matrix = args.mu_matrix
	ref = args.ref
	final_T = args.final_T

	run_effsize_generation(method=method, wk_dir=wk_dir, effsize_path=effsize_path, gff_in=gff_in, trait_n=trait_n, 
						causal_sizes=causal_sizes, es_lows=es_lows, es_highs=es_highs, norm_or_not=norm_or_not, 
						n_gen=n_gen, mut_rate=mut_rate, rand_seed = rand_seed, num_seed = num_seed,
						use_subst_matrix=use_subst_matrix, mu_matrix=mu_matrix, ref=ref, final_T = final_T)


if __name__ == "__main__":
	main()