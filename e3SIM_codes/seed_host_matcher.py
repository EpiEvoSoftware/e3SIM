import os, networkx as nx, json, pandas as pd, math, argparse, numpy as np
from collections import defaultdict
from pathlib import Path
from error_handling import CustomizedError
from base_func import *

# Magic numbers
HUNDRED = 100
ZERO = 0

def _build_dict_edges_node(ntwk_):
    """
    Returns a dictionary with degrees as keys and a list of nodes 
    of corresponding degrees as values (dict[int, list[int]]).

    Parameters:
        ntwk_ (nx.Graph): The contact network.
    """
    dict_edges_node = defaultdict(list)
    [dict_edges_node[n_edges].append(node) for node, n_edges in ntwk_.degree]
    return dict_edges_node

def _sort_node_by_edge(dict_edges_node):
    """
    Returns a list of nodes reversely sorted by their degress (list[int])
    and a list of their corresponding degrees (list[int]).

    Parameters:
        dict_edges_node (dict[int, list[int]]): A dictionary with degrees
        as keys and a list of nodes of corresponding degrees as values
    """
    nodes_sorted = []
    degree_sorted = []
    for degree in sorted(dict_edges_node.keys(), reverse = True):
        nodes = dict_edges_node[degree]
        nodes_sorted.extend(nodes)
        degree_sorted.extend([degree]*len(nodes))
    # degree_sorted is not used anywhere in the pipeline for now; 
    # we can delete it if we decide this is unnecessary
    return nodes_sorted, degree_sorted

def _save_dict_to_csv(dict_matching, file_path):
    """
    Writes a CSV that stores the matching dictionary with columns ['seed', 'host_id'].

    Parameters:
        dict_matching (dict[int, int]): A dictionary with seeds as keys and hosts as values
        file_path (str): Full path to write the desired CSV
    """
    # changed the order of columns, it used to be ['host_id', 'seed']
    df = pd.DataFrame(list(dict_matching.items()), columns = ['seed', 'host_id'])
    df.to_csv(file_path, index = False)
 
def _read_user_matchingfile_info_json(file):
    """
    Returns a dictionary of the matching and checks the user input JSON for matching.

    Parameters:
        file (str): Full path to the JSON
    """
    ## TODO: to check whether the contents of JSON satisfies the matching requirement
    return file

def _read_user_matchingfile_info_csv(file):
    """
    Returns a dictionary of the matching and checks the user input CSV for matching.

    Parameters:
        file (str): Full path to the CSV
    """
    ## TODO: to check whether the contents of CSV satisfies the matching requirement
    return file

def _percentile_to_index(percentile, node_per_percent):
    """
    Returns the two indices (int, int) corresponding to the given percentile ranges.

    Parameters: 
        percentile (list[int, int]): percentile range for the node selection.
        node_per_percent: The number of nodes equivalent to a percentile
    """
    if len(percentile) != 2:
        raise CustomizedError(f"The percentile range {percentile} is not a list of two element")
    l_per, h_per = percentile
    # check if the percentiles are int
    if type(l_per) != int or type(h_per) != int:
        raise CustomizedError(f"The percentile range {percentile} is not a list of integers, please reset the range")
    # check if the ends are out of bound
    if min(l_per, h_per) < ZERO or max(l_per, h_per) > HUNDRED:
        raise CustomizedError(f"The percentile range {percentile} is not within the closed 0-100 interval")
    # check the relationship between the lower and higher ends
    if h_per <= l_per:
        raise CustomizedError(f"The percentile range {percentile} is not a valid interval")
    # round the index to make sure those are integers
    low_idx = math.ceil(node_per_percent * l_per)
    high_idx = math.floor(node_per_percent * h_per)
    return low_idx, high_idx

def read_user_matchingfile(file_path):
    """
    Raises exceptions when user given file path is invalid/ format does not satisfy
    the matching file requirements. Returns a dictionary of matching where seeds are 
    keys and host_ids are values.

    Parameters:
        file path (str): Full path to the user matchingfile
    """
    file_path = Path(file_path)

    # check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Path to user defined matching file {file_path} not found")
    # check if the json file is a valid matching file
    if file_path.suffix.lower() == ".json":
        try:
            with open(file_path, 'r') as file:
                matching = json.load(file)
                return _read_user_matchingfile_info_json(matching)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {file_path}")
    # check if the csv file is a valid matching file
    elif file_path.suffix.lower() == ".csv":
        try:
            matching = pd.read_csv(file_path)
            return _read_user_matchingfile_info_csv(matching)
        except pd.errors.ParserError:
            raise ValueError(f"Invalid CSV format in {file_path}")
    raise ValueError("Host-seed matching file is not CSV or JSON")

def read_network(network_path):
    """
    Return a network (nx.Graph) written in the input path.

    Parameters:
        network_path (str): Full path to the network to read
    """
    network_path = Path(network_path)
    if not network_path.exists():
        raise FileNotFoundError(f"The provided networkX path '{network_path}' doesn't exist. "
                            "Please run network_generation first before running this script "
                              "and make sure the given working directory is correct.")
    return nx.read_adjlist(network_path, nodetype=int)

def match_random(nodes_sorted, taken_hosts, param = None):
    """
    Returns one random available host (int) to match one seed.

    Parameters:
        nodes_sorted (list[int]): Hosts sorted in reverse order by degrees.
        taken_hosts (list[int]): Hosts already matched to a seed.
    """
    available_host = list(set(nodes_sorted).difference(set(taken_hosts)))
    host = np.random.choice(available_host, 1)[0]
    return host

def match_ranking(nodes_sorted, taken_hosts, rank):
    """
    Returns one available host (int) of a specific rank (by degrees) to match one seed.

    Parameters:
        nodes_sorted (list[int]): Hosts sorted in reverse order by degrees.
        taken_hosts (list[int]): Hosts already matched to a seed.
    """
    ntwk_size = len(nodes_sorted)
    if type(rank) != int:
        raise CustomizedError(f"The provided rank {rank} of of type {type(rank)}, please provide an interger")
    if rank > ntwk_size:
        raise CustomizedError(f"Your provided ranking {rank} exceed host size {ntwk_size}")
    if rank < 1:
        raise CustomizedError(f"Your provided ranking {rank} is smaller than 1, while the rank starts from 1 (the highest rank)")
    host = nodes_sorted[rank - 1]
    if host in taken_hosts: 
        raise CustomizedError(f"Host of specified rank {rank} is already taken")
    return host

def match_percentile(nodes_sorted, taken_hosts, percentile):
    """
    Returns one available host (int) within the 
    given percentile range (by degrees) to match one seed.

    Parameters:
        nodes_sorted (list[int]): Hosts sorted in reverse order by degrees.
        taken_hosts (list[int]): Hosts already matched to a seed.
        percentile (list[int]): Range for host selection, can only be of length two.
    """
    node_per_percent = len(nodes_sorted) / HUNDRED
    low_idx, high_idx = _percentile_to_index(percentile, node_per_percent)
    if high_idx > low_idx:
        hosts_in_range = set(nodes_sorted[low_idx:high_idx])
        taken_hosts_in_range = hosts_in_range.intersection(taken_hosts)
        available_host = list(hosts_in_range.difference(taken_hosts_in_range))
        if available_host == []: 
            raise CustomizedError(f"There is no host left to match in the percentile {percentile}")
        host = np.random.choice(available_host, 1)[0]
        return host
    raise CustomizedError(f"There is no host to match in the range {percentile}%")
    
def write_match(match_dict, wk_dir):
    """
    Writes the matching to a CSV file sorted in ascending 
    order by host ids in the working directory.

    Parameters:
        match_dict (dict[int, int]): A dictionary of matching with seeds as keys and hosts as values.
        wk_dir (str): Full path to the working directory.
    """
    sorted_match = dict(sorted(match_dict.items(), key=lambda x:x[1]))
    match_path = os.path.join(wk_dir, "seed_host_match.csv")
    _save_dict_to_csv(sorted_match, match_path)
    return match_path

def match_all_hosts(ntwk_, match_method, param, num_seed):
    """
    Returns matching in the form of dictionary where keys are seeds and values are hosts.
    E.g., {0: 232, 1:256, 2:790, 3:4, 4:760} (dict[int, int])

    Parameters:
        ntwk_ (nx.Graph): Contact network.
        match_method (dict[int, str]): A dictionary of hosts as keys and matching method as values.
        param (dict[int, Union[int, list[int, int], None]]): A dictionary of hosts as k
        eys and matching parameters as values.
        num_seed (int): Number of seeds. 
    """
    ntwk_size = ntwk_.number_of_nodes()
    if num_seed > ntwk_size:
        raise CustomizedError(f"It is not allowed to match {num_seed} seeds to {ntwk_size} hosts. "
                        "Please reduce the number of seeds or increase the host population size")
    
    # Preprocess the network
    dict_edges_node = _build_dict_edges_node(ntwk_)
    nodes_sorted, _ = _sort_node_by_edge(dict_edges_node)
    taken_hosts_id = []

    # Gather the nodes by their matching method
    dict_method_seeds = {'ranking': [], 'percentile': [], 'random': []}

    for seed_id in range(num_seed):
        idx_method = match_method.get(str(seed_id))
        if idx_method not in [None, 'ranking', 'percentile', 'random']:
            raise CustomizedError("Please provide a valid matching method in ('ranking', "
                        f"'percentile', 'random') instead of {idx_method} for seed {seed_id}")
        elif idx_method != None:
            dict_method_seeds[idx_method].append(seed_id) 
        else: 
            dict_method_seeds["random"].append(seed_id)
    
    # Define the matching function by the method param and process methods in the specified order
    match_functions = {'ranking': match_ranking, 'percentile': match_percentile,'random': match_random}
    match_dict = {}
    for method in ['ranking', 'percentile', 'random']:
        match_function = match_functions[method]
        for seed_id in dict_method_seeds[method]:
            matched_host = match_function(nodes_sorted, taken_hosts_id, param.get(str(seed_id)))
            match_dict[seed_id] = matched_host
            taken_hosts_id.append(matched_host)
    return match_dict

def run_seed_host_match(method, wkdir, num_seed, path_matching="", match_scheme="", match_scheme_param = "", rand_seed = None):
    """
    Executes seed host matching process in command line and write the matching file 
    in the specified directory.

    Parameters:
        method (str): "user_input" for user-provided matching or "randomly_generate".
        wkdir (str): Full path to the workding directory.
        num_seed (int): Number of seeds for matching.
        path_matching (str): Full path to the user-provided matching file.
        match_scheme (str): JSON string of matching methods for seeds.
        match_scheme_param (str): JSON string of matching parameters for seeds.

    Returns:
        seed_vs_host (dict[int, int]): A dictionary that maps seeds to hosts.
        error_message (str): Error message.
    """
    if rand_seed != None:
        np.random.seed(rand_seed)		
    # Generate network path
    ntwk_path = os.path.join(wkdir, "contact_network.adjlist")
    seed_vs_host = None
    error_message = None
    # Process the parameters and save the matching results have we match all host
    try:
        match_path = None
        if method == "user_input":
            if path_matching=="":
                raise CustomizedError("Path to the user-provided matching file (-path_matching) "
                          "needs to be provided in user_provided mode")
            elif os.path.exists(path_matching) == False:
                raise CustomizedError("Path to the user-provided matching file (-path_matching) "
                          "doesn't exist")
            else: 
                read_user_matchingfile(path_matching)
        elif method == "randomly_generate":
            if match_scheme == "" and match_scheme_param == "":
                match_scheme = {seed_id: "random" for seed_id in range(num_seed)}
                match_scheme_param = {seed_id: None for seed_id in range(num_seed)}
            else:
                try:
                    match_scheme = json.loads(match_scheme)
                except json.decoder.JSONDecodeError:
                    raise CustomizedError(f"The matching methods {match_scheme} is "
                           "not a valid json format")
                try:
                    match_scheme_param = json.loads(match_scheme_param)
                except json.decoder.JSONDecodeError:
                    if list(set(match_scheme.values())) != ["random"]:
                        raise CustomizedError(f"The matching parameters {match_scheme_param} is "
                     "not a valid json format")
                    elif match_scheme_param == "": 
                        raise CustomizedError(f"Please specify the matching parameters (-match_scheme_param) for " \
                           "'ranking' or 'percentile'")
                    match_scheme_param = {seed_id: None for seed_id in range(num_seed)}
            seed_vs_host = match_all_hosts(ntwk_=read_network(ntwk_path), match_method = match_scheme, 
                               param = match_scheme_param, num_seed = num_seed)
            match_path = write_match(seed_vs_host, wkdir)
        else:
            raise CustomizedError(f"Please provide a permitted method (-method): "
                        f"user_input/randomly_generate instead of your current input {method}")
        print("******************************************************************** \n" +
              "                         SEEDS AND HOSTS MATCHED                     \n" +
              "********************************************************************")
        print("Seed host matching:", match_path)
    except Exception as e:
        print(f"Seed and host match - A error occured: {e}.")
        error_message = e

    return seed_vs_host, error_message

def read_config_and_match(config_all):
    """
    Write matching file given the configuration file.

    Parameters:
        file_path (str): Full path to the configuration file.
        num_seed (int): Number of seeds.
    """
    match_random_config = config_all["SeedHostMatching"]["randomly_generate"]
    match_method = match_random_config["match_scheme"]
    match_method_param = match_random_config["match_scheme_param"]
    path_matching = config_all["SeedHostMatching"]["user_input"]["path_matching"]
    method = config_all["SeedHostMatching"]['method']
    num_seed = config_all["SeedsConfiguration"]["seed_size"]
    rand_seed = config_all["BasicRunConfiguration"].get("random_number_seed", None)
    _, error = run_seed_host_match(method = method, wkdir = config_all["BasicRunConfiguration"]["cwdir"], 
                     num_seed = num_seed, path_matching = path_matching, 
    match_scheme = match_method, match_scheme_param = match_method_param, rand_seed = rand_seed)

    return error

def main():
    parser = argparse.ArgumentParser(description='Match seeds and hosts.')
    parser.add_argument('-method',  action='store',dest='method', type=str, required=True, 
                     help="Methods of the seed host matching")
    parser.add_argument('-wkdir', action='store',dest='wkdir', type=str, required=True, 
                     help="Working directory")
    parser.add_argument('-num_init_seq', action='store',dest='num_seed', type=int, required=True, 
                     help="Number of seeds to be matched.")
    ### optional parameters
    parser.add_argument('-path_matching',  action='store',dest='path_matching', type=str, 
                     required=False, help="Path to the user-provided matching file", default="")
    parser.add_argument('-match_scheme',  action='store',dest='match_scheme', type=str, 
                     required=False, help="Scheme of matching", default="")
    parser.add_argument('-match_scheme_param', action='store', dest='match_scheme_param', 
                     type=str, required=False, help="Matching parameters for each seed", default="")
    parser.add_argument('-random_seed', action = 'store', dest = 'random_seed', required = False, type = int, default = None)

    args = parser.parse_args()
    method = args.method
    wkdir = args.wkdir
    num_seed = args.num_seed
    path_matching = args.path_matching
    match_scheme = args.match_scheme
    match_scheme_param = args.match_scheme_param
    rand_seed = args.random_seed

    run_seed_host_match(method = method, wkdir = wkdir, num_seed = num_seed, path_matching = path_matching, 
                     match_scheme = match_scheme, match_scheme_param = match_scheme_param, rand_seed = rand_seed)

if __name__ == "__main__":
    main()