import networkx as nx, numpy as np, os, argparse
from base_func import *
from error_handling import CustomizedError

def write_network(ntwk_, wk_dir):
    """
    Writes a network to the working directory and returns the path to the network.
    The output file will be named contact_network.adjlist.

    Parameters:
        wk_dir (str): Working directory.
        ntwk_ (nx.Graph): Contact network.

    Returns:
        ntwk_path (str): Path to the network.
    """
    nx.write_adjlist(ntwk_, ntwk_path := os.path.join(wk_dir, "contact_network.adjlist"))
    return ntwk_path

def ER_generate(pop_size, p_ER):
    """
    Returns an Erdos_renyi graph with specified parameters.

    Parameters:
        p_ER (str): The probability of existing edges between two nodes.
        pop_size (int): Population size.
    """
    if not 0 < p_ER <= 1:
        raise CustomizedError("You need to specify a 0<p<=1 (-p_ER) in Erdos-Renyi graph")
    er_graph = nx.erdos_renyi_graph(pop_size, p_ER, seed = np.random)
    return er_graph

def fast_ER_generate(pop_size, p_ER):
    """
    Returns an Erdos_renyi graph faster than the vanilla implementation.

    Parameters:
        p_ER (str): The probability of existing edges between two nodes.
        pop_size (int): Population size.
    """
    if not 0 < p_ER <= 1:
        raise CustomizedError("You need to specify a 0<p<=1 (-p_ER) in Erdos-Renyi graph")
    er_graph = nx.fast_gnp_random_graph(pop_size, p_ER, seed = np.random)
    return er_graph

def rp_generate(pop_size, rp_size, p_within, p_between):
    """
    Returns a random partition graph with n groups, each group having a probability of
        within-group edge, and there is a global between-group edge probability.

    Parameters:
        rp_size (list[int]): The sizes of partitions.
        pop_size (int): Population size.
        p_within (list[float]): The probability of edges within each partition.
        p_between (float): The probability of edges among partitions.
    """
    block_size = len(rp_size)
    prob_size = len(p_within)

    # Check for validity of inputs
    if sum(rp_size) != pop_size:
        raise CustomizedError(
            f"Size of the partitions (-rp_size {rp_size}) has "
            f"to add up to the whole population size (-popsize {pop_size})"
        )
    if prob_size != block_size:
        raise CustomizedError(
            f"The number of partitions ({block_size}) does "
            "not match the number of given within partition connection "
            f"probabilities ({prob_size})"
        )
    if p_between == 0:
        print(
            "WARNING: You didn't specify a between partition connection probability (-p_between) "
            "or have set it to 0. This will lead to two completely isolated partitions.",
            flush=True,
        )

    # Construct contact probability matrix
    p = [[p_between for _ in range(block_size)] for _ in range(block_size)]
    for k in range(block_size):
        p[k][k] = p_within[k]

    # Generate random partition graph
    rp_graph = nx.stochastic_block_model(rp_size, p, seed = np.random)
    return rp_graph


def ba_generate(pop_size, m):
    """
    Returns a random graph using Barabasi-Albert preferential attachement.

    Parameters:
        pop_size (int): Population size.
        m (int): Number of edges to attach from a new node to existing nodes.
    """
    ba_graph = nx.barabasi_albert_graph(pop_size, m, seed = np.random)
    return ba_graph


def read_input_network(path_network, pop_size):
    """
    Checks the format of the input network and return it.

    Parameters:
        path_network (str): Path to the network file.
        pop_size (int): Population size.
    """
    if path_network == "":
        raise CustomizedError(
            f"You need to specify a path to the user-provided network (-path_network)"
        )
    elif not os.path.exists(path_network):
        raise FileNotFoundError(
            f"The provided network path ({path_network}) doesn't exist"
        )
    ntwk = nx.read_adjlist(path_network)
    if len(ntwk) != pop_size:
        raise CustomizedError(
            "The provided network doesn't have the same number of nodes "
            f"({len(ntwk)}) as the host population size ({pop_size}) specified."
        )

    return ntwk


def run_network_generation(
    pop_size,
    wk_dir,
    method,
    model="",
    path_network="",
    p_ER=0,
    rp_size=[],
    p_within=[],
    p_between=0,
    m=0,
    rand_seed=None,
):
    """
    Generates the contact network given parameters.

    Parameters:
        wk_dir (str): Working directory.
        path_network (str): Path to the network file.
        pop_size (int): Population size.
        method (str): Method to acquire the contact network.
        model (str): The network model to construct contact network.
        p_ER (float): param for ER graph.
        rp_size (float): param for RP graph.
        p_within (list[float]): param for RP graph.
        p_between (float): param for RP graph.
        m (int): param for BA graph.

    Returns:
        ntwk (nx.Graph): Generated network.
        error_message (str): Error message.
    """
    if rand_seed != None:
        np.random.seed(rand_seed)
    ntwk = None
    error_message = None
    try:
        if method == "user_input":
            # change function name
            ntwk = read_input_network(path_network, pop_size)

        elif method == "randomly_generate":
            if not model in ["ER", "RP", "BA"]:
                raise CustomizedError(
                    "Please specify a legit random graph model (-model) in random "
                    "generate mode. (Supported model: ER/RP/BA)"
                )

            if model == "ER":
                ntwk = fast_ER_generate(pop_size, p_ER)
            elif model == "RP":
                ntwk = rp_generate(pop_size, rp_size, p_within, p_between)
            elif model == "BA":
                ntwk = ba_generate(pop_size, m)
        else:
            raise CustomizedError(
                "Please provide a permitted method (user_input/randomly_generate)"
            )
        # No Error occur
        ntwk_path = write_network(ntwk, wk_dir)
        print(
            "******************************************************************** \n"
            + "                   CONTACT NETWORK GENERATED                         \n"
            + "********************************************************************",
            flush=True,
        )
        print("Contact network:", ntwk_path, flush=True)
    except Exception as e:
        print(f"Contact network generation - An error occured: {e}.", flush=True)
        error_message = e

    return ntwk, error_message


def network_generation_byconfig(all_config):
    ntwk_config = all_config["NetworkModelParameters"]
    wk_dir = all_config["BasicRunConfiguration"]["cwdir"]
    # shared params
    ntwk_method = ntwk_config["method"]
    pop_size = ntwk_config["host_size"]
    # user_input params
    path_network = ntwk_config["user_input"]["path_network"]
    # randomly generated network params
    model = ntwk_config["randomly_generate"]["network_model"]
    # ER param
    p_ER = ntwk_config["randomly_generate"]["ER"]["p_ER"]
    # RP params
    rp_params = ntwk_config["randomly_generate"]["RP"]
    rp_size = rp_params["rp_size"]
    p_within = rp_params["p_within"]
    p_between = rp_params["p_between"]
    # BA params
    ba_m = ntwk_config["randomly_generate"]["BA"]["ba_m"]

    # Random Number generator
    random_number_seed = all_config["BasicRunConfiguration"].get(
        "random_number_seed", None
    )

    _, error = run_network_generation(
        pop_size=pop_size,
        wk_dir=wk_dir,
        method=ntwk_method,
        path_network=path_network,
        model=model,
        p_ER=p_ER,
        p_within=p_within,
        p_between=p_between,
        rp_size=rp_size,
        m=ba_m,
        rand_seed=random_number_seed,
    )
    return error


def main():
    parser = argparse.ArgumentParser(
        description="Generate a contact network for the population \
                                     size specified and store it in the working directory as an adjacency list."
    )
    parser.add_argument(
        "-popsize", action="store", dest="popsize", required=True, type=int
    )
    parser.add_argument("-wkdir", action="store", dest="wkdir", required=True, type=str)
    parser.add_argument(
        "-method", action="store", dest="method", required=True, type=str
    )
    parser.add_argument(
        "-model", action="store", dest="model", required=False, type=str, default=""
    )
    parser.add_argument(
        "-path_network",
        action="store",
        dest="path_network",
        required=False,
        type=str,
        default="",
    )
    parser.add_argument(
        "-p_ER", action="store", dest="p_ER", required=False, type=float, default=0
    )
    parser.add_argument(
        "-rp_size",
        "--rp_size",
        nargs="+",
        help="Size of random partition graph groups",
        required=False,
        type=int,
        default=[],
    )
    parser.add_argument(
        "-p_within",
        "--p_within",
        nargs="+",
        help="probability of edges for different groups \
                        (descending order)",
        required=False,
        type=float,
        default=[],
    )
    parser.add_argument(
        "-p_between",
        action="store",
        dest="p_between",
        required=False,
        type=float,
        default=0,
    )
    parser.add_argument(
        "-m", action="store", dest="m", required=False, type=int, default=0
    )
    parser.add_argument(
        "-random_seed",
        action="store",
        dest="random_seed",
        required=False,
        type=int,
        default=None,
    )

    args = parser.parse_args()
    pop_size = args.popsize
    wk_dir = args.wkdir
    method = args.method
    model = args.model
    path_network = args.path_network
    p_ER = args.p_ER
    rp_size = args.rp_size
    p_within = args.p_within
    p_between = args.p_between
    m = args.m
    rand_seed = args.random_seed

    run_network_generation(
        pop_size=pop_size,
        wk_dir=wk_dir,
        method=method,
        model=model,
        path_network=path_network,
        p_ER=p_ER,
        rp_size=rp_size,
        p_within=p_within,
        p_between=p_between,
        m=m,
        rand_seed=rand_seed,
    )

if __name__ == "__main__":
    main()

