import os
import argparse
import networkx as nx
import numpy as np
from error_handling import CustomizedError


# ---------------- Base Class ---------------- #

class BaseNetworkGenerator:
    """Base class for network generators."""

    def __init__(self, pop_size, rand_seed=None):
        self.pop_size = pop_size
        self.rand_seed = rand_seed
        if rand_seed is not None:
            np.random.seed(rand_seed)

    def generate(self):
        raise NotImplementedError("Subclasses must implement generate()")


# ---------------- Concrete Implementations ---------------- #

class ERGenerator(BaseNetworkGenerator):
    def __init__(self, pop_size, p_ER, rand_seed=None):
        super().__init__(pop_size, rand_seed)
        if not 0 < p_ER <= 1:
            raise CustomizedError("You need to specify a 0<p<=1 (-p_ER) in Erdos-Renyi graph")
        self.p_ER = p_ER

    def generate(self):
        return nx.fast_gnp_random_graph(self.pop_size, self.p_ER, seed=np.random)


class RPGenerator(BaseNetworkGenerator):
    def __init__(self, pop_size, rp_size, p_within, p_between, rand_seed=None):
        super().__init__(pop_size, rand_seed)
        if sum(rp_size) != pop_size:
            raise CustomizedError(f"Partition sizes {rp_size} must sum to population size {pop_size}")
        if len(p_within) != len(rp_size):
            raise CustomizedError("Number of partitions and within-group probabilities mismatch")
        if p_between == 0:
            print("WARNING: Between-group probability is 0. Partitions will be isolated.")
        self.rp_size = rp_size
        self.p_within = p_within
        self.p_between = p_between

    def generate(self): # Thought it is called random partition, but can be any number of groups
        block_size = len(self.rp_size)
        prob_matrix = [[self.p_between for _ in range(block_size)] for _ in range(block_size)]
        for k in range(block_size):
            prob_matrix[k][k] = self.p_within[k]
        return nx.stochastic_block_model(self.rp_size, prob_matrix, seed=np.random)


class BAGenerator(BaseNetworkGenerator):
    def __init__(self, pop_size, m, rand_seed=None):
        super().__init__(pop_size, rand_seed)
        self.m = m

    def generate(self):
        return nx.barabasi_albert_graph(self.pop_size, self.m, seed=np.random)


class UserInputGenerator(BaseNetworkGenerator):
    def __init__(self, pop_size, path_network, rand_seed=None):
        super().__init__(pop_size, rand_seed)
        self.path_network = path_network

    def generate(self):
        if not self.path_network or not os.path.exists(self.path_network):
            raise FileNotFoundError(f"Network path {self.path_network} not found")
        ntwk = nx.read_adjlist(self.path_network)
        if len(ntwk) != self.pop_size:
            raise CustomizedError(f"Network nodes {len(ntwk)} do not match population size {self.pop_size}")
        return ntwk


# ---------------- Factory ---------------- #

class NetworkFactory:
    """Factory to instantiate network generators."""

    @staticmethod
    def create(method, model="", **kwargs):
        """
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
        """
        if method == "user_input":
            return UserInputGenerator(kwargs["pop_size"], kwargs["path_network"], kwargs.get("rand_seed"))

        elif method == "randomly_generate":
            if model == "ER":
                return ERGenerator(kwargs["pop_size"], kwargs["p_ER"], kwargs.get("rand_seed"))
            elif model == "RP":
                return RPGenerator(kwargs["pop_size"], kwargs["rp_size"], kwargs["p_within"], kwargs["p_between"], kwargs.get("rand_seed"))
            elif model == "BA":
                return BAGenerator(kwargs["pop_size"], kwargs["m"], kwargs.get("rand_seed"))
            else:
                raise CustomizedError("Supported models: ER, RP, BA")

        else:
            raise CustomizedError("Permitted methods: user_input / randomly_generate")


# ---------------- Manager ---------------- #

class NetworkManager:
    """Handles generation, saving, and error management."""

    def __init__(self, wk_dir):
        self.wk_dir = wk_dir

    def write_network(self, ntwk):
        ntwk_path = os.path.join(self.wk_dir, "contact_network.adjlist")
        nx.write_adjlist(ntwk, ntwk_path)
        return ntwk_path

    def run(self, method, model="", **kwargs):
        try:
            generator = NetworkFactory.create(method, model, **kwargs)
            ntwk = generator.generate()
            path = self.write_network(ntwk)
            print(
                "********************************************************************\n"
                "                   CONTACT NETWORK GENERATED\n"
                "********************************************************************",
                flush=True,
            )
            print("Contact network:", path, flush=True)
            return ntwk, None
        except Exception as e:
            print(f"Error during network generation: {e}")
            return None, e


# ---------------- Config-based Interface ---------------- #

def network_generation_byconfig(all_config):
    """
    Generate network based on config dict structure (same as old implementation).
    """
    ntwk_config = all_config["NetworkModelParameters"]
    wk_dir = all_config["BasicRunConfiguration"]["cwdir"]

    # Shared params
    ntwk_method = ntwk_config["method"]
    pop_size = ntwk_config["host_size"]

    # User input params
    path_network = ntwk_config["user_input"]["path_network"]

    # Random generation params
    model = ntwk_config["randomly_generate"]["network_model"]

    # ER
    p_ER = ntwk_config["randomly_generate"]["ER"]["p_ER"]

    # RP
    rp_params = ntwk_config["randomly_generate"]["RP"]
    rp_size = rp_params["rp_size"]
    p_within = rp_params["p_within"]
    p_between = rp_params["p_between"]

    # BA
    ba_m = ntwk_config["randomly_generate"]["BA"]["ba_m"]

    # RNG seed
    random_number_seed = all_config["BasicRunConfiguration"].get("random_number_seed", None)

    manager = NetworkManager(wk_dir)
    _, error = manager.run(
        method=ntwk_method,
        model=model,
        pop_size=pop_size,
        path_network=path_network,
        p_ER=p_ER,
        rp_size=rp_size,
        p_within=p_within,
        p_between=p_between,
        m=ba_m,
        rand_seed=random_number_seed,
    )
    return error


# ---------------- CLI ---------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate a contact network\
                                     population size specified and store it in \
                                     the working directory as an adjacency list.")
    parser.add_argument("-popsize", type=int, required=True)
    parser.add_argument("-wkdir", type=str, required=True)
    parser.add_argument("-method", type=str, required=True, choices=["user_input", "randomly_generate"])
    parser.add_argument("-model", type=str, default="")
    parser.add_argument("-path_network", type=str, default="")
    parser.add_argument("-p_ER", type=float, default=0)
    parser.add_argument("-rp_size", nargs="+", help = "Size of random partition graph groups", type=int, default=[])
    parser.add_argument("-p_within", nargs="+", help = "Probability of edges for different groups \
                         (decending order)", type=float, default=[])
    parser.add_argument("-p_between", type=float, default=0)
    parser.add_argument("-m", type=int, default=0)
    parser.add_argument("-random_seed", type=int, default=None)

    args = parser.parse_args()

    manager = NetworkManager(args.wkdir)
    manager.run(
        method=args.method,
        model=args.model,
        pop_size=args.popsize,
        path_network=args.path_network,
        p_ER=args.p_ER,
        rp_size=args.rp_size,
        p_within=args.p_within,
        p_between=args.p_between,
        m=args.m,
        rand_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
