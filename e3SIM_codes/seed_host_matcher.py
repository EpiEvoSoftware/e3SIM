import os
import networkx as nx
import json
import pandas as pd
import math
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

# Assuming these exist in your project
from error_handling import CustomizedError
from base_func import *

# Magic numbers
HUNDRED = 100
ZERO = 0


class NetworkPreprocessor:
    """Handles network preprocessing operations with NumPy optimization."""
    
    def __init__(self, network: nx.Graph):
        self.network = network
        self._sorted_nodes = None
        self._sorted_degrees = None
    
    def get_sorted_nodes(self) -> np.ndarray:
        """Get nodes sorted by degree (descending) - lazy loaded for performance."""
        if self._sorted_nodes is None:
            self._compute_sorted_data()
        return self._sorted_nodes
    
    def get_sorted_degrees(self) -> np.ndarray:
        """Get degrees sorted (descending) - lazy loaded for performance."""
        if self._sorted_degrees is None:
            self._compute_sorted_data()
        return self._sorted_degrees
    
    def _compute_sorted_data(self):
        """Compute sorted nodes and degrees using NumPy for better performance."""
        # Get all nodes and their degrees as NumPy arrays
        nodes = np.array(list(self.network.nodes()))
        degrees = np.array([self.network.degree[node] for node in nodes])
        
        # Sort by degrees in descending order (much faster than manual sorting)
        sort_indices = np.argsort(-degrees)
        
        self._sorted_nodes = nodes[sort_indices]
        self._sorted_degrees = degrees[sort_indices]


class HostMatcher:
    """Handles different host matching strategies."""
    
    def __init__(self, preprocessor: NetworkPreprocessor):
        self.preprocessor = preprocessor
        self.taken_hosts = set()
    
    def match_random(self, param = None) -> int:
        """Match to a random available host."""
        available_hosts = self._get_available_hosts(self.preprocessor.get_sorted_nodes())
        if len(available_hosts) == 0:
            raise CustomizedError("No available hosts for random matching")
        
        host = np.random.choice(available_hosts)
        self.taken_hosts.add(host)
        return host
    
    def match_ranking(self, rank: int) -> int:
        """Match to host with specific degree rank."""
        if not isinstance(rank, int):
            raise CustomizedError(f"Rank must be integer, got {type(rank)}")
        
        sorted_nodes = self.preprocessor.get_sorted_nodes()
        network_size = len(sorted_nodes)
        
        if rank > network_size:
            raise CustomizedError(f"Rank {rank} exceeds network size {network_size}")
        if rank < 1:
            raise CustomizedError(f"Rank {rank} must be >= 1")
        
        host = sorted_nodes[rank - 1]
        if host in self.taken_hosts:
            raise CustomizedError(f"Host at rank {rank} is already taken")
        
        self.taken_hosts.add(host)
        return host
    
    def match_percentile(self, percentile: List[int]) -> int:
        """Match to host within percentile range."""
        if not isinstance(percentile, list) or len(percentile) != 2:
            raise CustomizedError("Percentile must be list of 2 integers")
        
        low_per, high_per = percentile
        if not all(isinstance(p, int) for p in percentile):
            raise CustomizedError("Percentile values must be integers")
        
        if min(percentile) < 0 or max(percentile) > 100:
            raise CustomizedError("Percentile values must be between 0 and 100")
        
        if high_per <= low_per:
            raise CustomizedError("Invalid percentile range")
        
        # Calculate indices using NumPy
        sorted_nodes = self.preprocessor.get_sorted_nodes()
        network_size = len(sorted_nodes)
        node_per_percent = network_size / HUNDRED
        
        low_idx = math.ceil(node_per_percent * low_per)
        high_idx = math.floor(node_per_percent * high_per)
        
        if high_idx <= low_idx:
            raise CustomizedError(f"No hosts available in percentile range {percentile}")
        
        # Use NumPy slicing for efficiency
        hosts_in_range = sorted_nodes[low_idx:high_idx]
        available_hosts = self._get_available_hosts(hosts_in_range)
        
        if len(available_hosts) == 0:
            raise CustomizedError(f"No available hosts in percentile range {percentile}")
        
        host = np.random.choice(available_hosts)
        self.taken_hosts.add(host)
        return host
    
    def _get_available_hosts(self, candidates: np.ndarray) -> np.ndarray:
        """Get available hosts from candidates using NumPy operations."""
        if len(self.taken_hosts) == 0:
            return candidates
        
        # Convert to NumPy array for efficient set operations
        taken_array = np.array(list(self.taken_hosts))
        return np.setdiff1d(candidates, taken_array)


class SeedHostMatcher:
    """Main class for matching seeds to hosts."""
    
    def __init__(self, network: nx.Graph):
        self.network = network
        self.preprocessor = NetworkPreprocessor(network)
        self.matcher = HostMatcher(self.preprocessor)
    
    def match_all_seeds(self, match_methods: Dict[str, str], 
                       match_params: Dict[str, Any], 
                       num_seeds: int) -> Dict[int, int]:
        """
        Match all seeds to hosts using specified methods and parameters.
        
        Args:
            match_methods: Dict mapping seed_id (as string) to method name
            match_params: Dict mapping seed_id (as string) to method parameters
            num_seeds: Total number of seeds to match
            
        Returns:
            Dictionary mapping seed_id to host_id
        """
        network_size = self.network.number_of_nodes()
        if num_seeds > network_size:
            raise CustomizedError(f"Cannot match {num_seeds} seeds to {network_size} hosts")
        
        # Group seeds by method for efficient processing
        method_groups = self._group_seeds_by_method(match_methods, num_seeds)
        
        # Process in specific order: ranking, percentile, random
        seed_to_host = {}
        processing_order = ['ranking', 'percentile', 'random']
        
        for method in processing_order:
            if method not in method_groups:
                continue
            
            for seed_id in method_groups[method]:
                param = match_params.get(str(seed_id))
                host = self._match_single_seed(method, param)
                seed_to_host[int(seed_id)] = host
        
        return seed_to_host
    
    def _match_single_seed(self, method: str, param: Any) -> int:
        """Match a single seed using the specified method."""
        if method == 'random':
            return self.matcher.match_random(param)
        elif method == 'ranking':
            return self.matcher.match_ranking(param)
        elif method == 'percentile':
            return self.matcher.match_percentile(param)
        else:
            available_methods = ['random', 'ranking', 'percentile']
            raise CustomizedError(f"Unknown method '{method}'. Available: {available_methods}")
    
    def _group_seeds_by_method(self, match_methods: Dict[str, str], 
                              num_seeds: int) -> Dict[str, List[int]]:
        """Group seeds by their matching method."""
        method_groups = defaultdict(list)
        available_methods = ['random', 'ranking', 'percentile']
        
        for seed_id in range(num_seeds):
            method = match_methods.get(str(seed_id), 'random')
            
            if method not in available_methods:
                raise CustomizedError(f"Invalid method '{method}' for seed {seed_id}")
            
            method_groups[method].append(seed_id)
        
        return dict(method_groups)


class FileManager:
    """Handles file operations for matching data."""
    
    @staticmethod
    def save_matching_csv(seed_to_host: Dict[int, int], file_path: str):
        """Save matching to CSV file, sorted by host_id."""
        sorted_items = sorted(seed_to_host.items(), key=lambda x: x[1])
        df = pd.DataFrame(sorted_items, columns=['seed', 'host_id'])
        df.to_csv(file_path, index=False)
    
    @staticmethod
    def read_network(network_path: str) -> nx.Graph:
        """Read network from file."""
        network_path = Path(network_path)
        if not network_path.exists():
            raise CustomizedError(
                f"Network file '{network_path}' not found. "
                "Please run network generation first."
            )
        return nx.read_adjlist(network_path, nodetype=int)
    
    @staticmethod
    def read_user_matching_file(file_path: str) -> Dict[int, int]:
        """Read user-provided matching file (JSON or CSV)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise CustomizedError(f"Matching file {file_path} not found")
        
        if file_path.suffix.lower() == ".json":
            try:
                with open(file_path, 'r') as file:
                    return {int(k): v for k, v in json.load(file).items()}
            except json.JSONDecodeError as e:
                raise CustomizedError(f"Invalid JSON in {file_path}: {e}")
        
        elif file_path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(file_path)
                return dict(zip(df['seed'], df['host_id']))
            except Exception as e:
                raise CustomizedError(f"Invalid CSV in {file_path}: {e}")
        
        else:
            raise CustomizedError("Matching file must be JSON or CSV")


class MatchingOrchestrator:
    """Main orchestrator for the matching process."""
    
    def __init__(self, working_dir: str, random_seed: Optional[int] = None):
        self.working_dir = working_dir
        np.random.seed(random_seed)
    
    def run_matching(self, method: str, num_seeds: int, 
                    match_scheme: Union[str, Dict] = "", 
                    match_scheme_param: Union[str, Dict] = "",
                    path_matching: str = "") -> Tuple[Optional[Dict[int, int]], Optional[Exception]]:
        """Run the complete matching process."""
        try:
            if method == "user_input":
                return self._handle_user_input(path_matching)
            elif method == "randomly_generate":
                return self._handle_random_generation(num_seeds, match_scheme, match_scheme_param)
            else:
                raise CustomizedError(f"Invalid method '{method}'. Use 'user_input' or 'randomly_generate'")
        
        except Exception as e:
            print(f"Matching error: {e}")
            return None, e
    
    def _handle_user_input(self, path_matching: str) -> Tuple[Dict[int, int], None]:
        """Handle user-provided matching file."""
        if not path_matching:
            raise CustomizedError("Path to matching file required for user_input method")
        
        matching = FileManager.read_user_matching_file(path_matching)
        return matching, None
    
    def _handle_random_generation(self, num_seeds: int, 
                                 match_scheme: Union[str, Dict], 
                                 match_scheme_param: Union[str, Dict]) -> Tuple[Dict[int, int], None]:
        """Handle random matching generation."""
        # Parse parameters
        match_methods = self._parse_match_scheme(match_scheme, num_seeds)
        match_params = self._parse_match_params(match_scheme_param, match_methods, num_seeds)
        
        # Load network and perform matching
        network_path = os.path.join(self.working_dir, "contact_network.adjlist")
        network = FileManager.read_network(network_path)
        
        matcher = SeedHostMatcher(network)
        seed_to_host = matcher.match_all_seeds(match_methods, match_params, num_seeds)
        
        # Save results
        output_path = os.path.join(self.working_dir, "seed_host_match.csv")
        FileManager.save_matching_csv(seed_to_host, output_path)
        
        print("********************************************************************")
        print("                     SEEDS AND HOSTS MATCHED                       ")
        print("********************************************************************")
        print(f"Matching saved to: {output_path}")
        
        return seed_to_host, None
    
    def _parse_match_scheme(self, match_scheme: Union[str, Dict], num_seeds: int) -> Dict[str, str]:
        """Parse match scheme from string or dict."""
        if isinstance(match_scheme, dict):
            return match_scheme
        elif match_scheme == "":
            return {str(i): "random" for i in range(num_seeds)}
        else:
            try:
                return json.loads(match_scheme)
            except json.JSONDecodeError:
                raise CustomizedError("Invalid JSON format for match_scheme")
    
    def _parse_match_params(self, match_scheme_param: Union[str, Dict], 
                           match_methods: Dict[str, str], num_seeds: int) -> Dict[str, Any]:
        """Parse match parameters from string or dict."""
        if isinstance(match_scheme_param, dict):
            return match_scheme_param
        elif match_scheme_param == "":
            return {str(i): None for i in range(num_seeds)}
        else:
            try:
                return json.loads(match_scheme_param)
            except json.JSONDecodeError:
                # If all methods are random, params can be None
                if all(method == "random" for method in match_methods.values()):
                    return {str(i): None for i in range(num_seeds)}
                else:
                    raise CustomizedError("Invalid JSON format for match_scheme_param")


# Maintain backward compatibility with original functions
def match_all_hosts(ntwk_, match_method, param, num_seed):
    """Original function signature for backward compatibility."""
    matcher = SeedHostMatcher(ntwk_)
    return matcher.match_all_seeds(match_method, param, num_seed)


def write_match(match_dict, wk_dir):
    """Original function signature for backward compatibility."""
    match_path = os.path.join(wk_dir, "seed_host_match.csv")
    FileManager.save_matching_csv(match_dict, match_path)
    return match_path


def read_network(network_path):
    """Original function signature for backward compatibility."""
    return FileManager.read_network(network_path)


def read_user_matchingfile(file_path):
    """Original function signature for backward compatibility."""
    return FileManager.read_user_matching_file(file_path)


def run_seed_host_match(method: str, wkdir: str, num_seed: int, 
                       path_matching: str = "", match_scheme: str = "", 
                       match_scheme_param: str = "", rand_seed: Optional[int] = None):
    """Main function maintaining original API."""
    orchestrator = MatchingOrchestrator(wkdir, rand_seed)
    return orchestrator.run_matching(method, num_seed, match_scheme, match_scheme_param, path_matching)
#########################################################


def read_config_and_match(config_all):
    """Original function for config-based matching."""
    match_config = config_all["SeedHostMatching"]
    basic_config = config_all["BasicRunConfiguration"]
    
    orchestrator = MatchingOrchestrator(
        basic_config["cwdir"], 
        basic_config.get("random_number_seed")
    )
    
    _, error = orchestrator.run_matching(
        method=match_config['method'],
        num_seeds=config_all["SeedsConfiguration"]["seed_size"],
        match_scheme=match_config["randomly_generate"]["match_scheme"],
        match_scheme_param=match_config["randomly_generate"]["match_scheme_param"],
        path_matching=match_config["user_input"]["path_matching"]
    )
    
    return error


def main():
    """Command-line interface maintaining original structure."""
    parser = argparse.ArgumentParser(description='Match seeds and hosts.')
    parser.add_argument('-method', required=True, help="Matching method")
    parser.add_argument('-wkdir', required=True, help="Working directory")  
    parser.add_argument('-num_init_seq', required=True, type=int, help="Number of seeds")
    parser.add_argument('-path_matching', default="", help="User matching file path")
    parser.add_argument('-match_scheme', default="", help="Matching scheme JSON")
    parser.add_argument('-match_scheme_param', default="", help="Matching parameters JSON")
    parser.add_argument('-random_seed', type=int, help="Random seed")
    
    args = parser.parse_args()

    orchestrator = MatchingOrchestrator(
        args.wkdir, 
        args.random_seed
    )
    
    orchestrator.run_matching(
        method = args.method,
        num_seeds = args.num_init_seq,
        match_scheme = args.match_scheme,
        match_scheme_param = args.match_scheme_param,
        path_matching = args.path_matching
    )


if __name__ == "__main__":
    main()