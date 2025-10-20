### The main script to run the whole pipeline
import argparse, sys
from base_func import *
from network_generator import network_generation_byconfig
from seed_generator import seeds_generation_byconfig
from genetic_effect_generator import effsize_generation_byconfig
from seed_host_matcher import read_config_and_match
from outbreak_simulator import all_slim_simulation_by_config

header = r"""

            _____   ____    ___   __  __ 
      ___  |___ /  / ___|  |_ _| |  \/  |
     / _ \   |_ \  \___ \   | |  | |\/| |
    |  __/  ___) |  ___) |  | |  | |  | |
     \___| |____/  |____/  |___| |_|  |_|
                                      


"""


def main():
    parser = argparse.ArgumentParser(description='Run the whole simulation process by the configuration provided.')
    parser.add_argument('-path_config', action='store',dest='path_config', required=True, help="Path to config file (json)")

    args = parser.parse_args()

    config_path = args.path_config

    ## Read the parameters and store them into a dictionary
    param_dict = read_params(config_path, "user_config.json")

    print(header, flush = True)

    def _exit(error_m):
        if error_m != None:
            sys.exit("Simulation ends.")

    ## Network generation
    error = network_generation_byconfig(param_dict)
    _exit(error)

    ## Seeds generation
    error = seeds_generation_byconfig(param_dict)
    _exit(error)

    ## Effect size generation
    error = effsize_generation_byconfig(param_dict)
    _exit(error)

    ## Seed-host match
    error = read_config_and_match(param_dict)
    _exit(error)

    ## Simulation
    error = all_slim_simulation_by_config(param_dict)
    _exit(error)


if __name__ == "__main__":
    main()