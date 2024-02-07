"""
An application for visualizing networks

This file is the main entry-point for the GUI application. 
"""
import argparse
import os.path
import tkinter as tk
from tkinter import ttk

from tabs.t1_evolutionary_model import EvoModel
from tabs.t2_seeds import Seeds
from tabs.t3_genome_effsize import GenomeEffSize
from tabs.t4_network_seedhost import NetworkGraphApp
from tabs.t5_epi_model import EpiModel

def load_config_as_dict(config_file):
    """
    Loads the configuration from a file into a dictionary.

    Parameters:
        config_file (str): The path to the configuration file

    Returns:
        dict: Dictionary containing the configuration settings.
    """
    try:
        config_dict = {}
        with open(config_file, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    config_dict[key.strip()] = value.strip()
        return config_dict
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found.")
        return None


def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog='cluster', description='Application to view GUI')
    parser.add_argument('--config_path', type=str,
                        help='path to the configuration file', default="codes/params.config")
    parser.add_argument('-v', '--view', action='store_true',
                        help='visualize network graph')
    return parser.parse_args()


def launch_gui(config_file):
    """
    Launches the gui application
    """

    root = tk.Tk()

    tab_parent = ttk.Notebook(root)
    tab1 = ttk.Frame(tab_parent)
    tab2 = ttk.Frame(tab_parent)
    tab3 = ttk.Frame(tab_parent)
    tab4 = ttk.Frame(tab_parent)
    tab5 = ttk.Frame(tab_parent)
    network_app = EvoModel(tab1, tab_parent, config_file)
    network_app = Seeds(tab2, tab_parent, config_file)
    network_app = GenomeEffSize(tab3, tab_parent, config_file)
    network_app = NetworkGraphApp(tab4, tab_parent, config_file)
    network_app = EpiModel(tab5, tab_parent, config_file)
    tab_parent.add(tab1, text="Evolutionary Model")
    tab_parent.add(tab2, text="Seeds")
    tab_parent.add(tab3, text="Genome Effect Size")
    tab_parent.add(tab4, text="Network Graph")
    tab_parent.add(tab5, text="Epidemiological Model")

    tab_parent.pack(expand=1, fill='both')

    root.mainloop()


def execute():
    """
    Executes the application, according to the command line arguments specified.
    """
    args = parse_args()
    config = load_config_as_dict(args.config_path)
    if config:
        launch_gui(config)
    else:
        print("A valid configuration file is required to run the application.")


execute()