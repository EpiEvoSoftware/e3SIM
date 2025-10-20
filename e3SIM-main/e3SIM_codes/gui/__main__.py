"""
An application for visualizing networks

This file is the main entry-point for the GUI application. 
"""
import argparse
import os.path
import shutil
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import sys

e3SIM_dir = os.path.join(os.path.dirname(__file__), '../')
if e3SIM_dir not in sys.path:
    sys.path.insert(0, e3SIM_dir)

from tabs.t1_configuration import Configuration
from tabs.t2_evolutionarymodel import EvolutionaryModel
from tabs.t3_networkmodel import NetworkModel
from tabs.t4_seeds_configuration import SeedsConfiguration
from tabs.t5_genome_element import GenomeElement
from tabs.t6_seedhostmatch import HostMatch
from tabs.t7_epidemiology_model import EpidemiologyModel
from tabs.t8_post_processing import PostProcessing


def launch_gui(user_config_path, hide=False):
    """
    Launches the gui application
    """
    default_config_path = os.path.join(e3SIM_dir, "config_template/default_config.json")
    if os.path.samefile(user_config_path, os.path.join(e3SIM_dir, "config_template/user_config.json")):
        shutil.copy(default_config_path, user_config_path)
    config_path = user_config_path

    root = tk.Tk()
    root.iconbitmap('')
    root.title("e3SIM: Simulation Framework for Genetic Epidemiology")
    root.configure(background='black')
    tab_parent = ttk.Notebook(root)
    tab_parent.pack(expand=1, fill='both')

    style = ttk.Style(root)
    if 'aqua' in style.theme_names():
        style.theme_use('aqua')
    
     
    default_font = tkFont.nametofont("TkDefaultFont")
    font_family = default_font.cget("family")
    font_size = default_font.cget("size")
    style.configure("Bold.TLabel", font=(font_family, font_size, 'bold'))
    style.configure("Title.TLabel", font=(font_family, font_size+4, 'bold'))

    tab1 = ttk.Frame(tab_parent)
    tab2 = ttk.Frame(tab_parent)
    tab3 = ttk.Frame(tab_parent)
    tab4 = ttk.Frame(tab_parent)
    tab5 = ttk.Frame(tab_parent)
    tab6 = ttk.Frame(tab_parent)
    tab7 = ttk.Frame(tab_parent)
    tab8 = ttk.Frame(tab_parent)

    network_app = Configuration(tab1, tab_parent, config_path, "Basic Configuration", 0, hide)
    network_app = EvolutionaryModel(tab2, tab_parent, config_path, "Evolutionary Model", 1, hide)
    network_app = NetworkModel(tab3, tab_parent, config_path, "Network Model Parameters", 2, hide )
    network_app = SeedsConfiguration(tab4, tab_parent, config_path, "Seeds Configuration", 3, hide )
    network_app = GenomeElement(tab5, tab_parent, config_path, "Genome Element", 4, hide )
    network_app = HostMatch(tab6, tab_parent, config_path, "Seed Host Match", 5, hide )
    network_app = EpidemiologyModel(tab7, tab_parent, config_path, "Epidemiology Model", 6, hide )
    network_app = PostProcessing(tab8, tab_parent, config_path, "Post Processing Options", 7, hide)

    root.mainloop()

def execute():
    """
    Executes the application, according to the command line arguments specified.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(prog='cluster', description='Application to view GUI')
    parser.add_argument(
        '--config_path', type=str,default=os.path.join(e3SIM_dir, "config_template/user_config.json"), 
        help='path to the configuration JSON file')
    parser.add_argument(
        '--hide', action='store_true', 
        help='Whether to hide later tabs to enforce sequential logic flow throughout the gui. '
            'False by default.')
    args=parser.parse_args()

    launch_gui(args.config_path, args.hide)

execute()
