import tkinter as tk
import os
from tkinter import ttk, messagebox, filedialog
from utils import render_next_button, load_config_as_dict, save_config, CreateToolTip
from base_func import check_ref_format


class Configuration:
    def __init__(self, parent, tab_parent, config_path, tab_title, tab_index, hide=False):
        self.config_path = config_path
        initial_config = load_config_as_dict(self.config_path)
        self.cwd = initial_config["BasicRunConfiguration"]["cwdir"]
        self.n_replicates = initial_config["BasicRunConfiguration"]["n_replicates"]
        self.random_number_seed = initial_config["BasicRunConfiguration"]["random_number_seed"]
        self.ref_path = initial_config["GenomeElement"]["ref_path"]
        self.parent = parent
        self.tab_parent = tab_parent
        self.tab_index = tab_index

        self.tab_parent.add(self.parent, text=tab_title)
        self.control_frame = ttk.Frame(self.parent, width=300)
        self.control_frame.pack(padx=10, pady=10)

        self.render_working_directory()
        self.render_ref_path_control()
        self.render_n_replicates()
        self.render_random_seed()
        render_next_button(self.tab_index, self.tab_parent, self.parent, update_config=self.update_config)
        print("********************************************************************")
        print("              YOU ARE RUNNING THE GUI OF E3SIM.")
        print("          PLEASE FOLLOW THE INSTRUCTIONS IN THE GUI.")
        print("********************************************************************")
    
    def render_working_directory(self):
        def choose_directory():
            dir_inp = filedialog.askdirectory(title="Select a Directory", initialdir=os.getcwd())
            if dir_inp:
                self.cwd = dir_inp
                #print(self.cwd)
                self.user_working_directory_label.config(text=f"{self.cwd}")

                config = load_config_as_dict(self.config_path)
                config["BasicRunConfiguration"]["cwdir"] = self.cwd
                save_config(self.config_path, config)

        choose_directory_button = tk.Button(
            self.control_frame, text="Choose Directory",command=choose_directory)
        choose_directory_button.grid(row=2, column=0, sticky="e")
    
    def render_ref_path_control(self):
        def choose_ref_path():
            input_dir = filedialog.askopenfilename(title="Select a Genome Reference File")
            if not input_dir:
                return
            try:
                check_ref_format(input_dir)
            except Exception as e:
                messagebox.showerror(
                    "File Error", "Please check that your file is in correct fasta format.")
                return
                
            self.ref_path = input_dir
            self.ref_path_label.config(text=self.ref_path)
            config = load_config_as_dict(self.config_path)
            config["GenomeElement"]["ref_path"] = self.ref_path
            save_config(self.config_path, config)

        ref_path_label = ttk.Label(
            self.control_frame, text="Pathogen Reference Genome File (FASTA Format)", 
            style="Bold.TLabel")
        ref_path_label.grid(row=4, column=0, sticky="w", pady=5)

        CreateToolTip(ref_path_label, 
                      "Path to the reference genome of the pathogen (FASTA format).")

        if self.ref_path == "":
            self.ref_path_label = ttk.Label(
                self.control_frame, text="None selected", foreground="black", width=50)
        else:
            self.ref_path_label = ttk.Label(
                self.control_frame, text=self.ref_path, foreground="black", width=50)

        self.ref_path_label.grid(row=5, column=0, pady=5, sticky="w")

        choose_ref_path_button = tk.Button(
            self.control_frame, text="Choose File", command=choose_ref_path)
        choose_ref_path_button.grid(row=6, column=0, sticky="e", pady=5)

    def render_n_replicates(self):
        self.working_directory_label = ttk.Label(
            self.control_frame, text="Working Directory:", style="Bold.TLabel")
        self.working_directory_label.grid(row=0, column=0, pady=5, sticky="w")
        if self.cwd == "":
            self.user_working_directory_label = ttk.Label(
                self.control_frame, text="None Selected", foreground="black", width=50)
        else:
            self.user_working_directory_label = ttk.Label(
                self.control_frame, text=self.cwd, foreground="black", width=50)

        CreateToolTip(self.working_directory_label, 
                      "Path of the working directory of the current workflow, all files will be stored there.")
        self.user_working_directory_label.grid(row=1, column=0, pady=5, sticky="w")

        self.n_replicates_label = ttk.Label(
            self.control_frame, text="Number of Simulation Replicates (Integer)", 
            style="Bold.TLabel")

        CreateToolTip(self.n_replicates_label, 
                      "The number of simulation replicates to run with the current configuration.")
        self.n_replicates_label.grid(row=7, column=0, sticky="w", pady=5)
        self.n_replicates_entry = ttk.Entry(self.control_frame, foreground="black", width=20)
        self.n_replicates_entry.grid(row=8, column=0, sticky="w", pady=5)
        self.n_replicates_entry.insert(0, self.n_replicates)

    def render_random_seed(self):
        random_seed_label = ttk.Label(
            self.control_frame, text="Random Seed (Integer)", style="Bold.TLabel")
        CreateToolTip(random_seed_label, 
                      "Random seed used for the preprocessing steps.")
        random_seed_label.grid(row=9, column=0, sticky="w", pady=5)
        self.random_seed_entry = ttk.Entry(self.control_frame, foreground="black", width=20)
        self.random_seed_entry.grid(row=10, column=0, sticky="w", pady=5)
        self.random_seed_entry.insert(0, str(self.random_number_seed))

    def update_config(self):
        error_messages = []
        self.update_n_replicates(error_messages)
        self.update_random_seed(error_messages)
        if len(error_messages) == 0:
            return 0
        else:
            error_message_str = "\n".join(error_messages)
            messagebox.showerror("Update Error", error_message_str)
            return 1
    
    def update_random_seed(self, error_messages):
        if self.random_seed_entry.get() == "None":
            config = load_config_as_dict(self.config_path)
            config["BasicRunConfiguration"]["random_number_seed"] = None
            save_config(self.config_path, config)
            return

        try:
            new_random_seed = int(self.random_seed_entry.get())
            config = load_config_as_dict(self.config_path)
            config["BasicRunConfiguration"]["random_number_seed"] = new_random_seed
            save_config(self.config_path, config)
        except ValueError:
            error_messages.append("Please enter a valid integer for the random seed. " 
                                  "Enter 'None' if you don't want to provide a random seed.")
    
    def update_n_replicates(self, error_messages):
        try:
            new_n_replicates = int(self.n_replicates_entry.get())
            config = load_config_as_dict(self.config_path)
            config["BasicRunConfiguration"]["n_replicates"] = new_n_replicates
            save_config(self.config_path, config)
        except ValueError:
            error_messages.append("Please enter a valid integer for the number of replicates.")
