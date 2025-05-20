import tkinter as tk
from tkinter import ttk, messagebox
from utils import render_next_button, load_config_as_dict, save_config, CreateToolTip

class EvolutionaryModel:
    def __init__(self, parent, tab_parent, config_path, tab_title, tab_index, hide = False):
        self.parent = parent
        self.tab_parent = tab_parent
        self.tab_title = tab_title
        self.tab_index = tab_index
        self.tab_parent.add(self.parent, text=self.tab_title)
        if hide:
            self.tab_parent.tab(self.tab_index, state="disabled")

        self.control_frame = ttk.Frame(self.parent, width=300)
        self.control_frame.pack(padx=10, pady=10) 

        self.config_path = config_path
        self.initial_evolution_config = load_config_as_dict(self.config_path)['EvolutionModel']

        self.initial_render()
        render_next_button(self.tab_index, self.tab_parent, self.parent, update_config=self.update_config)

    def initial_render(self):
        self.render_n_generations()
        self.render_model_parameterization()
        self.render_within_host_reproduction()
        if self.initial_evolution_config['within_host_reproduction']:
            self.render_within_host_reproduction_rate(False)
        else:
            self.render_within_host_reproduction_rate(True)
        self.render_cap_withinhost() 

    def render_n_generations(self):
        self.n_generation_label = ttk.Label(
            self.control_frame, text="Number of Generations (Integer)", 
            style="Bold.TLabel", width = 50)
        CreateToolTip(self.n_generation_label, 
                      "Number of generations (ticks) to run in the simulation.")
        self.n_generation_label.grid(row = 1, column = 0, sticky = 'w', pady = 5, padx = 10)
        self.n_generation_entry = ttk.Entry(self.control_frame, foreground="black")
        self.n_generation_entry.insert(0, self.initial_evolution_config['n_generation'])  
        self.n_generation_entry.grid(row = 2, column = 0, sticky = 'w', pady = 2, padx = 10)

    def render_model_parameterization(self):
        def update(event):
            parameterization = self.model_parameterization_var.get()
            if parameterization == "mutation rate (single)":
                self.mut_rate_label.grid(row = 5, column = 0, sticky = 'w', pady = 5, padx=10)
                self.mut_rate_entry.grid(row = 6, column = 0, sticky = 'w', pady = 5, padx=10)
                self.mut_rate_matrix_label.grid_forget()
                self.mut_rate_matrix_container.grid_forget()
            elif parameterization == "mutation rate matrix":
                self.mut_rate_matrix_label.grid(
                    row = 5, column = 0, sticky = 'w', pady = 5, padx=10)
                self.mut_rate_matrix_container.grid(row = 6, column = 0, stick='w', padx = 10)
                self.mut_rate_label.grid_forget()
                self.mut_rate_entry.grid_forget()

        self.model_parameterization_label = ttk.Label(
            self.control_frame, text="Substitution Model Parameterization", style = "Bold.TLabel")
        self.model_parameterization_label.grid(
            row = 3, column = 0, columnspan = 2, sticky = 'w', pady = 5, padx=10)
        self.model_parameterization_var = tk.StringVar(
            value=self.initial_evolution_config['subst_model_parameterization'])
        self.model_parameterization_combobox = ttk.Combobox(
            self.control_frame,
            textvariable=self.model_parameterization_var, 
            values=["mutation rate (single)", "mutation rate matrix"], 
            state="readonly")
        self.model_parameterization_combobox.bind("<<ComboboxSelected>>", update)
        self.model_parameterization_combobox.grid(
            row = 4, column = 0, columnspan = 2, sticky = 'w', pady = 5, padx=10)
        
        CreateToolTip(self.model_parameterization_label, 
                      "Which nucleotide substitution model to use in the simulation.")

        self.render_mut_rate()
        self.render_mut_rate_matrix()
        if self.model_parameterization_var.get() != "mutation rate (single)":
            self.mut_rate_label.grid_forget()
            self.mut_rate_entry.grid_forget()
        if self.model_parameterization_var.get() != "mutation rate matrix":
            self.mut_rate_matrix_label.grid_forget()
            self.mut_rate_matrix_container.grid_forget()

    def render_mut_rate(self):
        self.mut_rate_label = ttk.Label(
            self.control_frame, 
            text="Mutation Rate Per Site Per Generation (Numerical)", 
            width = 50, 
            style = "Bold.TLabel")
        
        CreateToolTip(self.mut_rate_label, 
                      "Probability of point mutation per site per tick (by default Jukes-Cantor model)")
        self.mut_rate_label.grid(row = 5, column = 0, sticky = 'w', pady = 5, padx=10)
        self.mut_rate_entry = ttk.Entry(self.control_frame, foreground="black")
        self.mut_rate_entry.insert(0, self.initial_evolution_config['mut_rate'])
        self.mut_rate_entry.grid(row = 6, column = 0, sticky = 'w', pady = 5, padx=10)
    
    def render_mut_rate_matrix(self):
        self.mut_rate_matrix_label = ttk.Label(
            self.control_frame,
            text="Mutation Rate Matrix Per Site Per Generation (Numerical)",
            width = 50,
            style = "Bold.TLabel")
        self.mut_rate_matrix_label.grid(row = 5, column = 0, sticky = 'w', pady = 5, padx=10)
        CreateToolTip(self.mut_rate_matrix_label, 
                      "Probability of point mutation per site per tick (The off-diagnal values represents the probability of nucleotide(row)->nucleotide(column) mutations, in the order of A,C,G,T)")
        self.mut_rate_matrix_container = ttk.Frame(self.control_frame)
        self.mut_rate_matrix_container.grid(row = 6, column = 0, stick='w', padx = 10)

        self.mut_rate_matrix_entries = []
        for i in range(4):
            lst = []
            for j in range(4):
                lst.append(ttk.Entry(self.mut_rate_matrix_container, foreground="black", width=10))
            self.mut_rate_matrix_entries.append(lst)

        initial_matrix = self.initial_evolution_config['mut_rate_matrix']
        for i in range(4):
            for j in range(4):
                entry = self.mut_rate_matrix_entries[i][j]
                entry.insert(0, initial_matrix[i][j])
                entry.grid(row = i, column = j, stick='w')
                if i == j:
                    entry.config(state='disabled', foreground='light grey')

    def render_within_host_reproduction(self):
        def update():
            value = self.within_host_reproduction_var.get()
            if value:
                self.render_within_host_reproduction_rate(False)
            else:
                self.render_within_host_reproduction_rate(True)
        self.within_host_reproduction_var = tk.BooleanVar(
            value=self.initial_evolution_config['within_host_reproduction'])
        self.within_host_reproduction_label = ttk.Label(
            self.control_frame, text="Within-host Reproduction", style = "Bold.TLabel")
        self.rb_true = ttk.Radiobutton(
            self.control_frame,
            text="Yes",
            variable=self.within_host_reproduction_var,
            value=True,
            command=update)
        self.rb_false = ttk.Radiobutton(
            self.control_frame,
            text="No",
            variable=self.within_host_reproduction_var,
            value=False,
            command=update)

        CreateToolTip(self.within_host_reproduction_label, 
                      "Whether to activate within-host reproduction (Pathogen can go through replications within hosts).")
        self.within_host_reproduction_label.grid(
            row = 15, column = 0, sticky = 'w', pady = 5, padx=10)
        self.rb_true.grid(row = 16, column = 0, sticky = 'w', pady = 5, padx=10)
        self.rb_false.grid(row = 17, column = 0, sticky = 'w', pady = 5, padx=10)

    def render_within_host_reproduction_rate(self, disabled):
        self.within_host_reproduction_rate_label = ttk.Label(
            self.control_frame,
            text="Within-host Reproduction Rate Per Generation (Numerical)",
            style = "Bold.TLabel")
        self.within_host_reproduction_rate_entry = ttk.Entry(
            self.control_frame, foreground="black")
        self.within_host_reproduction_rate_entry.insert(
            0, self.initial_evolution_config['within_host_reproduction_rate'])
        
        CreateToolTip(self.within_host_reproduction_rate_label, 
                      "Probability of within-host reproduction per pathogen per tick.")
        
        self.within_host_reproduction_rate_label.grid(
            row = 15, column = 1, sticky = 'w', pady = 5, padx=0)
        self.within_host_reproduction_rate_entry.grid(
            row = 16, column = 1, sticky = 'w', pady = 5, padx=0)

        if disabled:
            self.within_host_reproduction_rate_label.configure(state="disabled")
            self.within_host_reproduction_rate_entry.configure(
                foreground='light grey', state="disabled")
        else:
            self.within_host_reproduction_rate_label.configure(state="normal")
            self.within_host_reproduction_rate_entry.configure(foreground='black', state="normal")
 
    def render_cap_withinhost(self):
        self.cap_withinhost_label = ttk.Label(
            self.control_frame, 
            text="Maximum Number of Pathogens within Host (Integer)", 
            style = "Bold.TLabel")
        
        CreateToolTip(self.cap_withinhost_label, 
                      "Maximum number of pathogens that is permitted within one host, the effective within-host reproduction rate will be 0 if this number is reached in one host.")
        self.cap_withinhost_label.grid(row = 18, column = 0, sticky = 'w', pady = 5, padx = 10)
        self.cap_withinhost_entry = ttk.Entry(self.control_frame, foreground="black")
        self.cap_withinhost_entry.insert(0, self.initial_evolution_config['cap_withinhost'])  
        self.cap_withinhost_entry.grid(row = 19, column = 0, sticky = 'w', pady = 5, padx = 10)

    def update_n_generation(self, error_messages):
        try:
            new_n_generation = int(self.n_generation_entry.get())
            config = load_config_as_dict(self.config_path) 
            config['EvolutionModel']['n_generation'] = new_n_generation 
            save_config(self.config_path, config)  
        except ValueError:
            error_messages.append("Please enter a valid integer for Number of Generations.")

    def update_mutation_rates(self, error_messages):
        config = load_config_as_dict(self.config_path) 
        if self.model_parameterization_var.get():
            parameterization = self.model_parameterization_var.get()
            #config['EvolutionModel']['subst_model_parameterization'] = parameterization

            if parameterization == "mutation rate (single)":
                try:
                    config['EvolutionModel']['subst_model_parameterization'] = "mut_rate"
                    config['EvolutionModel']['mut_rate'] = float(self.mut_rate_entry.get())
                    self.mut_rate = config['EvolutionModel']['mut_rate']
                except ValueError:
                    error_messages.append(
                        "Please enter a valid number for mutation rate (single).")
            elif parameterization == "mutation rate matrix":
                try:
                    config['EvolutionModel']['subst_model_parameterization'] = "mut_rate_matrix"
                    for i in range(4):
                        for j in range(4):
                            e = self.mut_rate_matrix_entries[i][j]
                            config['EvolutionModel']['mut_rate_matrix'][i][j] = float(e.get())
                            e.delete(0, tk.END)
                            e.insert(0, config['EvolutionModel']['mut_rate_matrix'][i][j])
                except ValueError:
                    error_messages.append(
                        "Please enter valid numbers into the mutation rate matrix.")
        
            save_config(self.config_path, config)

    def update_within_host_reproduction(self):
        new_val = self.within_host_reproduction_var.get()
        config = load_config_as_dict(self.config_path)
        config['EvolutionModel']['within_host_reproduction'] = new_val
        save_config(self.config_path, config)

    def update_within_host_reproduction_rate(self, error_messages):
        try:
            new_within_host_reproduction_rate = float(
                self.within_host_reproduction_rate_entry.get())
            config = load_config_as_dict(self.config_path) 
            config['EvolutionModel']['within_host_reproduction_rate'] = \
                new_within_host_reproduction_rate 
            save_config(self.config_path, config)  
        except ValueError:
            error_messages.append(
                "Please enter a valid number for Within-host Reproduction Rate Per Generation.")

    def update_cap_withinhost(self, error_messages):
        try:
            new_cap_withinhost = int(self.cap_withinhost_entry.get()) 
            config = load_config_as_dict(self.config_path) 
            config['EvolutionModel']['cap_withinhost'] = new_cap_withinhost 
            save_config(self.config_path, config)  
        except ValueError:
            error_messages.append("Please enter a valid integer for cap_withinhost.")

    def update_config(self):
        error_messages = []
        self.update_n_generation(error_messages)
        self.update_mutation_rates(error_messages)
        self.update_within_host_reproduction()
        self.update_within_host_reproduction_rate(error_messages)
        self.update_cap_withinhost(error_messages)
        if len(error_messages) == 0:
            return 0
        else:
            error_message_str = "\n\n".join(error_messages)
            messagebox.showerror("Update Error", error_message_str) 
            return 1
