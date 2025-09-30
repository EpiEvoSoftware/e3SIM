from tkinter import messagebox
import json
from utils import (load_config_as_dict, save_config, no_validate_update_val, no_validate_update,
                   TabBase, GroupControls, EasyPathSelector, EasyTitle, EasyCombobox,
                   EasyRadioButton, EasyButton, EasyEntry, EasyLabel, CreateToolTip)
from genetic_effect_generator import run_effsize_generation

class GenomeElement(TabBase):
    def __init__(self, parent, tab_parent, config_path, tab_title, tab_index, hide=False):
        super().__init__(parent, tab_parent, config_path, tab_title, tab_index, hide)

    def init_val(self, config_path):
        self.config_path = config_path
        self.initial_genome_config = load_config_as_dict(config_path)["GenomeElement"]

    def load_page(self):
        self.render_simulation_settings_title(False, 0, 0, 1)
        self.render_use_sigmoid_probs(False, 0, 1, 1)
        self.render_use_genetic_model(False, 0, 4, 1)

        self.global_group_control = GroupControls()
        self.init_num_traits_group()
        self.init_user_input_group()
        self.init_random_generate_group()

    def init_num_traits_group(self):
        hide = not self.initial_genome_config["use_genetic_model"]
        number_of_traits_title = self.render_number_of_traits_title(hide, 0, 7)
        transmissibility = self.render_transmissibility(hide, 0, 8)
        drug_resistance = self.render_drug_resistance(hide, 1, 8)
        generate_method = self.render_generate_method(0, 10, 2, hide, 30) ###?
        self.generate_method = generate_method

        lst = [number_of_traits_title, transmissibility, drug_resistance, generate_method]
        self.num_traits_group_control = GroupControls(lst)
        self.global_group_control.add(self.num_traits_group_control)
    
    def init_user_input_group(self):
        hide = (not self.initial_genome_config["use_genetic_model"] 
                or self.initial_genome_config["effect_size"]["method"] != "user_input")
        
        file_input = self.render_path_eff_size_table(hide, 0, 14, 2)
        run_button = self.render_run_button(hide, 0, 17, "user_input")

        self.user_input_group_control = GroupControls()
        self.user_input_group_control.add(file_input)
        self.user_input_group_control.add(run_button)
        if not hide:
            self.global_group_control.add(self.user_input_group_control)

    def init_random_generate_group(self):
        hide = (not self.initial_genome_config["use_genetic_model"] 
                or self.initial_genome_config["effect_size"]["method"] != "randomly_generate")

        gff = self.render_gff(hide, 0, 12, 1)
        genes_num = self.render_genes_num(hide, 0, 15)
        effsize_min = self.render_effsize_min(hide, 0, 17)
        effsize_max = self.render_effsize_max(hide, 0, 19)
        normalize = self.render_normalize(hide, 1, 12)
        normalize_f_trait = self.render_normalize_f_trait(hide, 1, 15)
        self.f_trait_control = normalize_f_trait
        run_button = self.render_run_button(hide, 0, 34, "randomly_generate")

        lst = [gff, genes_num, effsize_min, effsize_max, normalize, normalize_f_trait, run_button]
        self.random_generate_group_control = GroupControls(lst)
        if not hide:
            self.global_group_control.add(self.random_generate_group_control)

    def render_simulation_settings_title(self, hide=True, column=None, frow=None, columnspan=1):
        self.render_simulation_settings_title_text = "Simulation Settings"
        self.number_of_traits_label = EasyTitle(
            self.render_simulation_settings_title_text,
            self.control_frame,
            column,
            frow,
            hide,
            columnspan,
        )

    def render_use_genetic_model(self, hide=True, column=None, frow=None, columnspan=1):
        def radiobuttonselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
            if var.get():
                self.global_group_control.rerender_itself()
            else:
                self.global_group_control.derender_itself()
        
        keys_path = ["GenomeElement", "use_genetic_model"]
        text = "Do you want to use genetic architecture "
        "for traits (transmissibility/Drug-resistance)?"
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path,
            self.config_path,
            text,
            "use_genetic_model",
            self.control_frame,
            column,
            frow,
            hide,
            to_rerender,
            to_derender,
            columnspan,
            radiobuttonselected,
            labtext="If you want to set trait values that are adjusted by the mutations on the genome, please select \"YES\"."
        )
        return component

    def render_use_sigmoid_probs(self, hide=True, column=None, frow=None, columnspan=1):
        def radiobuttonselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
        
        keys_path = ["GenomeElement", "sigmoid_prob"]
        text = "Do you want to use sigmoidal smoothing"
        "for traits (transmissibility and drug-resistance)?"
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path, self.config_path, text, "sigmoid_prob", 
            self.control_frame, column, frow, hide, 
            to_rerender, to_derender,
            columnspan, radiobuttonselected,
            labtext="If you want to use a sigmoidal function to smooth and constrain the probability of infection and recovery, please select \"YES\"."
            )
        self.visible_components.add(component)
        
        return component

    def render_number_of_traits_title(self, hide=True, column=None, frow=None):
        text = "Number of traits where genetic architectures can be specified:"
        component = EasyLabel(text, self.control_frame, column, frow, hide, 
                              labtext="More than 1 trait can be specified for each trait type (transmissibility/drug-resistance).")
        return component

    def render_transmissibility(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "traits_num", "transmissibility"]
        text = "Transmissibility"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "transmissibility",
            self.control_frame,
            column,
            frow,
            "integer",
            hide,
            columnspan=1,
            labtext="Number of transmissibility traits you want to specify. In each epoch, only 1 transmissibility trait will be enabled, such that transmissibility is determined by its genetic architecture.",
        )
        self.visible_components.add(component)
        return component

    def render_drug_resistance(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "traits_num", "drug_resistance"]
        text = "Drug-Resistance"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "drug-resistance",
            self.control_frame,
            column,
            frow,
            "integer",
            hide,
            columnspan=1,
            labtext="Number of drug-resistance traits you want to specify. In each epoch, only 1 drug-resistance trait will be enabled, such that drug-resistance is determined by its genetic architecture.",
        )
        self.visible_components.add(component)
        return component

    def render_generate_method(self, column=None, frow=None, columnspan=1, hide=True, width=20):
        def comboboxselected(var, to_rerender, to_derender):
            val = var.get()
            from_ui_mapping = {v: k for k, v in to_ui_mapping.items()}
            converted_val = from_ui_mapping.get(val, "")
            no_validate_update_val(converted_val, self.config_path, keys_path)

            # Set render logic for generate method combobox
            match converted_val:
                case "user_input":
                    self.user_input_group_control.rerender_itself()
                    self.random_generate_group_control.derender_itself()
                case "randomly_generate":
                    self.random_generate_group_control.rerender_itself()
                    self.user_input_group_control.derender_itself()
            
            # Set render logic for use genetic model radiobutton
            match converted_val:
                case "user_input":
                    self.global_group_control.add(self.user_input_group_control)
                    if self.random_generate_group_control in self.global_group_control.items:
                        self.global_group_control.items.remove(self.random_generate_group_control)
                case "randomly_generate":
                    self.global_group_control.add(self.random_generate_group_control)
                    if self.user_input_group_control in self.global_group_control.items:
                        self.global_group_control.items.remove(self.user_input_group_control)
                    
        keys_path = ["GenomeElement", "effect_size", "method"]
        text = "Method to Generate the Genetic Architecture"
        to_rerender, to_derender = None, None
        to_ui_mapping = {
            "user_input": "User Input",
            "randomly_generate": "Random Generation from the GFF file",
        }
        values = list(to_ui_mapping.values())
        component = EasyCombobox(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            frow,
            values,
            to_rerender,
            to_derender,
            comboboxselected,
            hide,
            width,
            columnspan,
            to_ui_mapping,
        )
        return component

    def render_gff(self, hide=True, column=None, frow=None, columnspan=1):
        keys_path = ["GenomeElement", "effect_size", "randomly_generate", "gff"]
        text = "Please provide the genome annotation (gff-like format):"
        component = EasyPathSelector(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            hide,
            frow,
            columnspan,
            labtext="The file has to have at least 5 columns, with the 4th and 5th column showing the starting and ending positions of one genetic element."
        )
        return component

    def render_genes_num(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "effect_size", "randomly_generate", "genes_num"]
        text = "Number of Genomic Regions for each trait (List Integer)"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "genes_num",
            self.control_frame,
            column,
            frow,
            "list integer",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        return component

    def render_effsize_min(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "effect_size", "randomly_generate", "effsize_min"]
        text = "Minimum Effect Size of each region for each trait (List Numerical)"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Minimum Effect Size",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        return component

    def render_effsize_max(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "effect_size", "randomly_generate", "effsize_max"]
        text = "Maximum Effect Size of each region for each trait (List Numerical)"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Maximum Effect Size",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        return component

    def render_normalize(self, hide=True, column=None, frow=None, columnspan=1):
        def update(var, to_rerender, to_derender):
            if var.get():
                self.f_trait_control.label.configure(state="normal")
                self.f_trait_control.entry.configure(state="normal")
            else:
                self.f_trait_control.label.configure(state="disabled")
                self.f_trait_control.label.configure(state="disabled")
            
            config = load_config_as_dict(self.config_path)
            config["GenomeElement"]["effect_size"]["randomly_generate"]["normalize"] = var.get()
            save_config(self.config_path, config)

        text = "Whether to normalize randomly-selected effect sizes" \
                    " by the expected number of mutations?"
        keys_path = ["GenomeElement", "effect_size", "randomly_generate", "normalize"]
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path,
            self.config_path,
            text,
            "normalize",
            self.control_frame,
            column,
            frow,
            hide,
            to_rerender,
            to_derender,
            columnspan,
            update,
        )
        return component
    
    def render_normalize_f_trait(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "effect_size", "randomly_generate", "final_trait"]
        text = "Final trait normalization factor (Numerical)"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Final trait normalization factor",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            disabled=True,
            columnspan=1,
        )
        self.visible_components.add(component)
        return component

    def render_run_button(self, hide=True, column=None, frow=None, method=""):
        if method == "randomly_generate":
            text = "Run Effect Size Generation"
        elif method == "user_input":
            text = "Validating input (Required)"
        component = EasyButton(
            text,
            self.control_frame,
            column,
            frow,
            self.effect_size_generation,
            hide,
            sticky="w",
        )
        return component

    def render_path_eff_size_table(self, hide=True, column=None, frow=None, columnspan=1):
        text = "Please provide the Genetic Architecture File (CSV format):"
        keys_path = ["GenomeElement", "effect_size", "user_input", "path_effsize_table"]
        component = EasyPathSelector(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            hide,
            frow,
            columnspan,
        )
        return component

    def effect_size_generation(self):
        if self.global_update() == 1:
            return

        config = load_config_as_dict(self.config_path)
        evol_config = config["EvolutionModel"]
        genome_config = config["GenomeElement"]

        wk_dir = config["BasicRunConfiguration"]["cwdir"]
        rand_seed = config["BasicRunConfiguration"]["random_number_seed"]
        num_seed = config["SeedsConfiguration"]["seed_size"]
        n_gen = evol_config["n_generation"]
        use_subst_matrix = evol_config["subst_model_parameterization"] == "mutation rate matrix"
        mut_rate = evol_config["mut_rate"]
        mu_matrix_values = evol_config["mut_rate_matrix"]
        mu_matrix = json.dumps({"A": mu_matrix_values[0], "C": mu_matrix_values[1], 
                                "G": mu_matrix_values[2], "T": mu_matrix_values[3]})

        method = genome_config["effect_size"]["method"]
        ref = genome_config["ref_path"]
        trait_n = genome_config["traits_num"]

        if method == "user_input":
            effsize_path = genome_config["effect_size"]["user_input"]["path_effsize_table"]
            gff_in = None
            causal_sizes = None
            es_lows = None
            es_highs = None
            norm_or_not = None
            final_T = None
        elif method == "randomly_generate":
            effsize_path = ""
            gff_in = genome_config["effect_size"]["randomly_generate"]["gff"]
            causal_sizes = genome_config["effect_size"]["randomly_generate"]["genes_num"]
            es_lows = genome_config["effect_size"]["randomly_generate"]["effsize_min"]
            es_highs = genome_config["effect_size"]["randomly_generate"]["effsize_max"]
            norm_or_not = genome_config["effect_size"]["randomly_generate"]["normalize"]
            final_T = genome_config["effect_size"]["randomly_generate"]["final_trait"]

            if norm_or_not:
                if n_gen <= 0:
                    messagebox.showerror(
                        "Value Error", "Number of Generations (Evolutionary Model) "
                        "has to be a positive integer if normalization is turned on.")
                    return
                if not use_subst_matrix and mut_rate == 0:
                    messagebox.showerror(
                        "Value Error", "Mutation rate (Evolutionary Model) "
                        "has to be a positive number if normalization is turned on.")
                    return
                if use_subst_matrix:
                    if sum(sum(lst) for lst in mu_matrix_values) == 0:
                        messagebox.showerror(
                            "Value Error", "Mutation rate matrix (Evolutionary Model) cannot be"
                            " all 0s if normalization is turned on and using substitution matrix")
                        return
                    if ref == "":
                        messagebox.showerror(
                            "Value Error", "Pathogen Reference Genome File (Basic Configuration) "
                            "must be provided if normalization is turned on and if using "
                            "substitution matrix.")
                        return
        else:
            messagebox.showerror(
                "Value Error", "Please select a method to generate the genetic architecture.")
            return
    
        err = run_effsize_generation(
            method,
            wk_dir,
            effsize_path=effsize_path,
            gff_in=gff_in,
            trait_n=trait_n,
            causal_sizes=causal_sizes,
            es_lows=es_lows,
            es_highs=es_highs,
            norm_or_not=norm_or_not,
            n_gen=n_gen,
            num_seed=num_seed,
            use_subst_matrix=use_subst_matrix,
            mut_rate=mut_rate,
            mu_matrix=mu_matrix,
            rand_seed=rand_seed,
            ref=ref,
            final_T=final_T
        )
        if err:
            messagebox.showerror("Generation Error", "Generation Error: " + str(err))
        else:
            messagebox.showinfo("Success", "Effect size generation completed successfully!")
