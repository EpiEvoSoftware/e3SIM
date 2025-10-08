import tkinter as tk
from tkinter import messagebox, ttk
import platform
import json
from utils import (load_config_as_dict, save_config, no_validate_update_val, no_validate_update,
                   TabBase, GroupControls, EasyPathSelector, EasyTitle, EasyCombobox,
                   EasyRadioButton, EasyButton, EasyEntry, EasyLabel, CreateToolTip)
from e3SIM_codes.genetic_effect_generator import *

class GenomeElement(TabBase):
    def _ensure_scrollable_t5(self):
        # Create only once
        if hasattr(self, "_t5_scroll_ready") and self._t5_scroll_ready:
            return

        self.outer = self.control_frame

        outer = self.control_frame  # keep a handle to the current container

        self._t5_canvas = tk.Canvas(
            outer,
            highlightthickness=0,  # remove black outline
            bd=0                   # remove border
        )
        self._t5_vbar = ttk.Scrollbar(outer, orient="vertical",
                                      command=self._t5_canvas.yview)
        self._t5_canvas.configure(yscrollcommand=self._t5_vbar.set)


        # This is where all your existing widgets will be gridded
        self._t5_scrollable = ttk.Frame(self._t5_canvas)
        self._t5_canvas_window = self._t5_canvas.create_window(
            (0, 0), window=self._t5_scrollable, anchor="nw"
        )

        # --- make it expand to full tab ---
        self._t5_canvas.grid(row=0, column=0, sticky="nsew")
        self._t5_vbar.grid(row=0, column=1, sticky="ns")
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        # ----------------------------------

        # Let the scrollregion follow the content
        def _on_frame_config(event):
            self._t5_canvas.configure(scrollregion=self._t5_canvas.bbox("all"))

        # Make inner frame match canvas width when the window resizes
        def _on_canvas_config(event):
            self._t5_canvas.itemconfig(self._t5_canvas_window, width=event.width)

        self._t5_scrollable.bind("<Configure>", _on_frame_config)
        self._t5_canvas.bind("<Configure>", _on_canvas_config)

        # Mouse wheel support (Win/Mac/Linux)
        def _on_mousewheel(event):
            sys = platform.system()
            if sys == "Windows":
                self._t5_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif sys == "Darwin":
                self._t5_canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                self._t5_canvas.yview_scroll(int(-1 * event.delta), "units")

        def _bind_wheel(_):
            if platform.system() == "Linux":
                self._t5_canvas.bind_all("<Button-4>", lambda e: self._t5_canvas.yview_scroll(-1, "units"))
                self._t5_canvas.bind_all("<Button-5>", lambda e: self._t5_canvas.yview_scroll(1, "units"))
            else:
                self._t5_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(_):
            if platform.system() == "Linux":
                self._t5_canvas.unbind_all("<Button-4>")
                self._t5_canvas.unbind_all("<Button-5>")
            else:
                self._t5_canvas.unbind_all("<MouseWheel>")

        self._t5_scrollable.bind("<Enter>", _bind_wheel)
        self._t5_scrollable.bind("<Leave>", _unbind_wheel)

        # Pack the canvas+scrollbar inside the original control_frame
        self._t5_canvas.grid(row=0, column=0, sticky="nsew")
        self._t5_vbar.grid(row=0, column=1, sticky="ns")

        # give the canvas the space to grow
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        # self._t5_vbar.pack(side=tk.RIGHT, fill="y")

        # IMPORTANT: from now on, render into the scrollable frame
        self.control_frame = self._t5_scrollable
        # Optional: make the right column stretch
        self.control_frame.columnconfigure(0, weight=0)
        self.control_frame.columnconfigure(1, weight=1)

        self._t5_scroll_ready = True

    def __init__(self, parent, tab_parent, config_path, tab_title, tab_index, hide=False):
        super().__init__(parent, tab_parent, config_path, tab_title, tab_index, hide)

    def init_val(self, config_path):
        self.config_path = config_path
        self.initial_genome_config = load_config_as_dict(config_path)["GenomeElement"]

    def load_page(self):
        self._ensure_scrollable_t5()
        self.render_simulation_settings_title(False, 0, 0, 1)
        # self.render_use_sigmoid_probs(False, 0, 1, 1)
        self.render_use_genetic_model(False, 0, 4, 1)

        self.global_group_control = GroupControls()
        self.init_num_traits_group()
        self.init_user_input_group()
        self.init_random_generate_group()
        self.init_effsizecalibration_group()
        self.init_alphacalibration_group()

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
                or self.initial_genome_config["effect_size"]["method"] != "csv")
        
        file_input = self.render_path_eff_size_table(hide, 0, 13, 2)
        run_button = self.render_run_button(hide, 0, 17, "csv")

        self.user_input_group_control = GroupControls()
        self.user_input_group_control.add(file_input)
        self.user_input_group_control.add(run_button)
        if not hide:
            self.global_group_control.add(self.user_input_group_control)

    def init_random_generate_group(self):
        hide = (not self.initial_genome_config["use_genetic_model"] 
                or self.initial_genome_config["effect_size"]["method"] != "gff")
        parent = self.control_frame

        gff = self.render_gff(hide, 0, 13, 1)
        site_model = self.render_sitesmethod(hide, 0, 16)

        effsize_func = self.render_effsizefunc(hide, 0, 18)
        run_button = self.render_run_button(hide, 0, 34, "gff")
        lst = [gff, site_model, effsize_func, run_button]#, pis, Ks]
        self.random_generate_group_control = GroupControls(lst)
        if not hide:
            self.global_group_control.add(self.random_generate_group_control)


    def init_effsizecalibration_group(self):
        hide = (not self.initial_genome_config["use_genetic_model"])
        if self.initial_genome_config["effect_size"]["method"] == "csv":
            current_row = 18
        else:
            current_row = 35
        effsize_cali = self.render_calibration(hide, 0, current_row + 1)

    def init_alphacalibration_group(self):
        hide = (not self.initial_genome_config["use_genetic_model"])
        if self.initial_genome_config["effect_size"]["method"] == "csv":
            current_row = 21
        else:
            current_row = 38
        link_type = self.render_traitproblink(hide, 0, current_row + 1)
        alphatrans = self.render_alphatrans(hide, 0, current_row + 3)
        alphadrug = self.render_alphadr(hide, 1, current_row + 3)
        effsize_cali = self.render_alphacalibration(hide, 0, current_row + 5, 3)
        run_cali_alpha = self.render_runalphacali_button(hide, 0, current_row + 8)


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
                case "csv":
                    self.user_input_group_control.rerender_itself()
                    self.random_generate_group_control.derender_itself()
                case "gff":
                    self.random_generate_group_control.rerender_itself()
                    self.user_input_group_control.derender_itself()
            
            # Set render logic for use genetic model radiobutton
            match converted_val:
                case "csv":
                    self.global_group_control.add(self.user_input_group_control)
                    if self.random_generate_group_control in self.global_group_control.items:
                        self.global_group_control.items.remove(self.random_generate_group_control)
                case "gff":
                    self.global_group_control.add(self.random_generate_group_control)
                    if self.user_input_group_control in self.global_group_control.items:
                        self.global_group_control.items.remove(self.user_input_group_control)
                    
        keys_path = ["GenomeElement", "effect_size", "method"]
        text = "Method to Generate the Genetic Architecture"
        to_rerender, to_derender = None, None
        to_ui_mapping = {
            "csv": "User Input from a CSV file",
            "gff": "Random Generation from the GFF file",
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
        keys_path = ["GenomeElement", "effect_size", "filepath", "gff_path"]
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

    # def render_genes_num(self, hide=True, column=None, frow=None):
    def render_sitesmethod(self, hide=True, column=None, frow=None):
        def comboboxselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
            val = var.get()
            from_ui_mapping = {v: k for k, v in to_ui_mapping.items()}
            converted_val = from_ui_mapping.get(val, "")
            #Toggle SIR/SEIR Models
            if converted_val=="p":
                pis = self.render_pis(False, 1, 1, 16)
                ks = self.render_Ks(True, 1, 1, 16)
                # ks.derender_itself()
            else:
                pis = self.render_pis(True, 1, 1, 16)
                ks = self.render_Ks(False, 1, 1, 16)
                # pis.derender_itself()

        keys_path = ["GenomeElement", "effect_size", "causalsites_params", "method"]
        text = "Method to generate sites with non-zero effect sizes"
        
        to_ui_mapping = {
            "p": "Bernoulli trials on each candidate site for each trait",
            "n": "Choose a fixed number of causal sites for each trait",
        }
        values = list(to_ui_mapping.values())
        to_rerender, to_derender = None, None
        width=35
        columnspan = 1
        component = EasyCombobox(
            keys_path, self.config_path, text, self.control_frame,
            column, frow, values, to_rerender, to_derender,
            comboboxselected, hide, width, columnspan, to_ui_mapping
        )
        self.visible_components.add(component)

        return component


    def render_Ks(self, hide, column, columnspan, frow):
        text = "Number of sites causal for each trait"
        keys_path = ['GenomeElement','effect_size','causalsites_params', 'Ks']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Number of causal sites',
            self.control_frame, column, frow, 'list integer', hide, columnspan,
            labtext="Number of sites causal for each trait."
        )
        self.visible_components.add(component)
        self.global_group_control.add(component)
        self.random_generate_group_control.add(component)

        return component

    def render_pis(self, hide, column, columnspan, frow):
        text = "Probabilities of sites being causal for each trait"
        keys_path = ['GenomeElement','effect_size','causalsites_params', 'pis']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Probability of being causal per site',
            self.control_frame, column, frow, 'list float', hide, columnspan,
            labtext="Probabilities of sites being causal for each trait."
        )
        self.visible_components.add(component)
        self.random_generate_group_control.add(component)
        self.global_group_control.add(component)
        return component


    def render_effsize_normaltaus(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "effect_size", "effsize_params", "normal", "taus"]
        text = "Taus for normal distribution"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Taus for normal distribution",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        self.random_generate_group_control.add(component)
        self.global_group_control.add(component)
        return component

    def render_effsize_laplacebs(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "effect_size", "effsize_params", "laplace", "bs"]
        text = "bs for laplace distribution"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "bs for laplace distribution",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        self.random_generate_group_control.add(component)
        self.global_group_control.add(component)
        return component

    def render_effsize_sts(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "effect_size", "effsize_params", "studentst", "s"]
        text = "s for Student's t's distribution"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "s for st distribution",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        self.random_generate_group_control.add(component)
        self.global_group_control.add(component)
        return component


    def render_stnv(self, hide, column, columnspan, frow):
        text = "nv for Student's t's distribution"
        keys_path = ["GenomeElement", "effect_size", "effsize_params", "studentst", "nv"]
        component = EasyEntry(
            keys_path, self.config_path, text, 'Degree of freedom for the student\'s t\'s distribution',
            self.control_frame, column, frow, 'integer', hide, columnspan,
            labtext="nv"
        )
        self.visible_components.add(component)
        self.random_generate_group_control.add(component)
        self.global_group_control.add(component)
        return component


    def render_calibration(self, hide=True, column=None, frow=None, columnspan=1):
        def update(var, to_rerender, to_derender):
            if var.get():
                cali_var = self.render_Vtargets(False, 1, frow)
            else:
                cali_var = self.render_Vtargets(True, 1, frow)
            
            config = load_config_as_dict(self.config_path)
            config["GenomeElement"]["effect_size"]["calibration"]["do_calibration"] = var.get()
            self.global_group_control.add(cali_var)
            save_config(self.config_path, config)


        text = "Whether to calibrate effect sizes" \
                    " by the provided seeding populations?"
        keys_path = ["GenomeElement", "effect_size", "calibration", "do_calibration"]
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path,
            self.config_path,
            text,
            "calibration",
            self.control_frame,
            column,
            frow,
            hide,
            to_rerender,
            to_derender,
            columnspan,
            update,
        )
        
        self.global_group_control.add(component)
        return component
    

    def render_effsizefunc(self, hide=True, column=None, frow=None):
        def comboboxselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
            val = var.get()
            from_ui_mapping = {v: k for k, v in to_ui_mapping.items()}
            converted_val = from_ui_mapping.get(val, "")
            #Toggle SIR/SEIR Models
            if converted_val == "n":
                self.render_effsize_normaltaus(False, 0, 20)
                self.render_effsize_laplacebs(True, 0, 20)
                self.render_effsize_sts(True, 0, 20)
                self.render_stnv(True, 1, 1, 20)
            elif converted_val == "l":
                self.render_effsize_normaltaus(True, 0, 20)
                self.render_effsize_laplacebs(False, 0, 20)
                self.render_effsize_sts(True, 0, 20)
                self.render_stnv(True, 1, 1, 20)
            elif converted_val == "st":
                self.render_effsize_normaltaus(True, 0, 20)
                self.render_effsize_laplacebs(True, 0, 20)
                self.render_effsize_sts(False, 0, 20)
                self.render_stnv(False, 1, 1, 20)

        keys_path = ["GenomeElement", "effect_size", "effsize_params", "effsize_function"]
        text = "Distribution to generate effezt sizes for each trait given selected sites"
        to_ui_mapping = {
            "n": "Normal distribution",
            "l": "Laplace distribution",
            "st": "Student\'s t\'s distribution"
        }
        values = list(to_ui_mapping.values())
        
        to_rerender, to_derender = None, None
        width=20
        columnspan = 1
        component = EasyCombobox(
            keys_path, self.config_path, text, self.control_frame,
            column, frow, values, to_rerender, to_derender,
            comboboxselected, hide, width, columnspan, to_ui_mapping
        )
        self.visible_components.add(component)
        self.global_group_control.add(component)
        return component



    def render_run_button(self, hide=True, column=None, frow=None, method=""):
        if method == "gff":
            text = "Run Effect Size Generation"
        elif method == "csv":
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
        keys_path = ["GenomeElement", "effect_size", "filepath", "csv_path"]
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


    def render_traitproblink(self, hide=True, column=None, frow=None):
        def comboboxselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
            #Toggle SIR/SEIR Models
            if var.get() == "logit":
                return True
            elif var.get() == "cloglog":
                return True

        keys_path = ["GenomeElement", "trait_prob_link", "link"]
        text = "Method to link the trait values to event probabilities"
        values = ["logit", "cloglog"]
        to_rerender, to_derender = None, None
        width=20
        columnspan = 1
        component = EasyCombobox(
            keys_path, self.config_path, text, self.control_frame,
            column, frow, values, to_rerender, to_derender,
            comboboxselected, hide, width, columnspan
        )
        self.visible_components.add(component)
        self.global_group_control.add(component)
        return component


    def render_caliRs(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "trait_prob_link", "Rs"]
        text = "odds ratios for the calibration of the link coefficients"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Rs odss ratios etc.",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        self.global_group_control.add(component)
        return component

    def render_Vtargets(self, hide=True, column=None, frow=None):
        keys_path = ["GenomeElement", "effect_size", "calibration", "V_target"]
        text = "Target trait variances for effect size calibration"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Target trait variances for effect size calibration",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        self.global_group_control.add(component)
        return component


    def render_alphacalibration(self, hide=True, column=None, frow=None, columnspan=1):
        def update(var, to_rerender, to_derender):
            if var.get():
                self.render_caliRs(False, 1, frow)
            else:
                self.render_caliRs(True, 1, frow)
            
            config = load_config_as_dict(self.config_path)
            config["GenomeElement"]["trait_prob_link"]["calibration"] = var.get()
            save_config(self.config_path, config)

        text = "Whether to calibrate the link scale slopes?"
        keys_path = ["GenomeElement", "trait_prob_link", "calibration"]
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path,
            self.config_path,
            text,
            "Rs for calibration of alphas",
            self.control_frame,
            column,
            frow,
            hide,
            to_rerender,
            to_derender,
            columnspan,
            update,
        )
        self.global_group_control.add(component)
        return component


    def render_alphatrans(self, hide=True, column=None, frow=None):
        if self.initial_genome_config["trait_prob_link"]["link"] == "logit":
            keys_path = ["GenomeElement", "trait_prob_link", "logit", "alpha_trans"]
        elif self.initial_genome_config["trait_prob_link"]["link"] == "cloglog":
            keys_path = ["GenomeElement", "trait_prob_link", "cloglog", "alpha_trans"]
        text = "Link scale slope for transmissibiility trait"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "alpha values for transmissibiility",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        self.global_group_control.add(component)
        return component


    def render_alphadr(self, hide=True, column=None, frow=None):
        if self.initial_genome_config["trait_prob_link"]["link"] == "logit":
            keys_path = ["GenomeElement", "trait_prob_link", "logit", "alpha_drug"]
        elif self.initial_genome_config["trait_prob_link"]["link"] == "cloglog":
            keys_path = ["GenomeElement", "trait_prob_link", "cloglog", "alpha_drug"]
        text = "Link scale slope for drug-resistance trait"
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "alpha values for drug resistance",
            self.control_frame,
            column,
            frow,
            "list numerical",
            hide,
            columnspan=1,
        )
        self.visible_components.add(component)
        self.global_group_control.add(component)
        return component


    def render_runalphacali_button(self, hide=True, column=None, frow=None):
        text = "Run slope calibration"
        component = EasyButton(
            text,
            self.control_frame,
            column,
            frow,
            self.effect_size_generation,
            hide,
            sticky="w",
        )
        self.global_group_control.add(component)
        return component
    

########################
## RUN
    def effect_size_generation(self):
        if self.global_update() == 1:
            return

        config = load_config_as_dict(self.config_path)

        genome_config = config["GenomeElement"]

        wk_dir = config["BasicRunConfiguration"]["cwdir"]
        rand_seed = config["BasicRunConfiguration"]["random_number_seed"]
        num_seed = config["SeedsConfiguration"]["seed_size"]

        method = genome_config["effect_size"]["method"]

        try:
            config = GeneticEffectConfig(
                method = method,
                wk_dir = wk_dir,
                n_seed = num_seed,
                func = genome_config["effect_size"]["effsize_params"]["effsize_function"],
                calibration = genome_config["effect_size"]["calibration"]["do_calibration"],
                random_seed = rand_seed,
                csv = genome_config["effect_size"]["filepath"]["csv_path"],
                gff = genome_config["effect_size"]["filepath"]["gff_path"],
                trait_num = genome_config["traits_num"],
                pis = genome_config["effect_size"]["causalsites_params"]["pis"],
                Ks = genome_config["effect_size"]["causalsites_params"]["Ks"],
                taus = genome_config["effect_size"]["effsize_params"]["normal"]["taus"],
                bs = genome_config["effect_size"]["effsize_params"]["laplace"]["bs"],
                nv = genome_config["effect_size"]["effsize_params"]["studentst"]["nv"],
                s = genome_config["effect_size"]["effsize_params"]["studentst"]["s"],
                var_target = genome_config["effect_size"]["calibration"]["V_target"],
                calibration_link = genome_config["trait_prob_link"]["calibration"],
                Rs = genome_config["trait_prob_link"]["Rs"],
                link = genome_config["trait_prob_link"]["link"],
                site_method = genome_config["effect_size"]["causalsites_params"]["method"]
            )
        except Exception as e:
            return e

        generator = EffectGenerator(config) # no validation going on so leave it out of the try catch clause
        err = generator.run()

        if err:
            messagebox.showerror("Generation Error", "Generation Error: " + str(err))
        else:
            messagebox.showinfo("Success", "Effect size generation completed successfully!")
