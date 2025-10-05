import tkinter as tk
from tkinter import messagebox
import json, os
from utils import (load_config_as_dict, no_validate_update_val, no_validate_update,
                   TabBase, GroupControls, EasyPathSelector, EasyTitle, EasyCombobox, 
                   EasyRadioButton, EasyButton, EasyEntry, EasyRateMatrix, EasyImage,
                   CreateToolTip)
from seed_generator import *

class SeedsConfiguration(TabBase):
    def __init__(self, parent, tab_parent, config_path, tab_title, tab_index, hide=False):
        super().__init__(parent, tab_parent, config_path, tab_title, tab_index, hide)

    def init_val(self, config_path):
        self.config_path = config_path
        self.frow_val = 0

        initial_seeds_config = load_config_as_dict(self.config_path)["SeedsConfiguration"]

        # SeedsConfiguration
        self.seed_size = initial_seeds_config["seed_size"]
        self.method = initial_seeds_config["method"]
        self.use_reference: bool = initial_seeds_config["use_reference"]

        # user_input
        self.path_seeds_vcf = initial_seeds_config["user_input"]["path_seeds_vcf"]
        self.path_seeds_phylogeny = initial_seeds_config["user_input"]["path_seeds_phylogeny"]

        # SLiM_burnin_WF
        self.burn_in_Ne = initial_seeds_config["SLiM_burnin_WF"]["burn_in_Ne"]
        self.burn_in_generations_wf = initial_seeds_config["SLiM_burnin_WF"]["burn_in_generations"]
        self.burn_in_mutrate_wf = initial_seeds_config["SLiM_burnin_WF"]["burn_in_mutrate"]

        # SLiM_burnin_epi
        self.burn_in_generations_epi = \
            initial_seeds_config["SLiM_burnin_epi"]["burn_in_generations"]
        self.burn_in_mutrate_epi = initial_seeds_config["SLiM_burnin_epi"]["burn_in_mutrate"]
        self.seeded_host_id = initial_seeds_config["SLiM_burnin_epi"]["seeded_host_id"]
        self.S_IE_prob = initial_seeds_config["SLiM_burnin_epi"]["S_IE_prob"]
        self.E_I_prob = initial_seeds_config["SLiM_burnin_epi"]["E_I_prob"]
        self.E_R_prob = initial_seeds_config["SLiM_burnin_epi"]["E_R_prob"]
        self.latency_prob = initial_seeds_config["SLiM_burnin_epi"]["latency_prob"]
        self.I_R_prob = initial_seeds_config["SLiM_burnin_epi"]["I_R_prob"]
        self.I_E_prob = initial_seeds_config["SLiM_burnin_epi"]["I_E_prob"]
        self.R_S_prob = initial_seeds_config["SLiM_burnin_epi"]["R_S_prob"]
    
    def load_page(self):
        self.render_seeds_size(False, column=1, columnspan=3, frow=1)
        self.use_reference_control = self.render_use_reference(
            False, column=1, frow=4, columnspan=3)
        
        hide = True
        self.use_method_controls = self.render_use_method(
            hide=hide, column=1, frow=7, width=20, column_span=3)
        self.user_input_group_controls = self.init_user_input_group(hide)
        self.wf_group_controls = self.init_wf_group(hide)
        self.epi_group_controls = self.init_epi_group(hide)
        self.run_button_control = self.render_run_button(hide)

        if not self.use_reference:
            self.use_method_controls.rerender_itself()
            self.run_button_control.rerender_itself()
            match self.method:
                case "user_input":
                    self.user_input_group_controls.rerender_itself()
                case "SLiM_burnin_WF":
                    self.wf_group_controls.rerender_itself()
                case "SLiM_burnin_epi":
                    self.epi_group_controls.rerender_itself()
                case "":
                    self.run_button_control.derender_itself()
                case _:
                    raise ValueError("Invalid method")

        # Initial render logic for use reference controls
        renders = [self.use_method_controls]
        match self.method:
            case "user_input":
                renders.append(self.user_input_group_controls)
                renders.append(self.run_button_control)
            case "SLiM_burnin_WF":
                renders.append(self.wf_group_controls)
                renders.append(self.run_button_control)
            case "SLiM_burnin_epi":
                renders.append(self.epi_group_controls)
                renders.append(self.run_button_control)
            case "":
                pass
            case _:
                raise ValueError("Invalid method")
        use_reference_to_rerender = GroupControls(renders).rerender_itself
        use_reference_to_derender = GroupControls(renders).derender_itself
        self.use_reference_control.set_to_rerender(use_reference_to_rerender)
        self.use_reference_control.set_to_derender(use_reference_to_derender)

    def init_user_input_group(self, hide):
        grpctrs = GroupControls()
        grpctrs.add(self.render_path_seeds_vcf(hide, 1, 11, 1))
        grpctrs.add(self.render_path_seeds_phylogeny(hide, 1, 14, 1))
        return grpctrs

    def init_wf_group(self, hide):
        wf_grpctrls = GroupControls()
        wf_grpctrls.add(self.display_burn_in_settings_label(hide, 1, 9))
        wf_grpctrls.add(self.render_burn_in_ne(hide, 1, 11))
        wf_grpctrls.add(self.render_burn_in_generations_wf(hide, 2, 11))
        wf_grpctrls.add(self.render_subst_model_parameterization_wf(hide, 1, 13))
        self.mutrate_wf_control = self.render_mutrate_wf(hide, 1, 15)
        self.mutrate_matrix_wf_control = self.render_mutrate_matrix_wf(hide, 2, 15)
        wf_grpctrls.add(self.mutrate_wf_control)
        wf_grpctrls.add(self.mutrate_matrix_wf_control)
        wf_grpctrls.add(self.render_load_mutrate_wf(hide, 1, 19))
        return wf_grpctrls

    def init_epi_group(self, hide):
        epi_grpctrls = GroupControls()
        epi_grpctrls.add(self.display_burn_in_settings_label(hide, 0, 9))
        epi_grpctrls.add(self.render_seeded_host_id(hide, 0, 11))
        epi_grpctrls.add(self.render_burn_in_generations_epi(hide, 1, 11))
        epi_grpctrls.add(self.render_subst_model_parameterization_epi(hide, 2, 11))
        self.mutrate_epi_control = self.render_mutrate_epi(hide, 2, 13)
        self.mutrate_matrix_epi_control = self.render_mutrate_matrix_epi(hide, 3, 13)
        epi_grpctrls.add(self.mutrate_epi_control)
        epi_grpctrls.add(self.mutrate_matrix_epi_control)
        epi_grpctrls.add(self.render_load_mutrate_epi(hide, 2, 16))
        epi_grpctrls.add(self.render_S_IE_prob(hide, 0, 13))
        epi_grpctrls.add(self.render_E_I_prob(hide, 0, 15))
        epi_grpctrls.add(self.render_latency_prob(hide, 1, 13))
        epi_grpctrls.add(self.render_E_R_prob(hide, 1, 15))
        epi_grpctrls.add(self.render_I_E_prob(hide, 0, 17))
        epi_grpctrls.add(self.render_R_S_prob(hide, 0, 19))
        epi_grpctrls.add(self.render_I_R_prob(hide, 0, 21))

        epi_grpctrls.add(
            self.render_image(
                os.path.join(self.config_path, "../../../e3SIM_codes/gui/assets/t4.png"), 550, 255, hide, self.control_frame,
                frow=18, column=1, columnspan=2, rowspan=10))
        return epi_grpctrls

    def render_tab_title(self, hide=True, column=None, frow=None, columnspan=1):
        text = "Burn-in Settings:"
        component = EasyTitle(
            text,
            self.control_frame,
            column,
            frow,
            hide,
            columnspan,
        )
        self.visible_components.add(component)
        return component

    def render_seeds_size(self, hide, column, columnspan, frow):
        text = "Number of Seeding Pathogens (Integer)"
        keys_path = ["SeedsConfiguration", "seed_size"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Number of Seeding Pathogens",
            self.control_frame,
            column,
            frow,
            "integer",
            hide,
            columnspan,
            False,
            'w',
            "Number of seeding sequences (= number of initial patients)"
        )
        self.visible_components.add(component)
        return component

    def render_path_seeds_vcf(self, hide=True, column=None, frow=None, columnspan=1):
        text = "The vcf file of the seeding pathogen sequences (VCF format)"
        keys_path = ["SeedsConfiguration", "user_input", "path_seeds_vcf"]
        component = EasyPathSelector(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            hide,
            frow,
            columnspan,
            None,
            "Path to a VCF file that specifies the existing mutations of the seeding sequences."
            )
        self.visible_components.add(component)
        return component

    def render_path_seeds_phylogeny(self, hide=True, column=None, frow=None, columnspan=1):
        text = "The phylogeny of the seeding sequences (nwk format, optional)"
        keys_path = ["SeedsConfiguration", "user_input", "path_seeds_phylogeny"]
        component = EasyPathSelector(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            hide,
            frow,
            columnspan,
            None,
            "Phylogeny of the user-imported seeding sequences. They will be used in generating the final pathogen genealogy."
        )
        self.visible_components.add(component)
        return component

    def render_use_reference(
        self, hide=True, column=None, frow=None, columnspan=1):
        def radiobuttonselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
            if var.get():
                to_derender()
            else:
                to_rerender()

        text = "Do you want to use the same sequence (reference genome) as seeding sequences?"
        keys_path = ["SeedsConfiguration", "use_reference"]
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path,
            self.config_path,
            text,
            "use_reference",
            self.control_frame,
            column,
            frow,
            hide,
            to_rerender,
            to_derender,
            columnspan,
            radiobuttonselected,
            text=["Yes", "No (Run the burn-in process or provide seeding sequences)"],
        )

        self.visible_components.add(component)
        return component

    def render_use_method(
        self, hide=True, column=None, frow=None, width=20, column_span=20):
        def comboboxselected(var, to_rerender, to_derender):
            local_val = var.get()
            from_ui_mapping = {v: k for k, v in to_ui_mapping.items()}
            converted_val = from_ui_mapping.get(local_val, "")
            no_validate_update_val(converted_val, self.config_path, keys_path)

            # Set render logic for use reference control
            renders = [self.use_method_controls, self.run_button_control]
            match converted_val:
                case "user_input":
                    renders.append(self.user_input_group_controls)
                case "SLiM_burnin_WF":
                    renders.append(self.wf_group_controls)
                case "SLiM_burnin_epi":
                    renders.append(self.epi_group_controls)
                case _:
                    raise ValueError("Invalid method")
            use_reference_to_rerender = GroupControls(renders).rerender_itself
            use_reference_to_derender = GroupControls(renders).derender_itself
            self.use_reference_control.set_to_rerender(use_reference_to_rerender)
            self.use_reference_control.set_to_derender(use_reference_to_derender)

            # Call render logic for use method control
            match converted_val:
                case "user_input":
                    derender = [self.wf_group_controls, self.epi_group_controls]
                    GroupControls(derender).derender_itself()
                    self.user_input_group_controls.rerender_itself()
                case "SLiM_burnin_WF":
                    derender = [self.user_input_group_controls, self.epi_group_controls]
                    GroupControls(derender).derender_itself()
                    self.wf_group_controls.rerender_itself()
                case "SLiM_burnin_epi":
                    derender = [self.user_input_group_controls, self.wf_group_controls]
                    GroupControls(derender).derender_itself()
                    self.epi_group_controls.rerender_itself()
                case _:
                    raise ValueError("Invalid method")
            self.run_button_control.rerender_itself()

        text = "Method to Generate Sequences of the Seeding Pathogens"
        keys_path = ["SeedsConfiguration", "method"]
        to_rerender, to_derender = None, None
        to_ui_mapping = {
            "user_input": "User Input",
            "SLiM_burnin_WF": "Burn-in by a Wright-Fisher Model",
            "SLiM_burnin_epi": "Burn-in by an Epidemiological Model",
        }
        ui_values = list(to_ui_mapping.values())
        component = EasyCombobox(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            frow,
            ui_values,
            to_rerender,
            to_derender,
            comboboxselected,
            hide,
            width,
            column_span,
            to_ui_mapping,
        )
        self.visible_components.add(component)
        return component

    def display_burn_in_settings_label(self, hide=True, column=None, frow=None):
        component = EasyTitle("Burn-in Settings: ", self.control_frame, column, frow, hide, 1)
        self.visible_components.add(component)
        return component
    
    ##########################################################################
    ####### wf
    ##########################################################################

    def render_burn_in_ne(self, hide=True, column=None, frow=None):
        columnspan = 1
        text = "Effective Population Size (Integer)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_WF", "burn_in_Ne"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Effective Population Size",
            self.control_frame,
            column,
            frow,
            "integer",
            hide,
            columnspan,
        )
        self.visible_components.add(component)
        return component

    def render_burn_in_generations_wf(self, hide=True, column=None, frow=None):
        columnspan = 1
        text = "Number of Burn-in Generations (Integer)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_WF", "burn_in_generations"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Number of Burn-in Generations (WF)",
            self.control_frame,
            column,
            frow,
            "integer",
            hide,
            columnspan,
        )
        self.visible_components.add(component)
        return component
    
    def render_subst_model_parameterization_wf(self, hide=True, column=None, frow=None):
        def comboboxselected(var, to_rerender, to_derender):
            local_var = var.get()
            no_validate_update_val(local_var, self.config_path, keys_path)
            if local_var == "mutation rate (single)":
                self.mutrate_wf_control.label.config(state="normal")
                self.mutrate_wf_control.entry.config(state="normal", foreground="black")
                self.mutrate_matrix_wf_control.label.config(state="disabled")
                for i in range(4):
                    for j in range(4):
                        self.mutrate_matrix_wf_control.matrix_entries[i][j].config(
                            state="disabled", foreground="light grey")
            elif local_var == "mutation rate matrix":
                self.mutrate_wf_control.label.config(state="disabled")
                self.mutrate_wf_control.entry.config(state="disabled", foreground="light grey")
                self.mutrate_matrix_wf_control.label.config(state="normal")
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            self.mutrate_matrix_wf_control.matrix_entries[i][j].config(
                                state="normal", foreground="black")

        text = "Substitution Model Parameterization"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_WF", "subst_model_parameterization"]
        to_rerender, to_derender = None, None
        width = 20
        columnspan = 1
        component = EasyCombobox(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            frow,
            ["mutation rate (single)", "mutation rate matrix"],
            to_rerender,
            to_derender,
            comboboxselected,
            hide,
            width,
            columnspan,
            None,
            False,
            "Substitution parameterization for the burn-in process, using the same format as the formal simulation, but you can specify different rate values / model as needed."
        )
        self.visible_components.add(component)
        return component

    def render_mutrate_wf(self, hide=True, column=None, frow=None):
        columnspan = 1
        text = "Mutation Rate (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_WF", "burn_in_mutrate"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Mutation Rate",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            columnspan,
            disabled=True,
            sticky="wn",
        )
        self.visible_components.add(component)
        return component

    def render_mutrate_matrix_wf(self, hide=True, column=None, frow=None):
        columnspan = 1
        self.render_burn_in_mutrate_wf_text = "Mutation Rate Matrix (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_WF", "burn_in_mutrate_matrix"]
        component = EasyRateMatrix(
            keys_path,
            self.config_path,
            self.render_burn_in_mutrate_wf_text,
            "Mutation Rate Matrix (WF)",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            columnspan,
            disabled=True,
        )
        self.visible_components.add(component)
        return component
    
    def render_load_mutrate_wf(self, hide=True, column=None, frow=None):
        def click():
            config = load_config_as_dict(self.config_path)["EvolutionModel"]
            self.mutrate_wf_control.entry.delete(0, tk.END)
            self.mutrate_wf_control.entry.insert(0, str(config["mut_rate"]))
            for i in range(4):
                for j in range(4):
                    self.mutrate_matrix_wf_control.matrix_entries[i][j].delete(0, tk.END)
                    self.mutrate_matrix_wf_control.matrix_entries[i][j].insert(
                        0, str(config["mut_rate_matrix"][i][j]))

        text = "Load mut. rate(s) \n from Evolutionary Model"
        component = EasyButton(
            text,
            self.control_frame,
            column,
            frow,
            click,
            hide,
            sticky="w"
        )
        self.visible_components.add(component)
        return component

    ##########################################################################
    ####### epi
    ##########################################################################

    def render_burn_in_generations_epi(self, hide=True, column=None, frow=None):
        self.render_burn_in_generations_epi_text = ("Number of Burn-in Generations (Integer)")
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "burn_in_generations"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            self.render_burn_in_generations_epi_text,
            "Number of Burn-in Generations (epi)",
            self.control_frame,
            column,
            frow,
            "integer",
            hide,
            1,
        )
        self.visible_components.add(component)
        return component

    def render_subst_model_parameterization_epi(self, hide=True, column=None, frow=None):
        def comboboxselected(var, to_rerender, to_derender):
            local_var = var.get()
            no_validate_update_val(local_var, self.config_path, keys_path)
            if local_var == "mutation rate (single)":
                self.mutrate_epi_control.label.config(state="normal")
                self.mutrate_epi_control.entry.config(state="normal", foreground="black")
                self.mutrate_matrix_epi_control.label.config(state="disabled")
                for i in range(4):
                    for j in range(4):
                        self.mutrate_matrix_epi_control.matrix_entries[i][j].config(
                            state="disabled", foreground="light grey")
            elif local_var == "mutation rate matrix":
                self.mutrate_epi_control.label.config(state="disabled")
                self.mutrate_epi_control.entry.config(state="disabled", foreground="light grey")
                self.mutrate_matrix_epi_control.label.config(state="normal")
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            self.mutrate_matrix_epi_control.matrix_entries[i][j].config(
                                state="normal", foreground="black")

        text = "Substitution Model Parameterization"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "subst_model_parameterization"]
        to_rerender, to_derender = None, None
        width = 20
        column_span = 1
        component = EasyCombobox(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            frow,
            ["mutation rate (single)", "mutation rate matrix"],
            to_rerender,
            to_derender,
            comboboxselected,
            hide,
            width,
            column_span,
            None,
            False,
            "Substitution parameterization for the burn-in process, using the same format as the formal simulation, but you can specify different rate values / model as needed."
        )
        self.visible_components.add(component)
        return component

    def render_mutrate_epi(self, hide=True, column=None, frow=None):
        text = "Mutation Rate (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "burn_in_mutrate"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Mutation Rate (epi)",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            1,
            disabled=True
        )
        self.visible_components.add(component)
        return component
    
    def render_mutrate_matrix_epi(self, hide=True, column=None, frow=None):
        columnspan = 1
        text = "Mutation Rate Matrix (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "burn_in_mutrate_matrix"]
        component = EasyRateMatrix(
            keys_path,
            self.config_path,
            text,
            "Mutation Rate Matrix (epi)",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            columnspan=columnspan,
            disabled=True
        )
        self.visible_components.add(component)
        return component
    
    def render_load_mutrate_epi(self, hide=True, column=None, frow=None):
        def click():
            config = load_config_as_dict(self.config_path)["EvolutionModel"]
            self.mutrate_epi_control.entry.delete(0, tk.END)
            self.mutrate_epi_control.entry.insert(0, str(config["mut_rate"]))
            for i in range(4):
                for j in range(4):
                    self.mutrate_matrix_epi_control.matrix_entries[i][j].delete(0, tk.END)
                    self.mutrate_matrix_epi_control.matrix_entries[i][j].insert(
                        0, str(config["mut_rate_matrix"][i][j]))

        text = "Load mut. rate(s) \n from Evolutionary Model"
        component = EasyButton(
            text,
            self.control_frame,
            column,
            frow,
            click,
            hide,
            sticky="w"
        )
        self.visible_components.add(component)
        return component

    def render_seeded_host_id(self, hide=True, column=None, frow=None):
        text = "Seeded Host ID(s) (List Integer)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "seeded_host_id"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Seeded Host IDs",
            self.control_frame,
            column,
            frow,
            "list integer",
            hide,
            1,
            False,
            "w",
            "Patient 0's IDs, following the ID in the host population network",
        )
        self.visible_components.add(component)
        return component

    def render_S_IE_prob(self, hide=True, column=None, frow=None):
        text = "Transmission Prob. β (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "S_IE_prob"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Transmission Prob.",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            1,
            False,
            "w",
            "Probability of transmission (S->E/I) for each connected S-I host pair per tick."
        )
        self.visible_components.add(component)
        return component

    def render_E_R_prob(self, hide=True, column=None, frow=None):
        self.render_E_R_prob_text = "Latent Recovery Prob. τ (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "E_R_prob"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            self.render_E_R_prob_text,
            "Latent Recovery Prob.",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            1,
            False,
            "w",
            "Probability of recovery (E->R) for each exposed host per tick."
        )
        self.visible_components.add(component)
        return component

    def render_latency_prob(self, hide=True, column=None, frow=None):
        text = "Latency Prob. ζ (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "latency_prob"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Latency Prob.",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            1,
            False,
            "w",
            "Probability of being in the exposed state once succcessfully infected by another host."
        )
        self.visible_components.add(component)
        return component

    def render_E_I_prob(self, hide=True, column=None, frow=None):
        text = "Activation Prob. v (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "E_I_prob"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Activation Prob.",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            1,
            False,
            "w",
            "Probability of activation (E->I) for each exposed host per tick."
        )
        self.visible_components.add(component)
        return component

    def render_I_E_prob(self, hide=True, column=None, frow=None):
        text = "De-activation Prob. φ (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "I_E_prob"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "De-activation Prob.",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            1,
            False,
            "w",
            "Probability of deactivation (I->E) for each infected host per tick."
        )
        self.visible_components.add(component)
        return component

    def render_R_S_prob(self, hide=True, column=None, frow=None):
        text = "Immunity Loss Prob. ω (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "R_S_prob"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Immunity Loss Prob.",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            1,
            False,
            "w",
            "Probability of immunity loss (R->S) for each recovered host per tick."
        )
        self.visible_components.add(component)
        return component

    def render_I_R_prob(self, hide=True, column=None, frow=None):
        text = "Active Recovery Prob. γ (Numerical)"
        keys_path = ["SeedsConfiguration", "SLiM_burnin_epi", "I_R_prob"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Active Recovery Prob.",
            self.control_frame,
            column,
            frow,
            "numerical",
            hide,
            1,
            False,
            "w",
            "Probability of recovery (I->R) for each infected host per tick."
        )
        self.visible_components.add(component)
        return component

    def render_image(
            self, image_path, desired_width, desired_height, hide, 
            control_frame, frow, column, columnspan, rowspan):
        component = EasyImage(
            image_path,
            desired_width,
            desired_height,
            hide,
            control_frame,
            frow,
            column,
            columnspan,
            rowspan,
        )
        return component

    def render_run_button(self, hide):
        def seed_generation():
            # Try to update entry values
            if self.global_update() == 1:
                return
            
            config = load_config_as_dict(self.config_path)
            cwdir = config["BasicRunConfiguration"]["cwdir"]
            rand_seed = config["BasicRunConfiguration"]["random_number_seed"]
            ref_path = config["GenomeElement"]["ref_path"]

            seeds_config = config["SeedsConfiguration"]
            seed_size = seeds_config["seed_size"]
            method = seeds_config["method"]
            try:
                if method == "SLiM_burnin_WF":
                    WF_config = seeds_config["SLiM_burnin_WF"]
                    Ne = WF_config["burn_in_Ne"]
                    n_gen = WF_config["burn_in_generations"]
                    mu = WF_config["burn_in_mutrate"]
                    if WF_config["subst_model_parameterization"] == "":
                        messagebox.showerror(
                            "Error", "Please select a substitution model parameterization")
                        return
                    use_subst_matrix = \
                        WF_config["subst_model_parameterization"] == "mutation rate matrix"
                    mu_matrix = WF_config["burn_in_mutrate_matrix"]
                    mu_matrix = json.dumps({"A": mu_matrix[0], "C": mu_matrix[1], 
                                            "G": mu_matrix[2], "T": mu_matrix[3]})
                    try:
                        config = SeedConfig(
                            method=method,
                            wk_dir=cwdir,
                            seed_size=seed_size,
                            Ne=Ne,
                            mu=mu,
                            n_gen=n_gen,
                            ref_path=ref_path,
                            rand_seed=rand_seed,
                            use_subst_matrix=use_subst_matrix, 
                            mu_matrix=mu_matrix
                        )
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to initialize Seed Generator Configuration:\n{e}")
                        return
                    generator = SeedGenerator(config)
                    error = generator.run()
                    # error = run_seed_generation(
                    #     method=method,
                    #     wk_dir=cwdir,
                    #     seed_size=seed_size,
                    #     Ne=Ne,
                    #     mu=mu,
                    #     n_gen=n_gen,
                    #     ref_path=ref_path,
                    #     rand_seed=rand_seed,
                    #     use_subst_matrix=use_subst_matrix, 
                    #     mu_matrix=mu_matrix
                    # )
                elif method == "SLiM_burnin_epi":
                    epi_config = seeds_config["SLiM_burnin_epi"]
                    seeded_host_id = epi_config["seeded_host_id"]
                    n_gen = epi_config["burn_in_generations"]
                    mu = epi_config["burn_in_mutrate"]
                    if epi_config["subst_model_parameterization"] == "":
                        messagebox.showerror(
                            "Error", "Please select a substitution model parameterization")
                        return
                    use_subst_matrix = \
                        epi_config["subst_model_parameterization"] == "mutation rate matrix"
                    mu_matrix = epi_config["burn_in_mutrate_matrix"]
                    mu_matrix = json.dumps({"A": mu_matrix[0], "C": mu_matrix[1], 
                                            "G": mu_matrix[2], "T": mu_matrix[3]})
                    S_IE_prob = epi_config["S_IE_prob"]
                    E_I_prob = epi_config["E_I_prob"]
                    E_R_prob = epi_config["E_R_prob"]
                    latency_prob = epi_config["latency_prob"]
                    I_R_prob = epi_config["I_R_prob"]
                    I_E_prob = epi_config["I_E_prob"]
                    R_S_prob = epi_config["R_S_prob"]
                    host_size = config["NetworkModelParameters"]["host_size"]
                    # error = run_seed_generation(
                    #     method=method,
                    #     wk_dir=cwdir,
                    #     seed_size=seed_size,
                    #     use_subst_matrix=use_subst_matrix, 
                    #     mu=mu,
                    #     mu_matrix=mu_matrix,
                    #     n_gen=n_gen,
                    #     seeded_host_id=seeded_host_id,
                    #     S_IE_prob=S_IE_prob,
                    #     E_I_prob=E_I_prob,
                    #     E_R_prob=E_R_prob,
                    #     latency_prob=latency_prob,
                    #     I_R_prob=I_R_prob,
                    #     I_E_prob=I_E_prob,
                    #     R_S_prob=R_S_prob,
                    #     host_size=host_size,
                    #     ref_path=ref_path,
                    #     rand_seed=rand_seed
                    # )
                    try:
                        config = SeedConfig(
                            method=method,
                            wk_dir=cwdir,
                            seed_size=seed_size,
                            use_subst_matrix=use_subst_matrix, 
                            mu=mu,
                            mu_matrix=mu_matrix,
                            n_gen=n_gen,
                            seeded_host_id=seeded_host_id,
                            S_IE_prob=S_IE_prob,
                            E_I_prob=E_I_prob,
                            E_R_prob=E_R_prob,
                            latency_prob=latency_prob,
                            I_R_prob=I_R_prob,
                            I_E_prob=I_E_prob,
                            R_S_prob=R_S_prob,
                            host_size=host_size,
                            ref_path=ref_path,
                            rand_seed=rand_seed
                        )
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to initialize Seed Generator Configuration:\n{e}")
                        return
                    generator = SeedGenerator(config)
                    error = generator.run()
                elif method == "user_input":
                    path_seeds_vcf = seeds_config["user_input"]["path_seeds_vcf"]
                    path_seeds_phylogeny = seeds_config["user_input"]["path_seeds_phylogeny"]
                    # error = run_seed_generation(
                    #     method=method,
                    #     wk_dir=cwdir,
                    #     seed_size=seed_size,
                    #     seed_vcf=path_seeds_vcf,
                    #     path_seeds_phylogeny=path_seeds_phylogeny,
                    #     rand_seed=rand_seed
                    # )
                    try:
                        config = SeedConfig(
                            method=method,
                            wk_dir=cwdir,
                            seed_size=seed_size,
                            seed_vcf=path_seeds_vcf,
                            path_seeds_phylogeny=path_seeds_phylogeny,
                            rand_seed=rand_seed
                        )
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to initialize Seed Generator Configuration:\n{e}")
                        return
                    generator = SeedGenerator(config)
                    error = generator.run()

                if error is not None:
                    raise Exception(error)
                
                messagebox.showinfo("Success", "Seed generation completed successfully!")

            except Exception as e:
                messagebox.showerror("Seed Generation Error", str(e))

        column = 1
        frow = 100
        component = EasyButton(
            "Run Seed Generation",
            self.control_frame,
            column,
            frow,
            seed_generation,
            hide,
            "",
        )
        self.visible_components.add(component)
        return component
