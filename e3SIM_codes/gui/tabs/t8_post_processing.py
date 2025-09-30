from tkinter import messagebox
import os
import shutil
import sys
from outbreak_simulator import create_slim_config
from utils import (load_config_as_dict, save_config, no_validate_update, 
                   TabBase, EasyCombobox, EasyRadioButton, EasyEntry, EasyButton)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

class PostProcessing(TabBase):
    def __init__(self, parent, tab_parent, config_path, tab_title, tab_index, hide=False):
        super().__init__(parent, tab_parent, config_path, tab_title, tab_index, hide, False)
    
    def init_val(self, config_path):
        self.config_path = config_path

    def load_page(self):
        hide = False
        self.render_do_postprocess(hide, 0, 1)
        self.render_branch_color_trait(hide, 0, 4)
        self.render_heatmap(hide, 0, 6)
        self.render_vcf(hide, 0, 8)
        self.render_fasta(hide, 0, 11)
        self.render_config_button(hide, 0, 14)

    def render_do_postprocess(self, hide, column, frow, columnspan=1):
        def radiobuttonselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
            if var.get():
                self.render_branch_color_trait(hide, 0, 4, disabled=False)
                self.render_heatmap(hide, 0, 6, disabled=False)
                self.render_vcf(hide, 0, 8, disabled=False)
                self.render_fasta(hide, 0, 11, disabled=False)
            else:
                self.render_branch_color_trait(hide, 0, 4, disabled=True)
                self.render_heatmap(hide, 0, 6, disabled=True)
                self.render_vcf(hide, 0, 8, disabled=True)
                self.render_fasta(hide, 0, 11, disabled=True)

        to_rerender, to_derender = None, None
        keys_path = ["Postprocessing_options", "do_postprocess"]
        text = "Run Post-processing After the Simulation?\n"\
            "(Plotting Trajectories & Transmission trees, and Outputting VCF/FASTA Files.)"
        component = EasyRadioButton(
            keys_path,
            self.config_path,
            text,
            "Do Post-processing",
            self.control_frame,
            column,
            frow,
            hide,
            to_rerender,
            to_derender,
            columnspan,
            radiobuttonselected,
        )
        self.visible_components.add(component)
        return component

    def render_branch_color_trait(self, hide, column, frow, columnspan=1, disabled=False):
        text = ("Which Trait Do You Want to Use for Coloring the Branches "
                "in the Transmission Tree Plot (Integer)")
        keys_path = ["Postprocessing_options","tree_plotting","branch_color_trait"]
        component = EasyEntry(
            keys_path,
            self.config_path,
            text,
            "Branch Color Trait",
            self.control_frame,
            column,
            frow,
            "integer",
            hide,
            columnspan,
            disabled,
        )
        self.visible_components.add(component)
        self.branch_color_trait_control = component
        return component

    def render_heatmap(self, hide, column, frow, columnspan=1, disabled=False):
        def comboboxselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
            
        keys_path = ["Postprocessing_options","tree_plotting","heatmap",]
        text = ("Plot a Heatmap for a Fitness Effect of all Sampled Genomes "
                "in the Transmission Tree Plot?")
        to_rerender, to_derender = None, None
        width = 20
        component = EasyCombobox(
            keys_path,
            self.config_path,
            text,
            self.control_frame,
            column,
            frow,
            ["none", "transmissibility", "drug_resistance"],
            to_rerender,
            to_derender,
            comboboxselected,
            hide,
            width,
            columnspan,
        )
        if disabled:
            component.label.configure(state="disabled")
            component.combobox.configure(state="disabled")
        self.visible_components.add(component)
        return component
    
    def render_vcf(self, hide, column, frow, columnspan=1, disabled=False):
        def radiobuttonselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
        
        text = "Output a VCF file for sampled pathogens?"
        keys_path = ["Postprocessing_options", "sequence_output", "vcf"]
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path,
            self.config_path,
            text,
            "Output VCF",
            self.control_frame,
            column,
            frow,
            hide,
            to_rerender,
            to_derender,
            columnspan,
            radiobuttonselected,
            disabled,
        )
        self.visible_components.add(component)
        return component
    
    def render_fasta(self, hide, column, frow, columnspan=1, disabled=False):
        def radiobuttonselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)

        text = "Output a FASTA file of concatenated SNPs for sampled pathogens?"
        keys_path = ["Postprocessing_options", "sequence_output", "fasta"]
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path,
            self.config_path,
            text,
            "Output FASTA",
            self.control_frame,
            column,
            frow,
            hide,
            to_rerender,
            to_derender,
            columnspan,
            radiobuttonselected,
            disabled,
        )
        self.visible_components.add(component)
        return component

    def render_config_button(self, hide, column, frow, columnspan=1):
        component = EasyButton(
            "Generate the configuration file",
            self.control_frame,
            column,
            frow,
            self.generate_config_file,
            hide,
            "",
        )
        self.visible_components.add(component)
        return component

    def generate_config_file(self):
        config = load_config_as_dict(self.config_path)
        if config["Postprocessing_options"]["do_postprocess"]:
            try:
                val = int(self.branch_color_trait_control.entry.get())
            except ValueError:
                messagebox.showerror(
                    "Value Error", "Please enter a valid integer for trait number")
                return
            config["Postprocessing_options"]["tree_plotting"]["branch_color_trait"] = val
            save_config(self.config_path, config)

        cwdir = config["BasicRunConfiguration"]["cwdir"]
        target = os.path.join(cwdir, "config_file.json")
        try:
            create_slim_config(config)
            messagebox.showinfo("Configuration check Passed.")
        except IOError as e:
            print("Errors checking configuration: %s" % e)
        try:
            shutil.copy(self.config_path, target)
            messagebox.showinfo("Success", "Config file saved successfully")
            print("********************************************************************")
            print("              Config file is saved successfully to")
            print(target)
            print("  You can now run the simulation (OutbreakSimulator.py) in terminal.")
            print("* WE RECOMMEND NOT TO CLOSE THE GUI UNLESS YOU ARE SURE ABOUT YOUR INPUT VALUES*")
            print("   *OUTBREAKSIMULATOR.PY WILL CHECK VALIDITY OF YOUR CONFIG FILE.*")
            print("                 *IF IT DIDN'T PASS THE CHECK,*")
            print(    "*FEEL FREE TO GO BACK TO PREVIOUS TABS TO MAKE ANY EDITS.*")
            print("********************************************************************")
        except IOError as e:
            print("Unable to copy file. %s" % e)
