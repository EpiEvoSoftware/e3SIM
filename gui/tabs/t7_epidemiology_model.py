import tkinter as tk
from tkinter import ttk
import platform
from utils import (no_validate_update, TabBase, EasyPathSelector, EasyTitle,
                   EasyCombobox, EasyRadioButton, EasyEntry, EasyImage, CreateToolTip)

class EpidemiologyModel(TabBase):
    def __init__(self, parent, tab_parent, config_path, tab_title, tab_index, hide=False):
        super().__init__(parent, tab_parent, config_path, tab_title, tab_index, hide)

    def init_val(self, config_path):
        self.config_path = config_path

    def init_tab(self, parent, tab_parent, tab_title, tab_index, hide):
        self.parent = parent
        self.tab_parent = tab_parent
        self.tab_index = tab_index
        self.tab_parent.add(parent, text=tab_title)
        if hide:
            self.tab_parent.tab(self.tab_index, state="disabled")
        self.control_frame = ttk.Frame(self.parent, width=300)
        self.control_frame.pack(expand=1, fill='y')
        
        # Enable scrolling
        self.canvas = tk.Canvas(self.control_frame)
        self.scrollbar = ttk.Scrollbar(self.control_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set, width=1300, height=800)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        for i in range(4):  
            self.scrollable_frame.columnconfigure(i, weight=1)

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        def configure_frame(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        def configure_canvas(event):
            self.canvas.itemconfig(self.canvas_frame, width=event.width)
        
        def _bound_to_mousewheel(event):
            if platform.system() == 'Linux':
                self.canvas.bind_all("<Button-4>", _on_mousewheel)
                self.canvas.bind_all("<Button-5>", _on_mousewheel)
            else:
                self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbound_to_mousewheel(event):
            if platform.system() == 'Linux':
                self.canvas.unbind_all("<Button-4>")
                self.canvas.unbind_all("<Button-5>")
            else:
                self.canvas.unbind_all("<MouseWheel>")

        def _on_mousewheel(event):
            if platform.system() == 'Windows':
                self.canvas.yview_scroll(int(-1* (event.delta/120)), "units")
            elif platform.system() == 'Darwin':
                self.canvas.yview_scroll(int(-1 * (event.delta)), "units")
            else:
                if event.num == 4:
                    self.canvas.yview_scroll( -1, "units" )
                elif event.num == 5:
                    self.canvas.yview_scroll( 1, "units" ) 
        
        self.scrollable_frame.bind("<Configure>", configure_frame)
        self.canvas.bind("<Configure>", configure_canvas)
        self.scrollable_frame.bind('<Enter>', _bound_to_mousewheel)
        self.scrollable_frame.bind('<Leave>', _unbound_to_mousewheel)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

    def load_page(self):
        def frow(increment=True, by=2):
            '''
            Helper function that manages frow
            '''
            if increment:
                self.frow += by
            return self.frow
        
        self.frow = 1
        hide = False

        self.SEIR_only_entries = set()
        self.render_model(hide, 0, 1, frow(increment=False))
        self.render_slim_path_label(hide, 1, 1, frow(increment=False))
        self.render_n_epoch(hide, 0, 1, frow(by=3))
        self.render_epoch_changing_generation(hide, 1, 1, frow(increment=False))

        self.render_title("Evolutionary Components Setting", hide, 0, 1, frow())
        self.render_transmissibility(hide, 0, 1, frow(by=1))
        self.render_cap_transmissibility(hide, 1, 1, frow(increment=False))
        self.render_drug_resistance(hide, 0, 1, frow())
        self.render_cap_drugresist(hide, 1, 1, frow(increment=False))

        self.render_title("Transition Probabilities between Compartments", hide, 0, 1, frow())
        self.render_S_IE_prob(hide, 0, 1, frow(by=1))
        self.render_latency_prob(hide, 1, 1, frow(increment=False))
        self.render_E_I_prob(hide, 0, 1, frow())
        self.render_E_R_prob(hide, 1, 1, frow(increment=False))
        self.render_I_E_prob(hide, 0, 1, frow())
        self.render_I_R_prob(hide, 0, 1, frow())
        self.render_R_S_prob(hide, 0, 1, frow())
        self.render_sample_prob(hide, 0, 1, frow())
        self.render_sampling_recovery_prob(hide, 0, 1, frow())
        self.render_image(hide, "gui/assets/t7.png", 500, 300, 1, 1, 17)

        self.render_title("Massive Sampling Events", hide, 0, 1, frow(),)
        self.render_massive_sampling(hide, 0, 1, frow(by=1))
        self.render_massive_sampling_generation(hide, 0, 1, frow())
        self.render_massive_sampling_probability(hide, 0, 2, frow())
        self.render_massive_sampling_after_sampling(hide, 0, 1, frow())
        self.render_super_infection(hide, 0, 1, frow())

    def render_model(self, hide, column, columnspan, frow):
        def comboboxselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)
            #Toggle SIR/SEIR Models
            if var.get() == "SEIR":
                for entry in self.SEIR_only_entries:
                    entry.config(state='normal', foreground='black')
            elif var.get() == "SIR":
                for entry in self.SEIR_only_entries:
                    entry.config(state='disabled', foreground='light grey')
            
        text = "Compartmental Model"
        keys_path = ['EpidemiologyModel', 'model']
        values = ["SEIR", "SIR"]
        to_rerender, to_derender = None, None
        width=20
        component = EasyCombobox(
            keys_path, self.config_path, text, self.scrollable_frame,
            column, frow, values, to_rerender, to_derender,
            comboboxselected, hide, width, columnspan
        )
        self.visible_components.add(component)
        return component
    
    def render_slim_path_label(self, hide, column, columnspan, frow):
        text = "Slim Replicate Seed File Path"
        keys_path = ['EpidemiologyModel', 'slim_replicate_seed_file_path']
        component = EasyPathSelector(
            keys_path,self.config_path, text,
            self.scrollable_frame, column, hide, frow, columnspan,
            labtext="You can upload a text file with each row being one seed number, and the number of rows matching the number of simulation replicates specified in tab1, such that each specified random seed is used in each replicate."
        )
        self.visible_components.add(component)
        return component
    
    def render_n_epoch(self, hide, column, columnspan, frow):
        text = "Number of Epochs (Integer)"
        keys_path = ['EpidemiologyModel','epoch_changing','n_epoch']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Number of Epochs',
            self.scrollable_frame, column, frow, 'integer', hide, columnspan,
            labtext="In each epoch, one set of parameters can be used."
        )
        self.visible_components.add(component)
        return component
    
    def render_epoch_changing_generation(self, hide, column, columnspan, frow):
        text = "On Which Generation(s) Should the Simulation Go to the Next Epoch (List integer)"
        keys_path = ['EpidemiologyModel','epoch_changing','epoch_changing_generation']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Epoch-Changing Generation(s)',
            self.scrollable_frame, column, frow, 'list integer', hide, columnspan,
            labtext="Once the epoch is changed, all the parameters are shifted to the next set as specified in the list."
        )
        self.visible_components.add(component)
        return component

####################################
### Genetic Architecture
####################################

    def render_title(self, text, hide, column, columnspan, frow):
        component = EasyTitle(
            text, self.scrollable_frame, column, frow, hide, columnspan, pady=(25, 5))
        self.visible_components.add(component)
        return component

    def render_transmissibility(self, hide, column, columnspan, frow):
        text = "Which Transmissibility Trait to Use for Each Epoch (List integer)"
        keys_path = ['EpidemiologyModel','genetic_architecture','transmissibility']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Which Transmissibility Trait',
            self.scrollable_frame, column, frow, 'list integer', hide, columnspan,
            labtext="Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order). Each entry has to be an integer within the range specified in the \"genome element\" tab. Specifying \"0\" would mean no genetic effect for transmissibility in this epoch."
        )
        self.visible_components.add(component)
        return component

    def render_cap_transmissibility(self, hide, column, columnspan, frow):
        text = "The Maximum Transmissibility for Each Epoch (List Numerical)"
        keys_path = ['EpidemiologyModel','genetic_architecture','cap_transmissibility']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Max transmissibility', 
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan
        )
        self.visible_components.add(component)
        return component

    def render_drug_resistance(self, hide, column, columnspan, frow):
        text = "Which Drug-resistance Trait to Use for Each Epoch (List Integer)"
        keys_path = ['EpidemiologyModel','genetic_architecture','drug_resistance']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Which Drug-resistance Trait',
            self.scrollable_frame, column, frow, 'list integer', hide, columnspan,
            labtext="Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order). Each entry has to be an integer within the range specified in the \"genome element\" tab. Specifying \"0\" would mean no genetic effect for drug-resistance in this epoch."
        )
        self.visible_components.add(component)
        return component

    def render_cap_drugresist(self, hide, column, columnspan, frow):
        text = "The Maximum Drug-resistance for Each Epoch (List Numerical)"
        keys_path = ['EpidemiologyModel','genetic_architecture','cap_drugresist']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Max Drug-resistance',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan
        )
        self.visible_components.add(component)
        return component

####################################
### Transition Probabilities
####################################

    def render_S_IE_prob(self, hide, column, columnspan, frow):
        text = "Transmission Prob. β (List Numerical)"
        keys_path = ['EpidemiologyModel','transition_prob','S_IE_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Transmission Prob. β',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of transmission (S->E/I) for each connected S-I host pair per tick. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        return component

    def render_I_R_prob(self, hide, column, columnspan, frow):
        text = "Active Recovery Prob. γ (List Numerical)"
        keys_path = ['EpidemiologyModel','transition_prob','I_R_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Active Recovery Prob. γ', 
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of recovery (I->R) for each infected host per tick. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        return component

    def render_R_S_prob(self, hide, column, columnspan, frow):
        text = "Immunity Loss Prob. ω (List Numerical)"
        keys_path = ['EpidemiologyModel','transition_prob','R_S_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Immunity Loss Prob. ω',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of immunity loss (R->S) for each recovered host per tick. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        return component

    def render_latency_prob(self, hide, column, columnspan, frow):
        text = "Latency Probability ζ (List Numerical)"
        keys_path = ['EpidemiologyModel','transition_prob','latency_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Latency Probability ζ',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of being in the exposed state once succcessfully infected by another host. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        self.SEIR_only_entries.add(component.entry)
        return component
      
    def render_E_I_prob(self, hide, column, columnspan, frow):
        text = "Activation Prob. v (List Numerical)"
        keys_path = ['EpidemiologyModel','transition_prob','E_I_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Activation Prob. v',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of activation (E->I) for each recovered host per tick. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        self.SEIR_only_entries.add(component.entry)
        return component 

    def render_I_E_prob(self, hide, column, columnspan, frow):
        text = r'De-activation Prob. φ (List Numerical)'
        keys_path = ['EpidemiologyModel','transition_prob','I_E_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'De-activation Prob. φ',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of de-activation (I->E) for each recovered host per tick. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        self.SEIR_only_entries.add(component.entry)
        return component

    def render_E_R_prob(self, hide, column, columnspan, frow):
        text = "Latent Recovery Prob. τ (List Numerical)"
        keys_path = ['EpidemiologyModel','transition_prob','E_R_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Latent Recovery Prob. τ',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of recovery (E->R) for each exposed host per tick. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        self.SEIR_only_entries.add(component.entry)
        return component 
        
    def render_sample_prob(self, hide, column, columnspan, frow):
        text = r"Sample Prob. ε (List Numerical)"
        keys_path = ['EpidemiologyModel','transition_prob','sample_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Sample Prob. ε',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of being sampled for each infected host per tick. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        return component

    def render_sampling_recovery_prob(self, hide, column, columnspan, frow):
        text = r"Post-sampling Recovery Prob. δ (List Numerical)"
        keys_path = ['EpidemiologyModel','transition_prob','recovery_prob_after_sampling']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Post-sampling Recovery Prob. δ',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan,
            labtext="Probability of instant recovery for each sampled host. Has to be a list whose length is the number of epochs specified above, and each entry represents the parameter for one epoch (in time-order)."
            )
        self.visible_components.add(component)
        return component
    
    def render_image(
            self, hide, image_path, desired_width, desired_height, column, columnspan, frow):
        component = EasyImage(
            image_path, desired_width, desired_height, hide,
            self.scrollable_frame, frow, column, columnspan, rowspan=10
            )
        self.visible_components.add(component)
        return component

####################################
### Massive Sampling
####################################

    def render_massive_sampling(self, hide, column, columnspan, frow):
        text = "How Many Massive Sampling Events Will Be Simulated (Integer)"
        keys_path = ['EpidemiologyModel','massive_sampling','event_num']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Num Massive Sampling Events',
            self.scrollable_frame, column, frow, 'integer', hide, columnspan,
            labtext="Simultaneous sampling at specific ticks."
            )
        self.visible_components.add(component)
        return component   

    def render_massive_sampling_generation(self, hide, column, columnspan, frow):
        text = ("In Which Generation (s) Will the "
                "Massive Sampling Events be Simulated (List Integer)")
        keys_path = ['EpidemiologyModel','massive_sampling','generation']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Massive Sampling Generations',
            self.scrollable_frame, column, frow, 'list integer', hide, columnspan
            )
        self.visible_components.add(component)
        return component
    
    def render_massive_sampling_probability(self, hide, column, columnspan, frow):
        text = ("Probability of Being Sampled for Every Pathogen Genome "
                "in Each Massive Sample Event (List Numerical)")
        keys_path = ['EpidemiologyModel','massive_sampling','sampling_prob']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Massive Sampling Probability',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan
            )
        self.visible_components.add(component)
        return component

    def render_massive_sampling_after_sampling(self, hide, column, columnspan, frow):
        text = ("Probability of Recovering Once Sampled "
                "in Each Massive Sampling Event (List Numerical)")
        keys_path = ['EpidemiologyModel','massive_sampling','recovery_prob_after_sampling']
        component = EasyEntry(
            keys_path, self.config_path, text, 'Massive Sampling Recovery Probability',
            self.scrollable_frame, column, frow, 'list numerical', hide, columnspan
        )
        self.visible_components.add(component)
        return component

    def render_super_infection(self, hide, column, columnspan, frow):
        def radiobuttonselected(var, to_rerender, to_derender):
            no_validate_update(var, self.config_path, keys_path)

        text = "Enable Super-infection Events?"
        keys_path = ['EpidemiologyModel','super_infection']
        to_rerender, to_derender = None, None
        component = EasyRadioButton(
            keys_path, self.config_path, text, "Enable Super-infection", 
            self.scrollable_frame, column, frow, hide, 
            to_rerender, to_derender,
            columnspan, radiobuttonselected,
            labtext="By choosing \"YES\", you will allow exposed/infected to be infected again, provided that the capacity of within-host population size hasn't been reached."
            )
        self.visible_components.add(component)
        return component
