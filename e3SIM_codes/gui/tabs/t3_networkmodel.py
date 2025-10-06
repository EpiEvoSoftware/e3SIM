import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from utils import load_config_as_dict, save_config, CreateToolTip
import json
import networkx as nx
import os
from network_generator import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class NetworkModel:
    def __init__(self, parent, tab_parent, config_path, tab_title, tab_index, hide=False,):
        
        self.tab_parent = tab_parent
        self.tab_parent.add(parent, text=tab_title)
        if hide:
            self.tab_parent.tab(tab_index, state="disabled")
        self.config_path = config_path
        self.initial_config = load_config_as_dict(config_path)
        self.tab_title = tab_title
        self.tab_index = tab_index

        self.right_frame = ttk.Frame(parent, width=200)
        self.left_frame = ttk.Frame(parent)
        self.right_frame.pack(side="right", fill="y", expand=False)
        self.left_frame.pack(side="left", fill="both", expand=True)

        # NetworkModelConfigurations
        self.control_frame = ttk.Frame(self.right_frame, width=300)
        self.control_frame.pack(padx=10, pady=(0, 10))
        
        self.derender_group_network_method = set()
        self.derender_group_network_model = set()
        self.update_config_entries = {}

        self.render_initial()
        self.render_next_button()

        #NetworkModelGraph
        self.create_graph_frame()

    def derender(self, widget_set):
        for widget in widget_set:
            widget.pack_forget()
        widget_set.clear()

    def render_initial(self):
        self.render_host_size()
        self.render_method()

    def render_host_size(self):
        host_size_label = tk.ttk.Label(self.control_frame, text="Host Size", style="Bold.TLabel")
        host_size_entry = tk.ttk.Entry(self.control_frame, foreground="black")
        host_size_entry.insert(
            0, str(load_config_as_dict(self.config_path)["NetworkModelParameters"]["host_size"]))
        self.update_config_entries["host_size"] = host_size_entry

        CreateToolTip(host_size_label, 
                      "Number of hosts in the population.")
        host_size_label.pack()
        host_size_entry.pack(pady=(0, 10))

    def render_method(self):
        def update():
            config = load_config_as_dict(self.config_path)
            config["NetworkModelParameters"]["method"] = network_method_var.get()
            save_config(self.config_path, config)

            self.derender(self.derender_group_network_method)

            if network_method_var.get() == "user_input":
                self.render_select_file()
            elif network_method_var.get() == "randomly_generate":
                self.render_network_model()
            else:
                raise ValueError()

        network_method_var = tk.StringVar(
            value=load_config_as_dict(self.config_path)["NetworkModelParameters"]["method"])
        network_method_label = tk.ttk.Label(
            self.control_frame, text="Network Model Method", style="Bold.TLabel")
        network_method_rb_user = tk.ttk.Radiobutton(
            self.control_frame,
            text="User Input",
            variable=network_method_var,
            value="user_input",
            command=update)
        network_method_rb_random = tk.ttk.Radiobutton(
            self.control_frame,
            text="Random Generation",
            variable=network_method_var,
            value="randomly_generate",
            command=update)

        CreateToolTip(network_method_label, 
                      "Which random network model to use to generate a host contact network. You can import your own contact network by selecting \"user_input\" or randomly generate one from our provided models by selecting \"randomly_generate\".")
        network_method_label.pack()
        network_method_rb_user.pack()
        network_method_rb_random.pack(pady=(0, 10))

        if network_method_var.get() == "user_input":
            self.render_select_file()
        elif network_method_var.get() == "randomly_generate":
            self.render_network_model()

    def render_select_file(self):
        def choose_file():
            chosen_file = filedialog.askopenfilename(title="Select a File")

            if chosen_file:
                # Validate file
                if not self.update_config():
                    return

                config = load_config_as_dict(self.config_path)
                wk_dir = config["BasicRunConfiguration"]["cwdir"]
                pop_size = config["NetworkModelParameters"]["host_size"]

                try:
                    manager = NetworkManager(wk_dir)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to initialize NetworkManager:\n{e}")
                    return
                network, error = manager.run(
                    pop_size = pop_size,
                    wk_dir = wk_dir,
                    method = "user_input",
                    path_network = chosen_file
                )
                
                if error:
                    messagebox.showerror("Error", error)
                    return

                # Reflect in GUI
                config["NetworkModelParameters"]["user_input"]["path_network"] = chosen_file
                save_config(self.config_path, config)
                file_value_label.config(text=chosen_file)

                degrees = [network.degree(n) for n in network.nodes()]
                self.plot_degree_distribution(degrees)

                
                
        config = load_config_as_dict(self.config_path)
        file = config["NetworkModelParameters"]["user_input"]["path_network"]
        file_label = tk.ttk.Label(
            self.control_frame, text="Custom Network File", style="Bold.TLabel")

        if file == "":
            file_value_label = tk.ttk.Label(
                self.control_frame, text="No file selected", foreground="black")
        else:
            file_value_label = tk.ttk.Label(self.control_frame, text=file, foreground="black")

        file_button = tk.Button(self.control_frame, text="Choose File", command=choose_file)

        file_label.pack()
        file_value_label.pack()
        file_button.pack(pady=(0, 10))

        self.derender_group_network_method.update([file_label, file_value_label, file_button])

    def render_network_model(self):
        def update(event):
            config = load_config_as_dict(self.config_path)
            config["NetworkModelParameters"]["randomly_generate"]["network_model"] = \
                network_model_var.get()
            save_config(self.config_path, config)

            self.derender(self.derender_group_network_model)
            match network_model_var.get():
                case "ER":
                    self.render_ER()
                case "RP":
                    self.render_RP()
                case "BA":
                    self.render_BA()

        config = load_config_as_dict(self.config_path)
        model = config["NetworkModelParameters"]["randomly_generate"]["network_model"]
        network_model_var = tk.StringVar(value=model)
        network_model_label = tk.ttk.Label(self.control_frame, text="Network Model", style="Bold.TLabel")
        CreateToolTip(network_model_label, 
                      "Options: Erdos-Renyi (ER), Barabasi-Albert (BA), Random partition (RP)")
        network_model_combobox = ttk.Combobox(
            self.control_frame,
            textvariable=network_model_var,
            values=["ER", "RP", "BA"],
            state="readonly",
            width=20)
        network_model_combobox.bind("<<ComboboxSelected>>", update)

        network_model_label.pack()
        network_model_combobox.pack(pady=(0, 10))

        self.derender_group_network_method.update([network_model_label, network_model_combobox])

        match network_model_var.get():
            case "ER":
                self.render_ER()
            case "RP":
                self.render_RP()
            case "BA":
                self.render_BA()
    
    def render_ER(self):
        config = load_config_as_dict(self.config_path)
        p_ER = config["NetworkModelParameters"]["randomly_generate"]["ER"]["p_ER"]
        p_ER_label = tk.ttk.Label(
            self.control_frame, text="ER probability parameter", style="Bold.TLabel")
        p_ER_entry = tk.ttk.Entry(self.control_frame, foreground="black")
        p_ER_entry.insert(0, str(p_ER))
        self.update_config_entries["p_ER"] = p_ER_entry

        p_ER_label.pack()
        p_ER_entry.pack()

        self.derender_group_network_method.update([p_ER_label, p_ER_entry])
        self.derender_group_network_model.update([p_ER_label, p_ER_entry])

        self.render_run_network_generation()

    def render_RP(self):
        config = load_config_as_dict(self.config_path)
        rp_size = config["NetworkModelParameters"]["randomly_generate"]["RP"]["rp_size"]
        rp_size_label = tk.ttk.Label(
            self.control_frame, text="RP partition sizes", style="Bold.TLabel")
        rp_size_entry = tk.ttk.Entry(self.control_frame, foreground="black")
        rp_size_entry.insert(0, str(rp_size))
        self.update_config_entries["rp_size"] = rp_size_entry

        p_within = config["NetworkModelParameters"]["randomly_generate"]["RP"]["p_within"]
        p_within_label = tk.ttk.Label(
            self.control_frame, text="RP within-probabilities", style="Bold.TLabel")
        p_within_entry = tk.ttk.Entry(self.control_frame, foreground="black")
        p_within_entry.insert(0, str(p_within))
        self.update_config_entries["p_within"] = p_within_entry

        p_between = config["NetworkModelParameters"]["randomly_generate"]["RP"]["p_between"]
        p_between_label = tk.ttk.Label(
            self.control_frame, text="RP between-probability", style="Bold.TLabel")
        p_between_entry = tk.ttk.Entry(self.control_frame, foreground="black")
        p_between_entry.insert(0, str(p_between))
        self.update_config_entries["p_between"] = p_between_entry

        rp_size_label.pack()
        rp_size_entry.pack()
        p_within_label.pack()
        p_within_entry.pack()
        p_between_label.pack()
        p_between_entry.pack()

        self.derender_group_network_method.update(
            [rp_size_label, rp_size_entry, p_within_label,
             p_within_entry, p_between_label, p_between_entry])
        self.derender_group_network_model.update(
            [rp_size_label, rp_size_entry, p_within_label,
             p_within_entry, p_between_label, p_between_entry])

        self.render_run_network_generation()

    def render_BA(self):
        config = load_config_as_dict(self.config_path)
        ba_m = config["NetworkModelParameters"]["randomly_generate"]["BA"]["ba_m"]
        ba_m_label = tk.ttk.Label(self.control_frame, text="BA m parameter", style="Bold.TLabel")
        ba_m_entry = tk.ttk.Entry(self.control_frame, foreground="black")
        ba_m_entry.insert(0, str(ba_m))
        self.update_config_entries["ba_m"] = ba_m_entry

        ba_m_label.pack()
        ba_m_entry.pack()

        self.derender_group_network_method.update([ba_m_label, ba_m_entry])
        self.derender_group_network_model.update([ba_m_label, ba_m_entry])

        self.render_run_network_generation()

    def render_run_network_generation(self):
        def run_network_generate():
            if not self.update_config():
                return

            try:
                config = load_config_as_dict(self.config_path)
                wk_dir = config["BasicRunConfiguration"]["cwdir"]
                rand_seed = config["BasicRunConfiguration"]["random_number_seed"]
                pop_size = config["NetworkModelParameters"]["host_size"]
                network_model = \
                    config["NetworkModelParameters"]["randomly_generate"]["network_model"]

                if network_model == "ER":
                    p_ER = config["NetworkModelParameters"]["randomly_generate"]["ER"]["p_ER"]
                    try:
                        manager = NetworkManager(wk_dir)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to initialize NetworkManager:\n{e}")
                        return
                    _ , error = manager.run(
                        pop_size = pop_size,
                        wk_dir = wk_dir,
                        method = "randomly_generate",
                        model = "ER",
                        p_ER = p_ER,
                        rand_seed = rand_seed)
                elif network_model == "BA":
                    m = config["NetworkModelParameters"]["randomly_generate"]["BA"]["ba_m"]
                    try:
                        manager = NetworkManager(wk_dir)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to initialize NetworkManager:\n{e}")
                        return
                    _ , error = manager.run(
                        pop_size = pop_size,
                        wk_dir = wk_dir,
                        method = "randomly_generate",
                        model = "BA",
                        m = m,
                        rand_seed = rand_seed)
                elif network_model == "RP":
                    rp_size = config["NetworkModelParameters"]["randomly_generate"]["RP"]["rp_size"]
                    p_within = \
                        config["NetworkModelParameters"]["randomly_generate"]["RP"]["p_within"]
                    p_between = \
                        config["NetworkModelParameters"]["randomly_generate"]["RP"]["p_between"]
                    try:
                        manager = NetworkManager(wk_dir)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to initialize NetworkManager:\n{e}")
                        return
                    _ , error = manager.run(
                        pop_size = pop_size,
                        wk_dir = wk_dir,
                        method = "randomly_generate",
                        model = "RP",
                        rp_size = rp_size,
                        p_within = p_within,
                        p_between = p_between,
                        rand_seed = rand_seed)
                else:
                    raise ValueError("Unsupported model.")

                if error is not None:
                    messagebox.showerror("Network Generation Error", str(error))
                else:
                    messagebox.showinfo("Success", "Network Generated successfully!")

                G = nx.read_adjlist(os.path.join(wk_dir, "contact_network.adjlist"))
                degrees = [G.degree(n) for n in G.nodes()]
                self.plot_degree_distribution(degrees)

            except Exception as e:
                messagebox.showerror("Network Generation Error", str(e))

        run_network_generate_button = tk.Button(
            self.control_frame,
            text="Run Network Generation",
            command=run_network_generate,
        )
        run_network_generate_button.pack(pady=10)
        self.derender_group_network_method.add(run_network_generate_button)
        self.derender_group_network_model.add(run_network_generate_button)

    def render_next_button(self):
        def next_tab():
            if not self.update_config():
                return
            next_tab_index = (self.tab_index + 1) % self.tab_parent.index("end")
            self.tab_parent.tab(next_tab_index, state="normal")
            self.tab_parent.select(next_tab_index)
        next_button = tk.ttk.Button(self.control_frame, text="Next", command=next_tab)
        next_button.pack(side="bottom", pady=10)

    def update_config(self):
        config = load_config_as_dict(self.config_path)
        network_model_params = config["NetworkModelParameters"]
        try:
            host_size = int(self.update_config_entries["host_size"].get())
            network_model_params["host_size"] = host_size
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid int for Host Size.")
            return False
        if host_size == 0:
            messagebox.showerror("Warning", "Warning: Your host size is currently 0.")
        
        match network_model_params["randomly_generate"]["network_model"]:
            case "ER":
                try:
                    p_ER = float(self.update_config_entries["p_ER"].get())
                    assert(0 < p_ER <= 1)
                    network_model_params["randomly_generate"]["ER"]["p_ER"] = p_ER
                except (AssertionError, ValueError):
                    messagebox.showerror(
                        "Error", 
                        "Please enter a valid float for ER probability parameter. (0 < p_ER <= 1)")
                    return False
                
            case "RP":
                try:
                    rp_size = json.loads(self.update_config_entries["rp_size"].get())
                    rp_size[0] = int(rp_size[0])
                    rp_size[1] = int(rp_size[1])
                    assert(len(rp_size) == 2)
                    network_model_params["randomly_generate"]["RP"]["rp_size"] = rp_size
                except (AssertionError, ValueError):
                    messagebox.showerror("Error", "Please enter a valid list of two ints for partition sizes.")
                    return False
                
                try:
                    p_within = json.loads(self.update_config_entries["p_within"].get())
                    p_within[0] = float(p_within[0])
                    p_within[1] = float(p_within[1])
                    assert(len(rp_size) == 2)
                    network_model_params["randomly_generate"]["RP"]["p_within"] = p_within
                except (AssertionError, ValueError):
                    messagebox.showerror(
                        "Error", 
                        "Please enter a valid list of two floats for within-probabilities.")
                    return False
                
                try:
                    p_between = float(self.update_config_entries["p_between"].get())
                    network_model_params["randomly_generate"]["RP"]["p_between"] = p_between
                except ValueError:
                    messagebox.showerror(
                        "Error", "Please enter a valid float for between-probability.")
                    return False
                
            case "BA":
                try:
                    ba_m = int(self.update_config_entries["ba_m"].get())
                    network_model_params["randomly_generate"]["BA"]["ba_m"] = ba_m
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid integer for BA m parameter.")
                    return False

        save_config(self.config_path, config)
        return True
    

    ###################################
    ### NetworkModelGraph
    ###################################

    def create_graph_frame(self):
        self.graph_frame = ttk.Frame(self.left_frame)
        self.graph_frame.pack(fill="both", expand=True)
        self.plot_degree_distribution([])

    def plot_degree_distribution(self, degrees):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        if degrees:
            bin_size = max(1, int((max(degrees)-min(degrees))/30))
            ax.hist(
                degrees,
                bins=np.arange(min(degrees)-bin_size-0.5, max(degrees)+bin_size+0.5, bin_size),
                edgecolor="black")
        else:
            ax.hist([], bins=[])
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 1)
        ax.set_title("Degree Distribution")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Number of Nodes")

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
