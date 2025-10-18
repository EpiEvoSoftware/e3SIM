"""
This module contains helpers and commonly-used widget functionality.
"""

from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import ttk
import json
from tkinter import filedialog

def load_config_as_dict(config_path):
    with open(config_path, "r") as file:
        return json.load(file)

def save_config(config_path, config):
    with open(config_path, "w") as file:
        json.dump(config, file, indent=2)

def get_nested_dict_val(d, keys):
    for key in keys:
        d = d[key]
    return d

def update_nested_dict(d, keys, value):
    """
    Update a nested dictionary of arbitrary depth recursively
    """
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        update_nested_dict(d[keys[0]], keys[1:], value)

def update_numerical_param(entry, keys_path, config_path, error_messages, param, is_int):
    try:
        if is_int:
            new_val = int(entry.get())
        else:
            new_val = float(entry.get())
        config = load_config_as_dict(config_path)
        update_nested_dict(config, keys_path, new_val)
        save_config(config_path, config)
    except ValueError:  # This catches cases where conversion to integer fails
        if is_int:
            valtype = "integer"
        else:
            valtype = "numerical"
        error_messages.append(f"{param}: Please enter a valid {valtype} value.")
    except Exception as e:  # General error handling (e.g., file operation failures)
        error_messages.append(f"{param} : Update Error, {e}")

def update_list_param(entry, keys_path, config_path, error_messages, render_text_short, _type):
    try:
        stripped_entry = entry.get().strip()
        if stripped_entry[0] != "[" or stripped_entry[-1] != "]":
            raise ValueError
        cleaned_input = stripped_entry[1:-1].strip()

        if cleaned_input == "":
            new_parsed = []
        else:
            strings = [item.strip() for item in cleaned_input.split(",")]
            match _type:
                case "integer":
                    new_parsed = list(map(int, strings))
                case "numerical":
                    new_parsed = list(map(float, strings))
                case _:
                    raise ValueError("Unrecognized _type")

        config = load_config_as_dict(config_path)
        update_nested_dict(config, keys_path, new_parsed)
        save_config(config_path, config)
    except ValueError:  # This catches cases where conversion to integer fails
        error_messages.append(
            f"{render_text_short}: Please enter a valid list of {_type}s, separated by commas.")
    except Exception as e:  # General error handling (e.g., file operation failures)
        error_messages.append(f"{render_text_short + ": Update Error, " + str(e)}")

def render_next_button(tab_index, tab_parent, parent, update_config=None):
    def next_tab():
        match update_config():
            case 1: # Update failed
                return
    
        try:
            function()
        except Exception as e:
            pass
            # print(f"Error occurred: {e}")  # Logging to the console
            # tk.messagebox.showerror("Error", f"An error occurred: {e}")
        current_tab_index = tab_index
        next_tab_index = (current_tab_index + 1) % tab_parent.index("end")
        tab_parent.tab(next_tab_index, state="normal")
        tab_parent.select(next_tab_index)

    next_button = ttk.Button(parent, text="Next", command=next_tab)
    next_button.pack()

def no_validate_update(var, config_path, keys_path, mapping=None):
    if mapping:
        var_get = mapping[var.get()]
    else:
        var_get = var.get()
    config = load_config_as_dict(config_path)
    update_nested_dict(config, keys_path, var_get)
    save_config(config_path, config)

def no_validate_update_val(val, config_path, keys_path):
    config = load_config_as_dict(config_path)
    update_nested_dict(config, keys_path, val)
    save_config(config_path, config)


class TabBase:
    '''
    Used as a parent class for tabs in the notebook.
    Note: visible_components at the moment simply collects the components
        that are checked for updates for the config when global_update is
        called (as it is when the 'next' button is pressed).
    '''
    def __init__(
        self,
        parent,
        tab_parent,
        config_path,
        tab_title,
        tab_index,
        hide=False,
        render_nb=True,
    ):
        self.visible_components = set()
        self.init_val(config_path)
        self.init_tab(parent, tab_parent, tab_title, tab_index, hide)
        self.load_page()
        if render_nb:
            render_next_button(
                self.tab_index, self.tab_parent, self.parent, update_config=self.global_update)

    def init_val(self, config_path):
        pass

    def load_page(self):
        pass

    def global_update(self):
        users_validation_messages = []

        for component in self.visible_components:
            component.update(users_validation_messages)

        match len(users_validation_messages):
            case 0:
                return 0
            case _:
                error_message_str = "\n\n".join(users_validation_messages)
                tk.messagebox.showerror("Update Error", error_message_str)
                return 1

    def init_tab(self, parent, tab_parent, tab_title, tab_index, hide):
        self.parent = parent
        self.tab_parent = tab_parent
        self.tab_index = tab_index
        self.tab_parent.add(parent, text=tab_title)
        if hide:
            self.tab_parent.tab(self.tab_index, state="disabled")
        self.control_frame = ttk.Frame(self.parent)
        self.control_frame.pack(fill="both", expand=True, padx=10, pady=10)


class EasyWidgetBase:
    '''
    Parent class to other "EasyWidget" classes. 
    '''
    def __init__(self):
        self.local_components = set()
        self.grid_layout = {}

    def rerender_itself(self):
        for component in self.local_components:
            grid_info = self.grid_layout.get(component, {})
            component.grid(**grid_info)

    def derender_itself(self):
        for component in self.local_components:
            if component not in self.grid_layout:
                self.grid_layout[component] = component.grid_info()
            component.grid_forget()

    def set_to_rerender(self, to_rerender):
        self.to_rerender = to_rerender

    def set_to_derender(self, to_derender):
        self.to_derender = to_derender

    def update(self, error_messages):
        pass


class GroupControls:
    '''
    GroupControls provides a way to create (potentially nested) groups of 
    EasyWidgets for simultaneous rerendering and derendering. 
    '''
    def __init__(self, lst=None):
        self.items = []
        if lst:
            if isinstance(lst, list):
                self.add_list(lst)
            else:
                raise ValueError("Attempted to initialize GroupControls with non-list")

    def add(self, item):
        if isinstance(item, EasyWidgetBase) or isinstance(item, GroupControls):
            self.items.append(item)
        else:
            raise ValueError("Item must be an instance of EasyWidgetBase or GroupControls")
    
    def add_list(self, lst):
        if isinstance(lst, list):
            for item in lst:
                self.add(item)
        else:
            raise ValueError("lst must be an instance of list")

    def rerender_itself(self):
        for item in self.items:
            item.rerender_itself()

    def derender_itself(self):
        for item in self.items:
            item.derender_itself()

    def to_rerender(self):
        for item in self.items:
            item.to_rerender()

    def to_derender(self):
        for item in self.items:
            item.to_derender()


class EasyTitle(EasyWidgetBase):
    def __init__(
        self, render_text, control_frame, column, frow, hide, columnspan=1, sticky="we", pady=5):
        super().__init__()
        self.render_text = render_text
        self.label = ttk.Label(control_frame, text=self.render_text, style="Title.TLabel")
        if frow is None or column is None:
            self.label.grid()
        else:
            self.label.grid(
                row=frow, column=column, columnspan=columnspan, sticky=sticky, pady=pady)

        self.local_components = {self.label}
        self.derender_itself()
        if not hide:
            self.rerender_itself()

class EasyLabel(EasyWidgetBase):
    def __init__(
        self, render_text, control_frame, column, frow, hide, columnspan=1, sticky="w", labtext=""):
        super().__init__()
        self.render_text = render_text
        self.label = ttk.Label(control_frame, text=self.render_text, style="Bold.TLabel")
        if labtext != "":
            CreateToolTip(self.label, labtext)
        if frow is None or column is None:
            self.label.grid()
        else:
            self.label.grid(
                row=frow, column=column, columnspan=columnspan, sticky=sticky, pady=5)

        self.local_components = {self.label}
        self.derender_itself()
        if not hide:
            self.rerender_itself()

class EasyButton(EasyWidgetBase):
    def __init__(
        self, render_text, control_frame, column, frow, command, hide, sticky="e"):
        super().__init__()
        self.render_text = render_text
        self.button = tk.Button(control_frame, text=self.render_text, command=command)
        if frow is None or column is None:
            self.button.grid()
        else:
            self.button.grid(row=frow, column=column, sticky=sticky, pady=5)

        self.local_components = {self.button}
        self.derender_itself()
        if not hide:
            self.rerender_itself()


class EasyEntry(EasyWidgetBase):
    """
    replaces render_numerical_input
    """

    def __init__(
        self, keys_path, config_path, render_text, render_text_short, control_frame, column, frow,
        validate_for, hide, columnspan, disabled=False, sticky='w', labtext=''):
        super().__init__()
        self.keys_path = keys_path
        self.config_path = config_path
        self.render_text_short = render_text_short
        self.validate_for = validate_for
        if keys_path:
            dict_var = get_nested_dict_val(load_config_as_dict(config_path), keys_path)
        else:
            dict_var = ""
        label = ttk.Label(control_frame, text=render_text, style="Bold.TLabel")
        self.label = label
        if labtext != "":
            CreateToolTip(self.label, labtext)
        self.entry = ttk.Entry(control_frame, foreground="black")
        self.entry.insert(0, str(dict_var))

        if frow is None or column is None:
            label.grid()
            self.entry.grid()
        else:
            label.grid(row=frow, column=column, columnspan=columnspan, sticky=sticky, pady=5)
            self.entry.grid(
                row=frow + 1, column=column, columnspan=columnspan, sticky=sticky, pady=5)

        self.local_components = {self.entry, label}
        self.derender_itself()
        if not hide:
            self.rerender_itself()

        if disabled:
            label.configure(state="disabled")
            self.entry.configure(foreground="light grey", state="disabled")
        else:
            label.configure(state="normal")
            self.entry.configure(foreground="black", state="normal")

    def update(self, error_messages):
        match self.validate_for:
            case "list integer":
                update_list_param(
                    self.entry, self.keys_path, self.config_path,
                    error_messages, self.render_text_short, "integer")
            case "list numerical":
                update_list_param(
                    self.entry,self.keys_path,self.config_path,
                    error_messages, self.render_text_short,"numerical")
            case "integer":
                update_numerical_param(
                    self.entry, self.keys_path, self.config_path,
                    error_messages, self.render_text_short, True)
            case "numerical":
                update_numerical_param(
                    self.entry, self.keys_path, self.config_path,
                    error_messages, self.render_text_short, False,)
            case "no update":
                pass
            case _:
                raise ValueError("Invalid internal type.")

class EasyRateMatrix(EasyWidgetBase):
    "Used for 4x4 mutation rate matrices"
    
    def __init__(
        self, keys_path, config_path, render_text, render_text_short, control_frame, column, frow,
        validate_for, hide, columnspan, disabled=False):
        super().__init__()
        self.keys_path = keys_path
        self.config_path = config_path
        self.render_text_short = render_text_short
        self.validate_for = validate_for
        if keys_path:
            initial_matrix = get_nested_dict_val(load_config_as_dict(config_path), keys_path)
        else:
            initial_matrix = [["" for i in range(4)] for j in range(4)]
        label = ttk.Label(control_frame, text=render_text, style="Bold.TLabel")
        self.label = label
        matrix_container = ttk.Frame(control_frame)
        matrix_entries = [[ttk.Entry(matrix_container, foreground="black", width=10) for j in range(4)] for i in  range(4)]
        self.matrix_entries = matrix_entries

        if frow is None or column is None:
            label.grid()
            matrix_container.grid()
        else:
            label.grid(row=frow, column=column, columnspan=columnspan, sticky="w", pady=5)
            matrix_container.grid(row=frow+1, column=column, columnspan=columnspan, rowspan=4, sticky="w", pady=5)
        for i in range(4):
            for j in range(4):
                entry = matrix_entries[i][j]
                entry.insert(0, initial_matrix[i][j])
                entry.grid(row = i, column = j, stick='w')
                if i == j:
                    entry.config(state='disabled', foreground='light grey')

        self.local_components = {label, matrix_container}
        self.derender_itself()
        if not hide:
            self.rerender_itself()

        if disabled:
            self.disable()

    def update(self, error_messages):
        match self.validate_for:
            case "numerical":
                try:
                    config = load_config_as_dict(self.config_path)
                    new_matrix_vals = [[0, 0, 0, 0] for i in range(4)]
                    for i in range(4):
                        for j in range(4):
                            new_val = float(self.matrix_entries[i][j].get())
                            new_matrix_vals[i][j] = new_val
                    update_nested_dict(config, self.keys_path, new_matrix_vals)
                    save_config(self.config_path, config)
                except ValueError:
                    error_messages.append(f"{self.render_text_short}: Please enter a valid numerical values for all matrix entries.")
                except Exception as e:  # General error handling (e.g., file operation failures)
                    error_messages.append(f"{self.render_text_short + ": Update Error, " + str(e)}")
            case "no update":
                pass
            case _:
                raise ValueError("Invalid internal type.")
    
    def disable(self, include_label=True):
        if include_label:
            self.label.configure(state="disabled")
        for i in range(4):
            for j in range(4):
                self.matrix_entries[i][j].config(state='disabled', foreground='light grey')

    def enable(self):
        self.label.configure(state="normal")
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.matrix_entries[i][j].config(state='normal', foreground='black')

class EasyRadioButton(EasyWidgetBase):
    '''
    Creates a radio button control with 2 radio buttons, labeled "Yes" and "No" by default
    but alternate labels can be provided via the text argument to initialization.
    '''
    def __init__(
        self, keys_path, config_path, render_text, render_text_short, control_frame,
        column, frow, hide, to_rerender, to_derender, columnspan, radiobuttonselected,
        disabled=False, text=None, labtext=""):
        super().__init__()
        self.keys_path = keys_path
        self.config_path = config_path
        self.render_text_short = render_text_short
        self.to_rerender = to_rerender
        self.to_derender = to_derender
        self.radiobuttonselected = radiobuttonselected

        if keys_path:
            dict_var = get_nested_dict_val(load_config_as_dict(config_path), keys_path)
            self.var = tk.BooleanVar(value=dict_var)
        else:
            self.var = tk.BooleanVar(value=True)
        label = ttk.Label(control_frame, text=render_text, style="Bold.TLabel")
        self.label = label
        if labtext != "":
            CreateToolTip(self.label, labtext)
        self.rb_true = ttk.Radiobutton(
            control_frame,
            text="Yes" if not text else text[0],
            variable=self.var,
            value=True,
            command=self._update,
        )
        self.rb_false = ttk.Radiobutton(
            control_frame,
            text="No" if not text else text[1],
            variable=self.var,
            value=False,
            command=self._update,
        )
        if frow is None or column is None:
            label.grid()
            self.rb_true.grid()
            self.rb_false.grid()
        else:
            label.grid(
                row=frow, column=column, columnspan=columnspan, sticky="w", pady=5)
            self.rb_true.grid(
                row=frow + 1, column=column, columnspan=columnspan, sticky="w", pady=5)
            self.rb_false.grid(
                row=frow + 2, column=column, columnspan=columnspan, sticky="w", pady=5)

        self.local_components = {label, self.rb_true, self.rb_false}
        self.derender_itself()
        if not hide:
            self.rerender_itself()

        if disabled:
            label.configure(state="disabled")
            self.rb_true.configure(state="disabled")
            self.rb_false.configure(state="disabled")
        else:
            label.configure(state="normal")
            self.rb_true.configure(state="normal")
            self.rb_false.configure(state="normal")

    def _update(self):
        self.radiobuttonselected(self.var, self.to_rerender, self.to_derender)

class EasyCombobox(EasyWidgetBase):
    def __init__(
        self, keys_path, config_path, render_text, control_frame,
        column, frow, combobox_values, to_rerender, to_derender, comboboxselected, hide, width,
        columnnspan, val_to_ui_mapping=None, keys_path_none=False, labtext=""):
        super().__init__()
        self.keys_path = keys_path
        self.config_path = config_path
        self.control_frame = control_frame
        self.to_derender = to_derender
        self.to_rerender = to_rerender
        self.val_to_ui_mapping = val_to_ui_mapping
        self.comboboxselected = comboboxselected

        self.label = ttk.Label(self.control_frame, text=render_text, style="Bold.TLabel")
        if not keys_path_none:
            dict_var = get_nested_dict_val(load_config_as_dict(config_path), keys_path)
            if val_to_ui_mapping is None:
                var_val = dict_var
            else:
                var_val = val_to_ui_mapping.get(dict_var, "")
        else:
            var_val = ""
        self.var = tk.StringVar(value=var_val)
        self.combobox = ttk.Combobox(
            self.control_frame,
            textvariable=self.var,
            values=combobox_values,
            state="readonly",
            width=width,
        )
        self.combobox.bind("<<ComboboxSelected>>", self._updater)

        if frow is None or column is None:
            self.label.grid()
            self.combobox.grid()
        else:
            self.label.grid(
                row=frow, column=column, columnspan=columnnspan, sticky="w", pady=5)
            self.combobox.grid(
                row=frow + 1, column=column, columnspan=columnnspan, sticky="w", pady=5)
        if labtext != "":
            CreateToolTip(self.label, labtext)

        self.local_components = {self.label, self.combobox}
        self.derender_itself()
        if not hide:
            self.rerender_itself()

    def _updater(self, event):
        self.comboboxselected(self.var, self.to_rerender, self.to_derender)
        # self.comboboxselected(self.var, self.to_rerender, self.to_derender)

class EasyPathSelector(EasyWidgetBase):
    def __init__(
        self, keys_path, config_path, render_text, control_frame,
        column, hide, frow, columnspan, filetype=None, labtext=""):
        super().__init__()
        self.filetype = filetype
        self.keys_path = keys_path
        self.config_path = config_path

        dict_var = get_nested_dict_val(load_config_as_dict(config_path), keys_path)
        label = ttk.Label(control_frame, text=render_text, style="Bold.TLabel")
        if labtext != "":
            CreateToolTip(label, labtext)
        if dict_var == "":
            self.value_label = ttk.Label(
                control_frame, text="None selected", foreground="black")
        else:
            self.value_label = ttk.Label(
                control_frame, text=dict_var, foreground="black")

        button = tk.Button(control_frame, text="Choose File", command=self._update)

        if frow is None or column is None:
            label.grid()
            self.value_label.grid()
            button.grid()
        else:
            label.grid(row=frow, column=column, columnspan=columnspan, sticky="w", pady=5)
            self.value_label.grid(
                row=frow + 1, column=column, columnspan=columnspan, sticky="w", pady=5)
            button.grid(row=frow + 2, column=column, columnspan=columnspan, pady=5)

        self.local_components = {label, self.value_label, button}
        self.derender_itself()
        if not hide:
            self.rerender_itself()

    def _update(self):
        if self.filetype is None:
            file_input = filedialog.askopenfilename(title="Select a File")
        else:
            file_input = filedialog.askopenfilename(title="Select a File", filetypes=self.filetype)
        if file_input:
            no_validate_update_val(file_input, self.config_path, self.keys_path)
            self.value_label.config(text=file_input)

class EasyImage(EasyWidgetBase):
    def __init__(
        self, image_path, desired_width, desired_height, hide,
        control_frame, frow, column, columnspan, rowspan):
        super().__init__()
        with Image.open(image_path) as img:
            img = img.resize((desired_width, desired_height))
            self.photo = ImageTk.PhotoImage(img)
            image_label = tk.Label(control_frame, image=self.photo)

            if frow is None or column is None:
                image_label.grid()
            else:
                image_label.grid(
                    column=column,
                    row=frow,
                    columnspan=columnspan,
                    rowspan=rowspan,
                    sticky="nsew",
                )

            self.local_components = {image_label}
            self.derender_itself()
            if not hide:
                self.rerender_itself()

class CreateToolTip:
    """
    Create a tooltip on hover for a given widget.
    """
    def __init__(self, widget, text="widget info", waittime=300, wraplength=240, x_offset=25, y_offset=25,):
        self.widget = widget
        self.text = text
        self.waittime = waittime
        self.wraplength = wraplength
        self.x_offset = x_offset
        self.y_offset = y_offset

        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tooltip_window = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hide_tip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show_tip_wrapper)

    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
        self.id = None

    def show_tip_wrapper(self, event=None):
        return self.show_tip(self.x_offset, self.y_offset, event)

    def show_tip(self, x_offset, y_offset, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + x_offset
        y += self.widget.winfo_rooty() + y_offset

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify="left",
            foreground="#000",
            background="#eee",
            relief="solid",
            borderwidth=1,
            wraplength=self.wraplength,
        )
        label.pack(ipadx=1)

    def hide_tip(self):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None


def read_txt(filename):
    try:
        with open(filename, "r") as file:
            print("success")
    except FileNotFoundError:
        print("The directory does not exist.")
    except PermissionError:
        print("You don't have permission to access this directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

def list_files(directory):
    """
    Prints out all the files in the specified directory.

    :param directory: The path to the directory to list files from.
    """
    try:
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                print(item)
    except FileNotFoundError:
        print("The directory does not exist.")
    except PermissionError:
        print("You don't have permission to access this directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

def choose_ref_path(self, title, config_path, var, filetypes=None):
    filetypes = (
        (
            "Genome files",
            ("*.fasta", "*.fa", "*.gb", "*.gtf", "*.vcf", "*.bam", "*.sam", "*.fna"),
        ),
        ("All files", "*.*"),
    )
    chosen_file = filedialog.askopenfilename(title="Select a Genome Reference File")
    if chosen_file:
        var = chosen_file
        config = load_config_as_dict(config_path)
        config["GenomeElement"]["ref_path"] = var
        save_config(config_path, config)
