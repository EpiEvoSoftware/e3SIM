## e3SIM Zenodo Package  
*Supporting materials for the manuscript “e3SIM: epidemiological-ecological-evolutionary simulation framework for genomic epidemiology”*

This Zenodo record contains the source code, demo scripts, and example data used in the manuscript. Each archive includes its own `README.md` with detailed, step-by-step instructions.

---

### Contents
- **e3SIM-main.zip**: 
Source code for the e3SIM framework. See `e3SIM-main/README.md` for installation, dependencies, and usage examples. 

- **manuscript_data.zip**: 
Parameter files, input data, and scripts for reproducing the simulation examples and runtime profiling reported in the manuscript. See `demo/README.md` for details. 

- **tutorials.zip**: 
Two vignettes demonstrating end-to-end use of e3SIM, and the relevant data to run these vignettes. The `data` folder contains the data for two pathogen reference genomes. The `vignette` folder contains the two pipelines. Within `vignettes` folder, each folder includes a `run.sh` bash script with all commands needed to execute the tutorial. Before executing `run.sh`, please inspect the script `run.sh` and manually change the absolute path as instructed in it.
    - `5.1_test_minimal_model`: Tutorial described in Chapter 5.1 of the manual 
    - `5.2_test_drugresist`: Tutorial described in Chapter 5.2 of the manual 

- **e3SIM_manual.pdf**:
Comprehensive manual for the e3SIM software.

- **coverage.xml**:
    - The coverage report generated using `pytest --cov=. --cov-report=xml` with the following session information:

        ```
        platform darwin -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0
        rootdir: e3SIM-main
        plugins: cov-7.0.0
        ```

---


### System requirements
- **Operating systems**: macOS or Linux
- **Prerequisites:**  
    - [Python ≥ 3.12](https://www.python.org/) with `python` and `pip` available on your `PATH`
    - [Conda](https://docs.conda.io/) (Miniconda or Anaconda)  
    - [R ≥ 4.0](https://cran.r-project.org/) with command-line access (`Rscript`)  
- **Build tools & libraries:**  
    Listed in each archive’s `README.md` and in the provided Conda environment file (`.yml`). 


### Installation 
Follow these steps to install and verify e3SIM on your system. These steps mirror the installation section in `e3SIM-main/README.md`.

1. **Extract the source archive** 
    
    ```sh
    unzip e3SIM-main.zip
    cd e3SIM-main
    ```
    This creates the `e3SIM-main/` directory.
    

2. **Create and activate the Conda environment** 

    - **macOS**
        
        Plan A:

        ```sh
        # 1) If you are on a M-chip machine (a.k.a., NOT Intel chip), run the following to install the emulator for x86_64 conda environment
        softwareupdate --install-rosetta --agree-to-license # Ignore the 'Installing Rosetta 2 on this system is not supported.' output if it occurs
        # Run the following to start a temporary Rosetta shell session inside your existing terminal
        arch -x86_64 zsh

        # 2) Create the environment
        CONDA_SUBDIR=osx-64 conda env create -n e3SIM -f e3SIM_mac.yml

        # 3) Save the subdirectory setting in the environment so future installs use osx-64
        conda activate e3SIM
        conda env config vars set CONDA_SUBDIR=osx-64
        conda deactivate && conda activate e3SIM

        # 4) Install phylobase and its dependencies
        conda install conda-forge::r-phylobase
        conda install conda-forge::r-reshape2
        ```
        Plan B:

        If your are unable to install as in plan A on your system, follow the steps below to create the environment:

        ```sh
        # 1) Create and activate the environment
        conda env create --name e3SIM --file e3SIM_mac.yml
        conda activate e3SIM

        # 2) Install the R packages separately
        Rscript -e 'install.packages("phylobase", repos="https://cloud.r-project.org",
         type = "source", INSTALL_opts = c("--no-test-load", "--no-staged-install", "--no-byte-compile"))
        ```

     - **Linux**
        ```sh
        conda env create --name e3SIM --file e3SIM_linux.yml
        conda activate e3SIM
        ```

3. **Verify installation** \
Run a small simulation to confirm everything is set up correctly under the `e3SIM` environment:

    ```sh
    cd e3SIM_codes
    e3SIM=${PWD}
    cd ../test_installation/run
    
    # Update test_config.json with user's directory
    python update_config.py 
    
    # Run the test simulation
    python ${e3SIM}/outbreak_simulator.py -config test_config.json 
    ```
        
    - You should see progress messages in the console.
    - Upon completion, check for the existence output files (e.g., `all_SEIR_trajectory.png`) in `e3SIM-main/test_installation/run/output_trajectories/`.


---

### General Usage
`${e3SIM}` should be set to the absolute path of the `e3SIM_codes` directory inside your e3SIM installation:

```sh
export e3SIM="/path/to/e3SIM-main/e3SIM_codes"
```


1. **Set up your working directory** \
    Create a new empty directory outside `e3SIM-main` directory. This directory will be your working directory for a single simulation; all generated input files and simulation results will be saved here. 

    Set the path to this directory in the `WKDIR` by replacing `/path/to/working_dir` with your actual path:

    ```sh
    WKDIR="/path/to/working_dir"
    ```

2. **Generate prerequisite files and a configuration file** \
    You can prepare simulation input files using either the command-line tools or the Graphical User Interface (GUI). For explanations of configuration parameters, see ***Chapter 3.2*** of the manual.

    - **Command Line** \
        Run the following pre-simulation programs *in order*:
        
        - `NetworkGenerator`
        - `SeedGenerator`
        - `GeneticEffectGenerator`
        - `SeedHostMatcher`
    
        These must be run sequentially to generate all prerequisite files required for the simulation. For detailed instructions, see ***Chapter 2*** of the manual.

        After generating these files, create a configuration file by copying and editing the provided template:
        
        ```sh
        cp ${e3SIM}/config_template/slim_only_template.json ${WKDIR}/config_file.json
        ```
  
        Then edit `${WKDIR}/config_file.json` to match your simulation settings.


    - **GUI** \
        An interactive GUI is available for pre-simulation data generation. The GUI can be launched from any local directory by specifying its full path: 
    
        ```sh
        python ${e3SIM}/gui
        ```
        
        The GUI will:
        - Prompt you to select your working directory on the first tab (default: current directory).
        - Guide you through each tab *in order* to generate prerequisite files.
        - Create a `config_file.json` in your working directory based on your inputs.

        For more details on the GUI, see ***Chapter 7*** of the manual. 
    

3. **Run the simulation** \
    Execute the simulation using:

    ```sh
    python ${e3SIM}/outbreak_simulator.py -config ${WKDIR}/config_file.json
    ```


---


## Liscence
e3SIM is a free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Disclaimer
This program is distributed WITHOUT ANY WARRANTY.  See the
[GNU General Public License](\url{http://www.gnu.org/licenses/}) for more details.