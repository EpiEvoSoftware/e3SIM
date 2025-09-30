## $\textbf{e3SIM}$

$\text{e3SIM}$ (**E**pidemiological-**e**cological-**e**volutionary simulation framework for genetic epidemiology) is an outbreak simulator that simultaneously simulates transmission dynamics and molecular evolution of pathogens within a host population contact network using an agent-based, discrete, and forward-in-time approach. This software caters to users of all programming backgrounds. It has an easy-to-use graphical interface for beginners and allows advanced customization through command-line options for experienced coders. It works on both MacOS system and Linux system.
The test coverage is 0.4808 according to Codecov. Since $\text{e3SIM}$ is composed of many SLiM code chunks which is difficult to test one-by-one, our unit tests are mainly focused on the pre-simulation modules. The SLiM simulation codes are tested manually which are not logged in the Codecov calculation.

## Installation (Linux / macOS)

1. **Extract the source archive** \
    Download and unzip the source archive:
    
    ```sh
    unzip e3SIM-main.zip
    cd e3SIM-main
    ```
This creates the e3SIM-main/ directory.
    

2. **Create the conda environment** \
Create a conda environment with the provided environment file.

    - **macOS**
        ```sh
        conda env create --name e3SIM --file e3SIM_mac.yml
        ```
    
    - **Linux**
        ```sh
        conda env create --name e3SIM --file e3SIM_linux.yml
        ```

  
3. **Activate the environment**

    ```sh
    conda activate e3SIM
    ```
  
4. **Install required R packages** \
Make sure `Rscript` is in your `PATH` (test with `Rscript --help`).  

    - **macOS**
        ```sh
        R
        
        chooseCRANmirror(graphics = FALSE)
        install.packages(c("phylobase", "ape", "ggplot2", "R.utils", "data.table"))

        if (!requireNamespace("BiocManager", quietly = TRUE))
            install.packages("BiocManager")
        BiocManager::install(c("ggtree", "Biostrings"))

        q()
        ```

    - **Linux**
        ```sh
        R
        
        chooseCRANmirror(graphics = FALSE)
        install.packages("ade4")

        q()
        ```

5. **Verify Installation** \
Run a small simulation to confirm everything is set up correctly:

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
    - Upon completion, check for output files (e.g., `all_SEIR_trajectory.png`) in `e3SIM-main/test_installation/run/output_trajectories/`.



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
$\text{e3SIM}$ is a free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

## Disclaimer
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
[GNU General Public License](\url{http://www.gnu.org/licenses/}) for more details.
