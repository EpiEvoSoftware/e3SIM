## $\textbf{e3SIM}$

$\text{e3SIM}$ (**E**pidemiological-**e**cological-**e**volutionary simulation framework for genetic epidemiology) is an innovative outbreak simulator that simultaneously simulates transmission dynamics and molecular evolution of pathogens within a host population contact network using an agent-based, discrete, and forward-in-time approach. This software caters to users of all programming backgrounds. It has an easy-to-use graphical interface for beginners and allows advanced customization through command-line options for experienced coders. It works on both MacOS system and Linux system.

## Useful Links
1. For an overview of our e3SIM, please refer to our manuscript: [e3SIM: Epidemiological-ecological-evolutionary simulation framework for genetic epidemiology](https://www.biorxiv.org/content/10.1101/2024.06.29.601123v1){:target="_blank"}.
2. For the codes and configuration files used in the manunscript, please refer to Zenodo at [doi:10.5281/zenodo.11715269](https://doi.org/10.5281/zenodo.11715269){:target="_blank"}.
3. For a detailed manual of e3SIM, please refer to: [Manual](https://github.com/EpiEvoSoftware/original_pipeline/blob/main/e3SIM_manual.pdf){:target="_blank"}.


## Installation

  1. Find a directory on your device for the software and clone the repository from the terminal.
      ```sh
      git clone https://github.com/EpiEvoSoftware/e3SIM
      ```
  
  2. Create a conda environment with the provided environment file. For MacOS users, replace `${ENV_YML}` with `e3SIM_mac.yml`. For Linux users, replace `${ENV_YML}` with `e3SIM_linux.yml`. This step took 2.5 minutes on a M2 pro macbook. 
      ```sh
      cd e3SIM
      conda env create --name e3SIM --file ${ENV_YML}
      ```
      <!-- If environment creation fails or you encounter errors about importing packages in testing (step 4), do `conda deactivate` to deactivate the environment and delete it by `conda remove --name e3SIM --all`, then repeat this step by using the no-builds options of the yml file (`mac_env_wo_builds.yml` for MacOS or `linux_env_wo_builds.yml` for Linux). -->
  
  3. Activate the conda environment.
      ```sh
      conda activate e3SIM
      ```
  
  4. Install R and R packages. Note that R has to be directly callable without the full path (test by running `Rscript --help`)  \
      Download and install R from here: https://cran.r-project.org/. After successful installation of R, run the following command one by one to install required R packages.\
     **For MacOS users:** 
        ```sh
        R
        install.packages("phylobase")
        install.packages("ape")
        install.packages("ggplot2")
        install.packages("R.utils")
        install.packages("data.table")

        if (!require("BiocManager", quietly = TRUE))
            install.packages("BiocManager")
        BiocManager::install("ggtree")
        BiocManager::install("Biostrings")

        q()
        ```
        
      **For Linux users:**
        ```sh
        R
        install.packages("ade4")

        q()
        ```

  6. Test whether $\text{e3SIM}$ is successfully installed by running a simple model. This testing took ~2 minutes on a M2 pro macbook. 
      ```sh
      cd e3SIM
      e3SIM=${PWD}
      cd ../test/test_installation
      python update_config.py # To update the test_config.json with user's directory
      python ${e3SIM}/outbreak_simulator.py -config test_config.json # To run the simulation
      ```
      Standard output in the terminal should show progress of the simulator. After the simulation ends, output files(e.g., `all_SEIR_trajectory.png`) are expected in the directory (test_installation) if the installation is successful.


### Usage

1. Find a working directory (in most circumstances, create a new empty directory outside the github repo you cloned). This directory will be your "working directory" for one simulation --- the generated input files and simulated results will be saved here. Save the path to variable `WKDIR` by replacing `${YOUR_WORKING_DIRECTORY}` with your actual working directory.
    ```sh
    WKDIR=${YOUR_WORKING_DIRECTORY}
    ```

2. Generate a configuration file and all pre-requisite files for one simulation.
    * GUI
    
        We provide an interactive graphical user interface (GUI) option for the pre-simulation data generation. To access the GUI, please run the following command under your `original_pipeline` directory.
        ```sh
        python gui
        ```
        A window will pop up and you will be asked to navigate to your working directory in the first tab. By going through all the tabs, a configuation file called `simulation_config.json` will be generated in the working directory according to the given inputs. Please refer to Chapter 7 in the manual for more details on the GUI application.

    * Command Line
    
        Command line tools for the pre-simulatuion programs includes NetworkGenerator, SeedGenerator, GeneticEffectGenerator, and SeedHostMatcher. Please refer to the manual chapter 2 for how to run them sequentially. After running these programs, you need to create a configuration file by modifying the config file template. For explanations on the configuration file, please refer to Manual chapter 3.2. The following commands copy the template to your designated working directory.
        ```sh
        cp ${e3SIM}/config_template/slim_only_template.json ${WKDIR}/simulation_config.json
        ```
        Then manually change the configuration in `${WKDIR}/simulation_config.json`.

3. Run the simulation
    ```sh
    python ${e3SIM}/outbreak_simulator.py -config ${WKDIR}/simulation_config.json
    ```

4. (Alternative to 2 & 3) Run the pre-simulation programs and the simulation together in one command. You need to fill out a bigger configuration file.
    ```sh
    cp ${e3SIM}/config_template/base_params.json ${WKDIR}/simulation.config
    ```
    Then manually change the configuration in `${WKDIR}/simulation.config` and run
    ```sh
    python ${e3SIM}/enivol.py -config ${WKDIR}/simulation.config
    ```

5. Working examples: There are two working example runs of the simulation that are described in the [Manual](https://github.com/EpiEvoSoftware/original_pipeline/blob/main/e3SIM_manual.pdf) Chapter 5 step-by-step. It is recommended to read Chapter 5 and try out the whole pipeline as instructed to understand the whole workflow since e3SIM contains a lot of information to be digest for a first-time-user.



## Liscence

Copyright &copy; 2024 Jaehee Kim. All rights reserved.\
$\text{e3SIM}$ is a free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

## Disclaimer
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
[GNU General Public License](\url{http://www.gnu.org/licenses/}) for more details.
<!-- ================================

Run ```conda env update --name myenvname --file environment.yml --prune`` to be updated w/ dependencies for backend
or
Run ```conda env create -f environment.yml``` to initialize conda env

https://akrabat.com/creating-virtual-environments-with-pyenv/
https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file


deployment: https://github.com/TomSchimansky/CustomTkinter/issues/2322


distributing for macos:
pip install pyinstaller
pyinstaller --onefile --windowed --icon=__icon__.ico __script__.py

https://www.pythonguis.com/tutorials/packaging-tkinter-applications-windows-pyinstaller/ -->

<!-- ## Additional from Manuscript
Templates for all necessary input files are available in our GitHub repository. It is recommended to run the relevant pre-simulation modules with the $\texttt{-method user_input}$ flag to enable $\text{e3SIM}$ to validate the format of the user-provided files.  -->
