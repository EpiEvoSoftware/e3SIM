"""
Refactored Simulation Framework
A clean, object-oriented design for epidemic simulation management.
"""

# from ast import str
import os
from pickletools import int4
import shutil
import subprocess
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
# from xxlimited import Str
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
# from epievosoftware.original_pipeline.post_simulation_func import plot_per_transmission_tree, plot_strain_distribution_trajectory, plot_SEIR_trajectory, run_per_data_processing
from post_simulation_func import *
from error_handling import CustomizedError
from base_func import *


DEFAULT_ALPHA = 0.405

# ========================= Enums and Constants =========================

class EpiModel(Enum):
    """Epidemiological model types."""
    SIR = "SIR"
    SEIR = "SEIR"


class SubstitutionModel(Enum):
    """Substitution model parameterization types."""
    MUT_RATE = "mut_rate"
    MUT_RATE_MATRIX = "mut_rate_matrix"


class LinkFunction(Enum):
    """Link functions for trait-probability relationships."""
    LOGIT = "logit"
    CLOGLOG = "cloglog"


# ========================= Data Classes =========================

@dataclass
class NetworkConfig:
    """Configuration for network model parameters."""
    host_size: int
    contact_network_path: Optional[Path] = None
    
    def __post_init__(self):
        print("Checking \"NetworkModelParameters\"...... ", flush = True)
        if not Path(self.contact_network_path).exists():
            raise CustomizedError("NetworkGenerator hasn't been run. Please run NetworkGenerator before running this program")
        
        ConfigValidator.validate_integer(self.host_size, "Host population size")
        print("\"NetworkModelParameters\" Checked. ", flush = True)


@dataclass
class EvolutionConfig:
    """Configuration for evolution model parameters."""
    n_generation: int
    subst_model_parameterization: SubstitutionModel
    transition_matrix_path: Path
    mut_rate: Optional[float] = None
    mut_rate_matrix: Optional[np.ndarray] = None
    within_host_reproduction: bool = False
    within_host_reproduction_rate: float = 0.0
    cap_withinhost: int = 1

    
    def __post_init__(self):
        print("Checking \"EvolutionModel\"...... ", flush = True)
        ConfigValidator.validate_integer(self.n_generation, "Number of generations", 1)
        ConfigValidator.validate_boolean(self.within_host_reproduction, "Within-host reproduction flag")
       
        ConfigValidator.validate_integer(self.cap_withinhost, "Within-host capacity", 1)
        if self.within_host_reproduction: 
            if self.cap_withinhost == 1:
                print("Warning: Within-host capacity is 1, within-host reproduction will have no effect.")
            
            ConfigValidator.validate_float(self.within_host_reproduction_rate, \
                                           "Within-host reproduction probability", 0.0, 1.0)
            if self.within_host_reproduction_rate == 0.0:
                print("Warning: Within-host reproduction rate is 0, within-host reproduction will have no effect.")

        if self.subst_model_parameterization == SubstitutionModel.MUT_RATE_MATRIX:
            if not ConfigValidator.validate_and_write_mutation_matrix(np.array(self.mut_rate_matrix), self.transition_matrix_path):
                raise CustomizedError(f"The given mutation rate matrix {self.mut_rate_matrix} \
                                 does NOT meet the requirement 1) zeros on diagonals \
                AND 2) non-negative numbers on non-diagonals")
        else:
            try:
                ConfigValidator.validate_integer(self.mut_rate, "Mutation rate", 1)
            except CustomizedError:
                ConfigValidator.validate_float(self.mut_rate, "Mutation rate", 0.0, 1.0)

        print("\"EvolutionModel\" checked.", flush = True)
    



@dataclass
class EpidemiologyConfig:
    """Configuration for epidemiology model parameters."""
    model: EpiModel
    n_epoch: int
    n_generation: int
    transmissibility_effsize: float | int
    drug_resistance_effsize: float | int
    S_IE_rate: float | int
    I_R_rate: float | int
    R_S_rate: float | int
    latency_prob: float | int
    E_I_rate: float | int
    I_E_rate: float | int
    E_R_rate: float | int
    surviv_prob: float | int
    sample_rate: float | int
    recovery_prob_after_sampling: float | int
    n_massive_sample: int
    massive_sample_generation: list[int]
    massive_sample_prob: list[float]
    massive_sample_recover_prob: list[float]
    cap_withinhost: int
    slim_replicate_seed_file_path: str
    traits_num: Dict
    epoch_changing_generation: List[int] = field(default_factory=list)
    super_infection: bool = False

    def __post_init__(self):
        """Validate epidemiology configuration."""
        print("Checking \"EpidemiologyModel\"...... ", flush = True)
        try:
            self.model = EpiModel[self.model]
        except KeyError:
            raise CustomizedError(f"Invalid model '{self.model}', must be one of {[m.name for m in EpiModel]}")
        
        ConfigValidator.validate_integer(self.n_epoch, "Number of epochs")
        if self.n_epoch > 1:
            if not (
                isinstance(self.epoch_changing_generation, list) 
                and len(self.epoch_changing_generation) == self.n_epoch - 1):
                raise CustomizedError(f'"epoch_changing_generation" must be a list of length n_epoch - 1')
            if any(not isinstance(i, int) or 
                not (1 <= i <= self.n_generation) for i in self.epoch_changing_generation):
                raise CustomizedError(
            f'"epoch_changing_generation" values must be integers in range 1..{self.n_generation}'
        )

        if self.model == EpiModel.SIR:
            for param in ("latency_prob", "E_I_rate", "I_E_rate", "E_R_rate"):
                setattr(self, param, [0] * self.n_epoch)

        prob_params = [
            "S_IE_rate", "I_R_rate", "R_S_rate",
            "latency_prob", "E_I_rate", "I_E_rate", "E_R_rate",
            "sample_rate", "recovery_prob_after_sampling", "surviv_prob"
        ]
        for param in prob_params:
            values = getattr(self, param)

            if not isinstance(values, list):
                raise CustomizedError(f"({param}) has to be a list []")

            if len(values) != self.n_epoch:
                raise CustomizedError(
                    f"{param} {values} must have the same length "
                    f"as number of epochs ({self.n_epoch})"
                )

            for v in values:
                ConfigValidator.validate_probability(v, param, 
                strict = param in {"surviv_prob", "S_IE_rate"})

        effsize_param = ["transmissibility_effsize", "drug_resistance_effsize"]
        for param in effsize_param:
            values = getattr(self, param)

            if not isinstance(values, list):
                raise CustomizedError(f"({param}) has to be a list []")
            
            if len(values) != self.n_epoch:
                raise CustomizedError(
                    f"{param} {values} must have the same length "
                    f"as number of epochs ({self.n_epoch})"
                )
            
            for v in values:
                ConfigValidator.validate_integer(v, param, min_val = 0)

        self._check_effect_size("transmissibility_effsize", self.traits_num.get("transmissibility"))
        self._check_effect_size("drug_resistance_effsize", self.traits_num.get("drug_resistance"))

        ConfigValidator.validate_integer(self.n_massive_sample, "The number of massive sampling events", min_val = 0)
        if self.n_massive_sample:
            values = self.massive_sample_generation

            if not isinstance(values, list) or len(values) != self.n_massive_sample:
                raise CustomizedError('Ticks for massive sampling ("massive_sample_generation") must be a list '
                f"of length {self.n_massive_sample}")

            for v in values: 
                ConfigValidator.validate_integer(v, "massive sampling generation")

            if any(i > self.n_generation for i in values):
                raise CustomizedError(
                    f'Ticks for massive sampling ("massive_sample_generation") must be valid '
                    f"(1..{self.n_generation})")
            
            for param in ("massive_sample_prob", "massive_sample_recover_prob"):
                values = getattr(self, param)

                if not isinstance(values, list) or len(values) != self.n_massive_sample:
                    raise CustomizedError(f'("{param}") must be a list of length {self.n_massive_sample}')
            
                for i in values: ConfigValidator.validate_probability(i, param) 
            
        ConfigValidator.validate_boolean(self.super_infection, "Whether to apply super infection")
        if self.super_infection and self.cap_withinhost == 1:
            print(
                "WARNING: Super-infection is activated, but within-host capacity is 1, "
                "so super-infection cannot actually happen.",
                flush=True,
            )
        
        print("\"EpidemiologyModel\" Checked. ", flush = True)


    def _check_effect_size(self, param: str, max_value: int):
        values = getattr(self, param)

        if any(i > max_value for i in values):
            raise CustomizedError(
            f"({param}) must be chosen from {list(range(max_value + 1))}")

@dataclass
class SeedInfo:
    """Configuration for seed parameters"""
    seed_size: int
    workding_dir: Path
    seed_host_matching_path: str
    use_reference: bool = False

    def __post_init__(self):
        print("Checking \"SeedsConfiguration\"...... ", flush = True)
        ConfigValidator.validate_integer(self.seed_size, "Number of seeding sequence")
        ConfigValidator.validate_boolean(self.use_reference, "Whether to use reference genome")

        if not (self.use_reference or Path(os.path.join(self.workding_dir, "originalvcfs")).exists()):
            raise CustomizedError("Reference genome isn't provided for seed sequences, but the alternative SeedGenerator hasn't been run."
            "Please run SeedGenerator before running this program or specify use_reference=true")
        
        if not Path(self.seed_host_matching_path).exists():
            raise CustomizedError("HostSeedMatcher hasn't been run. \
                Please run HostSeedMatcher before running this program")

        
        print("\"SeedsConfiguration\" Checked. ", flush = True)

@dataclass
class GenomeElement:
    """Configuration for genome elements"""
    ref_path: str
    use_genetic_model: bool
    workding_dir: str
    traits_num: dict[str, int]
    link: Any
    alpha_trans: list[int | float]
    alpha_drug: list[int | float]
    causal_gene_path: Path

    def __post_init__(self):
        print("Checking \"GenomeElement\"...... ", flush = True)
        if not Path(self.ref_path).exists():
            raise CustomizedError(f"Reference genome path {self.ref_path} doesn't exist.")
        
        ConfigValidator.validate_boolean(self.use_genetic_model, "Whether to apply genetic model")
        if self.use_genetic_model and not self.causal_gene_path.exists():
            raise CustomizedError("Genetic model for trait is used, but GeneticElementGenerator hasn't been run. "
                         "Please run GeneticElementGenerator before running this program")

        if not (
        isinstance(self.traits_num, dict)
        and len(self.traits_num) == 2
        and all(isinstance(v, int) for v in self.traits_num.values())
        ):
            raise CustomizedError(
                'Number of traits ("traits_num") must be a dict ({}) of length 2 '
                "with integer values for transmissibility and drug-resistance."
            )
        print("\"GenomeElement\" Checked. ", flush = True)

        for param in ("alpha_trans", "alpha_drug"):
            # value = getattr(self, param)
            # try:
            #     ConfigValidator.validate_integer(value, param)
            # except CustomizedError:
            #     ConfigValidator.validate_float(value, param, min_val = 0)
            #     if value < 0:
            #         raise CustomizedError(f"({param}) has to be positive")

            values = getattr(self, param)

            if not isinstance(values, list):
                raise CustomizedError(f"({param}) has to be a list []")

            if param =="alpha_trans":
                if len(values) == 0:
                    setattr(self, param, [DEFAULT_ALPHA] * self.traits_num["transmissibility"])
                    print(f"Warning: The link scale slope is not specified for transmissibility traits. "
                        f"Will use default values {DEFAULT_ALPHA}")
                elif len(values) != self.traits_num["transmissibility"]:
                    raise CustomizedError(
                        f"{param} {values} must have the same length "
                        f"as number of transmissibility traits ({self.traits_num["transmissibility"]})"
                )
            if param =="alpha_drug":
                if len(values) == 0:
                    setattr(self, param, [DEFAULT_ALPHA] * self.traits_num["drug_resistance"])
                    print(f"Warning: The link scale slope is not specified for drug-resistance traits. "
                        f"Will use default values {DEFAULT_ALPHA}")
                elif len(values) != self.traits_num["drug_resistance"]:
                    raise CustomizedError(
                        f"{param} {values} must have the same length "
                        f"as number of drug_resistance traits ({self.traits_num["drug_resistance"]})"
                )
            if len(values)>0:
                for v in values:
                    ConfigValidator.validate_float(v, param, min_val = 0)

            # for v in values:
            #     ConfigValidator.validate_probability(v, param, 
            #     strict = param in {"surviv_prob", "S_IE_rate"})

@dataclass
class Postprocessing:
    """Complete postprocessing configuration"""
    do_process: bool
    n_traits: dict[str, int]
    branch_color_trait: int
    heatmap_trait: str
    vcf: bool
    fasta: bool

    def __post_init__(self):
        print("Checking \"Postprocessing_options\"...... ", flush = True)
        ConfigValidator.validate_boolean(self.do_process, "Whether to postprocess results")
        if not self.do_process:
            print("Post-simulation data processing is not enabled.", flush = True)
            return
        
        ConfigValidator.validate_integer(self.branch_color_trait, "The trait for branch coloring", min_val = 0)
        if self.branch_color_trait > sum(self.n_traits.values()):
            raise CustomizedError(f"What trait to use to color the branches of the tree (branch_color_trait) should "
                        f"be an integer chosen from (0: color by seed, 1..{sum(self.n_traits.values())}: trait id")
        if self.branch_color_trait == 0:
            print("The tree will be colored by seed.", flush = True)
        else:
            print(f"The tree will be colored by its trait {self.branch_color_trait}", flush = True)
        
        if self.heatmap_trait not in ("none", "drug_resistance", "transmissibility"):
            raise CustomizedError(f"The trait for heatmap is not permitted. The possible choices are: none / drug_resistance / transmissibility")

        ConfigValidator.validate_boolean(self.vcf, "Whether to output VCF file")
        ConfigValidator.validate_boolean(self.fasta, "Whether to output FASTA file")
        print("\"Postprocessing_options\" Checked. ", flush = True)


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    working_dir: Path
    n_replicates: int
    network: NetworkConfig
    evolution: EvolutionConfig
    epidemiology: EpidemiologyConfig
    seed_info: SeedInfo
    genome_config: GenomeElement
    postprocess_config: Postprocessing
    slim_seed_file: Optional[Path] = None
    
    def __post_init__(self):
        print("Checking \"BasicRunConfiguration\"...... ", flush = True)
        self.working_dir = Path(self.working_dir)
        if not self.working_dir.exists():
            raise ValueError(f"Working directory {self.working_dir} doesn't exist")

        ConfigValidator.validate_integer(self.n_replicates, "Number of replicates", 1)
        print("\"BasicRunConfiguration\" checked.", flush = True)


# ========================= Validators =========================

class ConfigValidator:
    """Validates configuration parameters."""
    
    @staticmethod
    def validate_boolean(value: Any, name: str) -> None:
        """Validate integer parameter."""
        if not isinstance(value, bool):
            raise CustomizedError(f"{name} must be a boolean (true/false)")

    @staticmethod 
    def validate_integer(value: Any, name: str, min_val: int = 1) -> None:
        """Validate integer parameter."""
        if not isinstance(value, int) or value < min_val:
            raise CustomizedError(f"{name} must be an integer >= {min_val}")
    
    @staticmethod
    def validate_float(value: Any, name: str, min_val: float = None, max_val: float = None) -> None:
        """Validate float parameter."""
        if not isinstance(value, (int, float)):
            raise CustomizedError(f"{name} must be a number")
        
        if min_val is not None and value < min_val:
            raise CustomizedError(f"{name} must be >= {min_val}")
            
        if max_val is not None and value > max_val:
            raise CustomizedError(f"{name} must be <= {max_val}")
    
    @staticmethod
    def validate_probability(value: Any, name: str, strict: bool = False) -> None:
        """Validate probability parameter."""
        ConfigValidator.validate_float(value, name, 0.0, 1.0)
        
        if strict and (value == 0.0 or value == 1.0):
            raise CustomizedError(f"{name} must be strictly between 0 and 1")
    
    @staticmethod
    def validate_and_write_mutation_matrix(matrix: np.ndarray, path_to_write) -> bool:
        """Validate mutation rate matrix."""
        # Check diagonal is zero
        if not np.allclose(np.diag(matrix), 0):
            return False
        
        # Check non-negative values
        if not (matrix >= 0).all():
            return False
        
        col_names = ["A", "C", "G", "T"]
        df = pd.DataFrame(matrix, columns = col_names)
        df.to_csv(path_to_write)
            
        return True


# ========================= Configuration Parser =========================

class ConfigParser:
    """Parses and validates configuration files."""
    
    # def __init__(self, validator: ConfigValidator = None):
    #     self.validator = validator or ConfigValidator()
    def __init__(self):
    #     self.validator = validator or ConfigValidator()
        pass

    
    def parse_config_file(self, config_path: Union[str, Path]) -> SimulationConfig:
        """Parse configuration from JSON/YAML file."""
        config_path = Path(config_path)
        print(config_path)
        
        if not config_path.exists():
            raise CustomizedError(f"Config file {config_path} not found")
        
        config_dict = read_params(config_path, "default_config.json")
        
        return self.parse_config_dict(config_dict)
    
    def parse_config_dict(self, config: Dict[str, Any]) -> SimulationConfig:
        """Parse configuration from dictionary."""
        # Parse basic run configuration
        basic_config = config.get("BasicRunConfiguration", {})

        # Parse network configuration
        network_config = self._parse_network_config(config.get("NetworkModelParameters", {}), Path(basic_config.get("cwdir", ".")))
        
        # Parse evolution configuration
        evolution_config = self._parse_evolution_config(config.get("EvolutionModel", {}), Path(basic_config.get("cwdir", ".")))

        # Parse genome element configuration
        genome_element = self._parse_genome_element_config(config.get("GenomeElement", {}), Path(basic_config.get("cwdir", ".")))
        
        # Parse epidemiology configuration
        epidemiology_config = self._parse_epidemiology_config(config.get("EpidemiologyModel", {}), 
        evolution_config.n_generation, evolution_config.cap_withinhost, genome_element.traits_num)

        # Parse seeding information configuration
        seed_info = self._parse_seed_info(config.get("SeedsConfiguration", {}), basic_config.get("cwdir", "."))
        
        # Parse postprocessing
        postprocessing = self._parse_postprocess_config(config.get("Postprocessing_options", {}), genome_element.traits_num)
        
        return SimulationConfig(
            working_dir = basic_config.get("cwdir", "."),
            n_replicates = basic_config.get("n_replicates", 1),
            network = network_config,
            evolution = evolution_config,
            epidemiology = epidemiology_config,
            seed_info = seed_info,
            genome_config = genome_element,
            postprocess_config = postprocessing,
            slim_seed_file = Path(epidemiology_config.get("slim_replicate_seed_file_path", ""))
            if epidemiology_config.slim_replicate_seed_file_path else None
        )
    
    def _parse_network_config(self, config: Dict, cwdir) -> NetworkConfig:
        """Parse network configuration."""
        return NetworkConfig(
            host_size = config.get("host_size", 0),
            contact_network_path = os.path.join(cwdir, "contact_network.adjlist")
        )
    
    def _parse_evolution_config(self, config: Dict, cwdir: Path) -> EvolutionConfig:
        """Parse evolution configuration."""
        subst_model = config.get("subst_model_parameterization", "mut_rate")
        
        return EvolutionConfig(
            n_generation = config.get("n_generation", 100),
            subst_model_parameterization = SubstitutionModel(subst_model).value,
            mut_rate = config.get("mut_rate"),
            mut_rate_matrix = np.array(config.get("mut_rate_matrix")) if config.get("mut_rate_matrix") else None,
            within_host_reproduction = config.get("within_host_reproduction", False),
            within_host_reproduction_rate = config.get("within_host_reproduction_rate", 0.0),
            cap_withinhost = config.get("cap_withinhost", 1),
            transition_matrix_path = cwdir / "muts_transition_matrix.csv"
        )
    
    def _parse_epidemiology_config(self, config: Dict, n_generation: int, cap_withinhost: int, traits_num: Dict) -> EpidemiologyConfig:
        """Parse epidemiology configuration."""
        return EpidemiologyConfig(
            model = EpiModel(config.get("model", "SIR")).value,
            n_epoch = config.get("epoch_changing", {}).get("n_epoch", 1),
            epoch_changing_generation = config.get("epoch_changing", {}).get("epoch_changing_generation", []),
            n_generation = n_generation,
            super_infection = config.get("super_infection", False),
            transmissibility_effsize = config.get("genetic_architecture").get("transmissibility"),
            drug_resistance_effsize = config.get("genetic_architecture").get("drug_resistance"),
            S_IE_rate = config.get("transition_prob").get("S_IE_prob"),
            I_R_rate = config.get("transition_prob").get("I_R_prob"),
            R_S_rate = config.get("transition_prob").get("R_S_prob"),
            latency_prob = config.get("transition_prob").get("latency_prob"),
            E_I_rate = config.get("transition_prob").get("E_I_prob"),
            I_E_rate = config.get("transition_prob").get("I_E_prob"),
            E_R_rate = config.get("transition_prob").get("E_R_prob"),
            surviv_prob = config.get("transition_prob").get("surviv_prob"),
            sample_rate = config.get("transition_prob").get("sample_prob"),
            recovery_prob_after_sampling = config.get("transition_prob").get("recovery_prob_after_sampling"),
            n_massive_sample = config.get("massive_sampling").get("event_num"),
            massive_sample_generation = config.get("massive_sampling").get("generation"),
            massive_sample_prob = config.get("massive_sampling").get("sampling_prob"),
            massive_sample_recover_prob = config.get("massive_sampling").get("recovery_prob_after_sampling"),
            slim_replicate_seed_file_path = config.get("slim_replicate_seed_file_path"),
            cap_withinhost = cap_withinhost,
            traits_num = traits_num
        )

    def _parse_genome_element_config(self, config: Dict, cwdir: Path) -> GenomeElement:
        """Parse genome element configuration."""
        link = config.get("trait_prob_link").get("link", "")
        return GenomeElement(
            ref_path = config.get("ref_path", ""),
            use_genetic_model = config.get("use_genetic_model", True),
            traits_num = config.get("traits_num", {}),
            link = link,
            alpha_trans = config.get("trait_prob_link").get(link).get("alpha_trans"),
            alpha_drug = config.get("trait_prob_link").get(link).get("alpha_drug"),
            workding_dir = cwdir,
            causal_gene_path = cwdir / "causal_gene_info.csv"
        )

    def _parse_seed_info(self, config: Dict, cwdir) -> SeedInfo:
        """Parse seed information configuration"""
        return SeedInfo(
            seed_size = config.get("seed_size", 1),
            use_reference = config.get("use_reference", ""),
            workding_dir = cwdir,
            seed_host_matching_path = os.path.join(cwdir, "seed_host_match.csv")
        )
    
    def _parse_postprocess_config(self, config: Dict, traits_num: Dict) -> Postprocessing:
        """Parse post processing configuration"""
        return Postprocessing( # didn't decide whether to plot in the first place
            do_process = config.get("do_postprocess"),
            n_traits = traits_num,
            branch_color_trait = config.get("tree_plotting").get("branch_color_trait"),
            heatmap_trait = config.get("tree_plotting").get("heatmap"),
            vcf = config.get("sequence_output").get("vcf"),
            fasta = config.get("sequence_output").get("fasta"),
        )



# ========================= SLiM Script Generator =========================

class SlimScriptGenerator:
    """Generates SLiM simulation scripts."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.script_components = []
        self.code_path = Path(__file__).parent / "slim_scripts"
    
    def generate_script(self, output_path: Path) -> Path:
        """Generate complete SLiM script."""

        print("******************************************************************** \n" + 
        "                       CREATING SLIM SCRIPT						   \n" + 
        "********************************************************************", flush = True)

        output_path = Path(output_path)
        
        # Clear existing script
        if output_path.exists():
            output_path.unlink()
        
        # Build script components
        self._add_initialization()
        self._add_mutation_blocks()
        self._add_control_blocks()
        self._add_transmission_blocks()
        self._add_state_transitions()
        self._add_logging()
        self._add_finalization()
        
        # Write script
        self._write_script(output_path)
        
        print(f"Generated SLiM script: {output_path}", flush = True)
        return output_path
    
    def _add_initialization(self):
        """Add initialization blocks."""
        self.script_components.extend([
            "trait_calc_function.slim",
            "initialization_pt1.slim",
            "read_config.slim",
            "initialization_pt2.slim"
        ])
        
        if self.config.genome_config.use_genetic_model:
            self.script_components.append("genomic_element_init_effsize.slim")
        else:
            self.script_components.append("genomic_element_init.slim")
        
        self.script_components.append("initialization_pt3.slim")
    
    def _add_mutation_blocks(self):
        """Add mutation-related blocks."""
        if self.config.genome_config.use_genetic_model:
            self.script_components.append("mutation_effsize.slim")
    
    def _add_control_blocks(self):
        """Add control and setup blocks."""
        self.script_components.append("block_control.slim")
        
        # Seeds and network
        if self.config.seed_info.use_reference:
            self.script_components.append("seeds_read_in_noburnin.slim")
        else:
            self.script_components.append("seeds_read_in_network.slim")
        
        self.script_components.append("contact_network_read_in.slim")
        
        # Epoch changing
        if self.config.epidemiology.n_epoch > 1:
            self.script_components.append("change_epoch.slim")
    
    def _add_transmission_blocks(self):
        """Add transmission-related blocks."""
        self.script_components.append("self_reproduce.slim")
        
        # Transmission based on genetic model
        if not self.config.genome_config.use_genetic_model:
            self.script_components.append("transmission_nogenetic.slim")
        else:
            # Add appropriate transmission model based on configuration
            if any (effsize > 0 for effsize in self.config.epidemiology.transmissibility_effsize):
                link_func = self.config.genome_config.link
                self.script_components.append(f"transmission_additive_{link_func}.slim")
            if any (effsize == 0 for effsize in self.config.epidemiology.transmissibility_effsize):
                self.script_components.append("transmission_nogenetic.slim")
        
        # Within-host reproduction
        if self.config.evolution.within_host_reproduction:
            self.script_components.append("within_host_reproduce.slim")
    
    def _add_state_transitions(self):
        """Add state transition blocks."""
        self.script_components.extend([
            "kill_old_pathogens.slim",
            "store_current_states.slim"
        ])
        
        # SEIR-specific blocks
        if self.config.epidemiology.model.value == "SEIR":
            self.script_components.append("Exposed_process.slim")
        
        # Infection process
        if not self.config.genome_config.use_genetic_model:
            self.script_components.append("Infected_process_nogenetic.slim")
        else: 
            if any (effsize > 0 for effsize in self.config.epidemiology.drug_resistance_effsize):
                link_func = self.config.genome_config.link
                self.script_components.append(f"Infected_process_additive_{link_func}.slim")
            if any (effsize == 0 for effsize in self.config.epidemiology.transmissibility_effsize):
                self.script_components.append("Infected_process_nogenetic.slim")

        # Massive sampling
        if self.config.epidemiology.n_massive_sample > 0:
            self.script_components.append("massive_sampling.slim")
        
        # New infections
        if self.config.epidemiology.super_infection:
            self.script_components.append("New_infection_process_superinfection.slim")
        else:
            self.script_components.append("New_infection_process.slim")
        
        # Recovery
        if any(rate != 0 for rate in self.config.epidemiology.R_S_rate):
            self.script_components.append("Recovered_process.slim")
    
    def _add_logging(self):
        """Add logging blocks."""
        self.script_components.append("log.slim")
    
    def _add_finalization(self):
        """Add finalization blocks."""
        self.script_components.append("finish_simulation.slim")
    
    def _write_script(self, output_path: Path):
        """Write the complete script to file."""
        with open(output_path, 'w') as outfile:
            for component in self.script_components:
                component_path = self.code_path / component
                if component_path.exists():
                    with open(component_path, 'r') as infile:
                        outfile.write(infile.read())
                else:
                    print(f"Script component not found: {component_path}", flush = True)


# ========================= Simulation Runner =========================

class SimulationRunner:
    """Manages and runs simulations."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.successful_runs = []
    
    def run_all_simulations(self) -> List[int]:
        """Run all simulation replicates."""
        print("******************************************************************** \n" + 
          "                     RUNNING THE SIMULATION						    \n" + 
          "********************************************************************", flush = True)
        
        for run_id in range(1, self.config.n_replicates + 1):
            print(f"Running replicate {run_id}/{self.config.n_replicates}...")
            
            if self._run_single_simulation(run_id):
                self.successful_runs.append(run_id)
                # print(f"Processing replication {run_id} treesequence file...", flush = True)
            else:
                print(f"WARNING: Replicate {run_id} failed or produced no output", flush = True)
        
        print(f"Completed {len(self.successful_runs)}/{self.config.n_replicates} simulations successfully", flush = True)
        return self.successful_runs
    
    def _run_single_simulation(self, run_id: int) -> bool:
        """Run a single simulation replicate."""
        output_dir = self.config.working_dir / str(run_id)
        
        # Clean and create output directory
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        
        # Get random seed if specified
        seed = self._get_seed_for_run(run_id)
        
        # Build command
        cmd = self._build_slim_command(run_id, seed)
        
        # Run simulation
        stdout_path = output_dir / "slim.stdout"
        with open(stdout_path, 'w') as stdout_file:
            result = subprocess.run(cmd, stdout=stdout_file, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print(f"SLiM failed for run {run_id}: {result.stderr.decode()}", flush = True)
            return False
   
        # Check for output
        sample_path = output_dir / "sample.csv.gz"

        return sample_path.exists()
    
    def _get_seed_for_run(self, run_id: int) -> Optional[int]:
        """Get random seed for specific run."""
        if self.config.slim_seed_file and self.config.slim_seed_file.exists():
            seeds_df = pd.read_csv(self.config.slim_seed_file)
            if run_id <= len(seeds_df):
                return int(seeds_df.loc[run_id - 1, "random_number_seed"])
        return None
    
    def _build_slim_command(self, run_id: int, seed: Optional[int] = None) -> List[str]:
        """Build SLiM command."""
        config_path = self.config.working_dir / "slim.params"
        script_path = self.config.working_dir / "simulation.slim"
        
        cmd = [
            "slim",
            "-d", f'config_path="{config_path}"',
            "-d", f"runid={run_id}",
        ]
        
        if seed is not None:
            cmd.extend(["-seed", str(seed)])
        
        cmd.append(str(script_path))
        
        return cmd


# ========================= Post-Processing =========================

class PostProcessor:
    """Handles post-simulation data processing."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.output_dir = config.working_dir / "output_trajectories"
    
    def process_all_results(self, successful_runs: List[int]):
        """Process results from all successful runs."""
        if not self.config.postprocess_config.do_process:
            print("Post-processing disabled", flush = True)
            return
        
        print("Starting post-processing...", flush = True)
        
        # Create output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir()
        

        # Process each successful run
        for run_id in successful_runs:
            self._process_single_run(run_id)
        
        # Generate aggregate plots
        self._generate_aggregate_plots(successful_runs)
        
        print("Post-processing completed", flush = True)
    
    def _process_single_run(self, run_id: int):
        """Process results from a single run."""
        run_dir = self.config.working_dir / str(run_id)
        
        if (run_dir / "sample.csv.gz").exists():
            # Process tree sequences
            self._process_sequences(run_dir, run_id)
            # Generate plots
            self._generate_run_plots(run_dir, run_id)
        else:
            print(f"There's no sampled genome in replicate {run_id}. \
                Either the simulation failed or the sampling rate is too low. \
            Please check your config file and confirm those are your desired \
                simulation parameters.", flush = True)
                

    
    def _process_sequences(self, run_dir: Path, run_id: int):
        """Process sequence data."""
        print(f"Processing replication {run_id} treesequence file...", flush = True)
        run_per_data_processing(
            self.config.genome_config.ref_path,
            self.config.working_dir,
            self.config.genome_config.use_genetic_model,
            run_id,
            self.config.postprocess_config.n_traits,
            self.config.seed_info.seed_host_matching_path,
            {"vcf": self.config.postprocess_config.vcf, "fasta": self.config.postprocess_config.fasta},
            self.config.postprocess_config.branch_color_trait,
        )
    
    def _generate_run_plots(self, run_dir: Path, run_id: int):
        """Generate plots for a single run."""
        print(f"Plotting transmission tree for replication {run_id}...", flush = True)
        seed_phylo = "" if self.config.seed_info.use_reference else \
            os.path.join(self.config.working_dir, "seeds.nwk")
        plot_per_transmission_tree(
            run_dir, 
            self.config.seed_info.seed_size,
            self.config.working_dir / "slim.params",
            self.config.postprocess_config.n_traits,
            seed_phylo,
            self.config.postprocess_config.heatmap_trait
        )
        print(f"Plotting strain distribution trajectory for replication {run_id}...", flush = True)
        plot_strain_distribution_trajectory(
            run_dir,
            self.config.seed_info.seed_size,
            self.config.evolution.n_generation,
        )
        # if os.path.exists(os.path.join(each_wkdir, "SEIR_trajectory.csv.gz")):
        if Path(run_dir / "SEIR_trajectory.csv.gz").exists():
            print(f"Plotting SEIR trajectory for replication {run_id}...", flush = True)
            plot_SEIR_trajectory(
                run_dir, 
                self.config.seed_info.seed_size, 
                self.config.network.host_size, 
                self.config.evolution.n_generation
            )
        
    
    def _generate_aggregate_plots(self, successful_runs: List[int]):
        """Generate aggregate plots across all runs."""
        # Implementation depends on specific plotting functions
        print(f"Plotting the aggregated SEIR trajectory...", flush = True)
        plot_all_SEIR_trajectory(self.config.working_dir, 
                                self.config.seed_info.seed_size, 
                                self.config.network.host_size, 
                                self.config.evolution.n_generation, 
                                successful_runs)
        print(f"Plotting the aggregated strain distribution trajectory...", flush = True)
        plot_all_strain_trajectory(self.config.working_dir, 
                                self.config.seed_info.seed_size, 
                                self.config.network.host_size, 
                                self.config.evolution.n_generation, 
                                successful_runs)


# ========================= Main Orchestrator =========================

class SimulationOrchestrator:
    """Main orchestrator for the simulation pipeline."""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.parser = ConfigParser()
        self.config = None
        self.script_generator = None
        self.runner = None
        self.post_processor = None
    
    def initialize(self):
        """Initialize all components."""
        print(f"Initializing simulation framework...", flush = True)
        
        # Parse configuration
        self.config = self.parser.parse_config_file(self.config_path)
        
        # Initialize components
        self.script_generator = SlimScriptGenerator(self.config)
        self.runner = SimulationRunner(self.config)
        self.post_processor = PostProcessor(self.config)
        
        # Generate parameter file
        self._generate_parameter_file()
        
    
    def run(self):
        """Run the complete simulation pipeline."""
        try:
            # Initialize
            self.initialize()
            
            # Generate SLiM script
            script_path = self.config.working_dir / "simulation.slim"
            self.script_generator.generate_script(script_path)
            
            # Run simulations
            successful_runs = self.runner.run_all_simulations()
            
            # Post-process results
            self.post_processor.process_all_results(successful_runs)
            # self.post_processor.process_all_results([1])
            
            print("Simulation pipeline completed successfully", flush = True)

            print("******************************************************************** \n" + 
          "                FINISHED. THANK YOU FOR USING.					    \n" + 
          "********************************************************************", flush = True)

            return None
            
        except CustomizedError as e:
            print(f"Simulation pipeline failed: {e}")
            return e
    
    def _generate_parameter_file(self):
        """Generate SLiM parameter file."""
        param_file = self.config.working_dir / "slim.params"
        
        with open(param_file, 'w') as f:
            # Write all necessary parameters
            f.write(f"cwdir:{self.config.working_dir}\n")
            f.write(f"n_replicates:{self.config.n_replicates}\n")
            f.write(f"n_generation:{self.config.evolution.n_generation}\n")
            # f.write(f"transition_matrix:{self.config.evolution.mut_rate_matrix}\n")
            f.write(f"transition_matrix:{self._writebinary(self.config.evolution.subst_model_parameterization == SubstitutionModel.MUT_RATE_MATRIX)}\n")
            f.write(f"mut_rate:{self.config.evolution.mut_rate}\n")
            f.write(f"within_host_reproduction:{self._writebinary(self.config.evolution.within_host_reproduction)}\n")
            f.write(f"within_host_reproduction_rate:{self.config.evolution.within_host_reproduction_rate}\n")
            f.write(f"cap_withinhost:{self.config.evolution.cap_withinhost}\n")
            f.write(f"seed_size:{self.config.seed_info.seed_size}\n")
            f.write(f"use_reference:{self._writebinary(self.config.seed_info.use_reference)}\n")
            f.write(f"seed_host_matching_path:{self.config.seed_info.seed_host_matching_path}\n")
            f.write(f"ref_path:{self.config.genome_config.ref_path}\n")
            f.write(f"causal_gene_path:{self.config.genome_config.causal_gene_path}\n")
            f.write(f"use_genetic_model:{self._writebinary(self.config.genome_config.use_genetic_model)}\n")
            f.write(f"alpha_trans:{self._print_list_no_space(self.config.genome_config.alpha_trans)}\n")
            f.write(f"alpha_drug:{self._print_list_no_space(self.config.genome_config.alpha_drug)}\n")
            f.write(f"contact_network_path:{self.config.network.contact_network_path}\n")
            f.write(f"host_size:{self.config.network.host_size}\n")
            f.write(f"epi_model:{self.config.epidemiology.model.value}\n")
            f.write(f"n_epoch:{self.config.epidemiology.n_epoch}\n")
            f.write(f"super_infection:{self._writebinary(self.config.epidemiology.super_infection)}\n")
            f.write(f"n_massive_sample:{self.config.epidemiology.n_massive_sample}\n")
            if self.config.epidemiology.epoch_changing_generation:
                f.write(f"epoch_changing_generation:{self._print_list_no_space(self.config.epidemiology.epoch_changing_generation)}\n")
            if self.config.epidemiology.S_IE_rate:
                f.write(f"S_IE_rate:{self._print_list_no_space(self.config.epidemiology.S_IE_rate)}\n")
            if self.config.epidemiology.I_R_rate:
                f.write(f"I_R_rate:{self._print_list_no_space(self.config.epidemiology.I_R_rate)}\n")
            if self.config.epidemiology.R_S_rate:
                f.write(f"R_S_rate:{self._print_list_no_space(self.config.epidemiology.R_S_rate)}\n")
            if self.config.epidemiology.latency_prob:
                f.write(f"latency_prob:{self._print_list_no_space(self.config.epidemiology.latency_prob)}\n")
            if self.config.epidemiology.E_I_rate:
                f.write(f"E_I_rate:{self._print_list_no_space(self.config.epidemiology.E_I_rate)}\n")
            if self.config.epidemiology.I_E_rate:
                f.write(f"I_E_rate:{self._print_list_no_space(self.config.epidemiology.I_E_rate)}\n")
            if self.config.epidemiology.E_R_rate:
                f.write(f"E_R_rate:{self._print_list_no_space(self.config.epidemiology.E_R_rate)}\n")
            if self.config.epidemiology.surviv_prob:
                f.write(f"surviv_prob:{self._print_list_no_space(self.config.epidemiology.surviv_prob)}\n")
            if self.config.epidemiology.sample_rate:
                f.write(f"sample_rate:{self._print_list_no_space(self.config.epidemiology.sample_rate)}\n")
            if self.config.epidemiology.recovery_prob_after_sampling:
                f.write(f"recovery_prob_after_sampling:{self._print_list_no_space(self.config.epidemiology.recovery_prob_after_sampling)}\n")
            if self.config.epidemiology.transmissibility_effsize:
                f.write(f"transmissibility_effsize:{self._print_list_no_space(self.config.epidemiology.transmissibility_effsize)}\n")
            if self.config.epidemiology.drug_resistance_effsize:
                f.write(f"drugresistance_effsize:{self._print_list_no_space(self.config.epidemiology.drug_resistance_effsize)}\n")
            if self.config.epidemiology.massive_sample_generation:
                f.write(f"massive_sample_generation:{self._print_list_no_space(self.config.epidemiology.massive_sample_generation)}\n")
            if self.config.epidemiology.massive_sample_prob:
                f.write(f"massive_sample_prob:{self._print_list_no_space(self.config.epidemiology.massive_sample_prob)}\n")
            if self.config.epidemiology.massive_sample_recover_prob:
                f.write(f"massive_sample_recover_prob:{self._print_list_no_space(self.config.epidemiology.massive_sample_recover_prob)}\n")
            # Add more parameters as needed
        print(f"Generated parameter file: {param_file}", flush = True)


    def _writebinary(self, v):
        """
        Convert str/int/float value to binary string representation.

        Parameters:
            v (str): String value to be converted.

        Returns:
            str: Binary string representation ('T' for True, '' for False).
        """
        return "T" if v else ""

    def _print_list_no_space(self, lst):
        return ",".join(str(x) for x in lst)

# ======================== Config-based Interface ================= 
def all_slim_simulation_by_config(all_config):
    try:
        orchestrator = SimulationOrchestrator(all_config)
    except CustomizedError as e:
        return e

    error = orchestrator.run()

    return error


# ========================= CLI Interface =========================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run epidemic simulations with SLiM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-config',
        required=True,
        help='Path to the configuration file'
    )
    
    args = parser.parse_args()
    
    # Run simulation
    orchestrator = SimulationOrchestrator(args.config)
    orchestrator.run()


if __name__ == "__main__":
    main()