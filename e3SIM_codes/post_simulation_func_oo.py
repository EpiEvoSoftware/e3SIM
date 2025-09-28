"""
Refactored Post-Processing Framework
Object-oriented design for simulation data analysis and visualization.
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import tskit
import pyslim
from Bio import SeqIO
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ========================= Data Classes =========================

@dataclass
class TreeSequenceData:
    """Container for tree sequence data."""
    tree_sequence: tskit.TreeSequence
    sampled_tree: tskit.TreeSequence
    sample_size: range
    n_generations: int
    working_dir: Path
    
    @classmethod
    def from_file(cls, working_dir: Union[str, Path]) -> 'TreeSequenceData':
        """Load tree sequence data from file."""
        working_dir = Path(working_dir)
        ts_path = working_dir / "sampled_genomes.trees"
        
        if not ts_path.exists():
            raise FileNotFoundError(f"Tree sequence file not found: {ts_path}")
        
        ts = tskit.load(str(ts_path))
        n_gens = ts.metadata["SLiM"]["tick"]
        sample_size = range(ts.tables.individuals.num_rows)
        sampled_tree = ts.simplify(samples=[2 * i for i in sample_size])
        
        return cls(
            tree_sequence=ts,
            sampled_tree=sampled_tree,
            sample_size=sample_size,
            n_generations=n_gens,
            working_dir=working_dir
        )


@dataclass
class TraitConfiguration:
    """Configuration for trait analysis."""
    transmissibility_traits: int = 0
    drug_resistance_traits: int = 0
    color_trait_index: int = 0
    use_sigmoid: bool = False
    
    @property
    def total_traits(self) -> int:
        """Total number of traits."""
        return self.transmissibility_traits + self.drug_resistance_traits
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format."""
        return {
            "transmissibility": self.transmissibility_traits,
            "drug_resistance": self.drug_resistance_traits
        }


@dataclass
class NodeMetadata:
    """Metadata for a single node."""
    node_id: int
    name: str
    node_time: int
    subpopulation_id: int
    parent_id: int
    color: str
    trait_values: List[float] = field(default_factory=list)
    
    def to_csv_row(self) -> List[str]:
        """Convert to CSV row format."""
        return [
            str(self.node_id),
            self.name,
            str(self.node_time),
            str(self.subpopulation_id),
            str(self.parent_id),
            self.color
        ] + [str(v) for v in self.trait_values]


# ========================= Tree Sequence Analyzer =========================

class TreeSequenceAnalyzer:
    """Analyzes tree sequence data."""
    
    def __init__(self, ts_data: TreeSequenceData):
        self.ts_data = ts_data
        self.tree = ts_data.sampled_tree.first()
        self.traversal_order = list(self.tree.nodes(order="preorder"))
    
    def get_node_labels(self) -> Dict[int, str]:
        """Generate labels for nodes (generation.host_id format)."""
        labels = {}
        tables = self.ts_data.sampled_tree.tables
        
        leaf_times = self.ts_data.n_generations - tables.nodes.time[self.ts_data.sample_size].astype(int)
        
        for leaf_id in self.ts_data.sample_size:
            subpop = tables.individuals[leaf_id].metadata["subpopulation"]
            labels[leaf_id] = f"{leaf_times[leaf_id]}.{subpop}"
        
        return labels
    
    def get_roots_by_seed(self, seed_host_match_path: Path) -> Dict[int, int]:
        """Get mapping of roots to seed IDs."""
        df_match = pd.read_csv(seed_host_match_path)
        match_dict = df_match.set_index('host_id')['seed'].to_dict()
        
        roots_map = {}
        for root in self.tree.roots:
            root_subpop = self.ts_data.sampled_tree.tables.individuals[root].metadata["subpopulation"]
            roots_map[root] = match_dict.get(root_subpop, -1)
        
        return roots_map
    
    def extract_mutations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mutation positions and affected nodes."""
        positions = []
        node_ids = []
        
        for mut_idx in range(self.ts_data.sampled_tree.tables.mutations.num_rows):
            mut = self.ts_data.sampled_tree.mutation(mut_idx)
            positions.append(self.ts_data.sampled_tree.site(mut.site).position + 1)
            node_ids.append(mut.node)
        
        return np.array(positions), np.array(node_ids)


# ========================= Trait Calculator =========================

class TraitCalculator:
    """Calculates trait values for nodes."""
    
    def __init__(self, config: TraitConfiguration):
        self.config = config
        self.effect_sizes = None
        self.causal_sites = None
    
    def load_causal_genes(self, causal_gene_path: Path) -> None:
        """Load causal gene information."""
        if not causal_gene_path.exists():
            raise FileNotFoundError(f"Causal gene file not found: {causal_gene_path}")
        
        self.effect_sizes = pd.read_csv(causal_gene_path)
        self.causal_sites = np.array(self.effect_sizes["Sites"], dtype=float).flatten()
    
    def calculate_traits(self, ts_analyzer: TreeSequenceAnalyzer) -> List[Dict[int, float]]:
        """Calculate trait values for all nodes."""
        if self.config.total_traits == 0:
            return []
        
        if self.effect_sizes is None:
            raise ValueError("Causal genes not loaded. Call load_causal_genes first.")
        
        # Extract mutations
        positions, node_ids = ts_analyzer.extract_mutations()
        
        # Find mutations at causal sites
        matches = np.isin(positions, self.causal_sites)
        
        if not np.any(matches):
            logger.warning("No mutations found at causal sites")
            return self._create_zero_traits(ts_analyzer)
        
        matched_positions = positions[matches]
        matched_indices = np.where(matches)[0]
        site_indices = np.searchsorted(self.causal_sites, matched_positions)
        
        # Calculate traits
        trait_values = []
        num_nodes = ts_analyzer.ts_data.sampled_tree.tables.nodes.num_rows
        
        for trait_idx in range(self.config.total_traits):
            # Calculate effect contributions
            node_effects = np.zeros(num_nodes)
            
            for i, mut_idx in enumerate(matched_indices):
                effect_size = self.effect_sizes.iloc[site_indices[i], trait_idx + 1]
                node_effects[node_ids[mut_idx]] += effect_size
            
            # Propagate effects through tree
            trait_dict = self._propagate_effects(node_effects, ts_analyzer)
            
            # Apply sigmoid if configured
            if self.config.use_sigmoid:
                trait_dict = self._apply_sigmoid(trait_dict)
            
            trait_values.append(trait_dict)
        
        return trait_values
    
    def _create_zero_traits(self, ts_analyzer: TreeSequenceAnalyzer) -> List[Dict[int, float]]:
        """Create zero-valued traits for all nodes."""
        num_nodes = ts_analyzer.ts_data.sampled_tree.tables.nodes.num_rows
        zero_dict = {i: 0.0 for i in range(num_nodes)}
        return [zero_dict.copy() for _ in range(self.config.total_traits)]
    
    def _propagate_effects(self, node_effects: np.ndarray, ts_analyzer: TreeSequenceAnalyzer) -> Dict[int, float]:
        """Propagate trait effects through tree."""
        trait_values = {-1: 0.0}  # Virtual root
        
        for node_id in ts_analyzer.traversal_order:
            parent_value = trait_values[ts_analyzer.tree.parent(node_id)]
            trait_values[node_id] = parent_value + node_effects[node_id]
        
        del trait_values[-1]  # Remove virtual root
        return trait_values
    
    def _apply_sigmoid(self, trait_dict: Dict[int, float]) -> Dict[int, float]:
        """Apply sigmoid transformation to trait values."""
        return {k: 1 / (1 + np.exp(-v)) for k, v in trait_dict.items()}


# ========================= Color Mapper =========================

class ColorMapper:
    """Maps nodes to colors based on various criteria."""
    
    def __init__(self, colormap: str = 'viridis'):
        self.colormap = colormap
        self.cmap = cm.get_cmap(colormap)
    
    def color_by_trait(self, trait_values: Dict[int, float]) -> Dict[int, str]:
        """Color nodes based on normalized trait values."""
        values = np.array(list(trait_values.values()))
        
        if values.max() > values.min():
            normalized = (values - values.min()) / (values.max() - values.min())
            colors = [self.cmap(v) for v in normalized]
            hex_colors = [mcolors.to_hex(c) for c in colors]
            return {i: hex_colors[i] for i in range(len(trait_values))}
        else:
            return {i: "#000000" for i in range(len(trait_values))}
    
    def color_by_seed(self, ts_analyzer: TreeSequenceAnalyzer, seed_host_match_path: Path) -> Dict[int, str]:
        """Color nodes based on seed origin."""
        # Get seed mapping
        roots_to_seeds = ts_analyzer.get_roots_by_seed(seed_host_match_path)
        unique_seeds = len(set(roots_to_seeds.values()))
        
        # Create color palette
        colors = self.cmap(np.linspace(0, 1, unique_seeds))
        hex_colors = [mcolors.to_hex(c) for c in colors]
        
        # Assign colors
        node_colors = {}
        num_nodes = ts_analyzer.ts_data.sampled_tree.tables.nodes.num_rows
        
        # Initialize with root colors
        for root, seed_id in roots_to_seeds.items():
            if seed_id >= 0:
                node_colors[root] = hex_colors[seed_id]
        
        # Propagate colors through tree
        for node in ts_analyzer.traversal_order:
            if node not in node_colors:
                parent = ts_analyzer.tree.parent(node)
                if parent in node_colors:
                    node_colors[node] = node_colors[parent]
                else:
                    node_colors[node] = "#000000"
        
        return node_colors


# ========================= Output Generators =========================

class NewickGenerator:
    """Generates Newick format tree files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "transmission_tree"
        self._prepare_output_dir()
    
    def _prepare_output_dir(self):
        """Prepare output directory."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
    
    def generate(self, ts_analyzer: TreeSequenceAnalyzer, labels: Dict[int, str], 
                 seed_host_match_path: Path) -> None:
        """Generate Newick files for each root."""
        df_match = pd.read_csv(seed_host_match_path)
        match_dict = df_match.set_index('host_id')['seed'].to_dict()
        
        for root in ts_analyzer.tree.roots:
            root_subpop = ts_analyzer.ts_data.sampled_tree.tables.individuals[root].metadata["subpopulation"]
            seed_id = match_dict.get(root_subpop, -1)
            
            if seed_id >= 0:
                output_path = self.output_dir / f"{seed_id}.nwk"
                newick_str = ts_analyzer.tree.as_newick(root=root, node_labels=labels)
                
                with open(output_path, 'w') as f:
                    f.write(newick_str + "\n")


class MetadataGenerator:
    """Generates metadata CSV files."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
    
    def generate(self, ts_analyzer: TreeSequenceAnalyzer, trait_values: List[Dict[int, float]], 
                 node_colors: Dict[int, str], trait_config: TraitConfiguration) -> None:
        """Generate metadata CSV file."""
        metadata_list = []
        tables = ts_analyzer.ts_data.sampled_tree.tables
        
        for node_id in ts_analyzer.traversal_order:
            # Extract node information
            node_time = ts_analyzer.ts_data.n_generations - tables.nodes.time[node_id].astype(int)
            
            if node_id < len(ts_analyzer.ts_data.sample_size):
                subpop_id = tables.individuals[node_id].metadata["subpopulation"]
                name = f"{node_time}.{subpop_id}"
            else:
                subpop_id = -1
                name = "."
            
            parent_id = ts_analyzer.tree.parent(node_id)
            
            # Create metadata object
            metadata = NodeMetadata(
                node_id=node_id,
                name=name,
                node_time=node_time,
                subpopulation_id=subpop_id,
                parent_id=parent_id if parent_id != -1 else -1,
                color=node_colors.get(node_id, "#000000"),
                trait_values=[tv[node_id] for tv in trait_values] if trait_values else []
            )
            
            metadata_list.append(metadata)
        
        self._write_csv(metadata_list, trait_config)
    
    def _write_csv(self, metadata_list: List[NodeMetadata], trait_config: TraitConfiguration) -> None:
        """Write metadata to CSV file."""
        with open(self.output_path, 'w') as f:
            # Write header
            header = ["node_id", "name", "node_time", "subpop_id", "parent_id", "color_trait"]
            
            for i in range(trait_config.transmissibility_traits):
                header.append(f"transmissibility_{i + 1}")
            
            for i in range(trait_config.drug_resistance_traits):
                header.append(f"drug_resistance_{i + 1}")
            
            f.write(",".join(header) + "\n")
            
            # Write data
            for metadata in metadata_list:
                f.write(",".join(metadata.to_csv_row()) + "\n")


class SequenceOutputGenerator:
    """Generates sequence output files (VCF/FASTA)."""
    
    def __init__(self, working_dir: Path, reference_path: Path):
        self.working_dir = working_dir
        self.reference_path = reference_path
        self.reference_seq = None
        self._load_reference()
    
    def _load_reference(self):
        """Load reference sequence."""
        with open(self.reference_path) as f:
            for record in SeqIO.parse(f, 'fasta'):
                self.reference_seq = str(record.seq)
                break
    
    def generate_vcf(self, ts_data: TreeSequenceData, labels: Dict[int, str]) -> None:
        """Generate VCF file."""
        vcf_tmp = self.working_dir / "sampled_pathogen_sequences.vcf.tmp"
        vcf_final = self.working_dir / "sampled_pathogen_sequences.vcf"
        
        # Write temporary VCF
        with open(vcf_tmp, 'w') as f:
            converted_ts = pyslim.convert_alleles(ts_data.sampled_tree)
            converted_ts.write_vcf(f, individual_names=labels.values())
        
        # Adjust positions (1-indexed)
        with open(vcf_tmp, 'r') as infile, open(vcf_final, 'w') as outfile:
            for line in infile:
                if line.startswith("#"):
                    outfile.write(line)
                else:
                    fields = line.split("\t")
                    fields[1] = str(int(fields[1]) + 1)
                    outfile.write("\t".join(fields))
        
        # Clean up
        vcf_tmp.unlink()
    
    def generate_fasta(self, ts_data: TreeSequenceData) -> None:
        """Generate FASTA file."""
        # Run R script for initial processing
        rscript_path = Path(__file__).parent / "generate_fas.r"
        subprocess.run(["Rscript", str(rscript_path), str(self.working_dir)])
        
        # Generate full genome FASTA
        self._generate_full_fasta()
    
    def _generate_full_fasta(self):
        """Generate full genome FASTA from SNPs."""
        # Load SNP positions
        snp_positions = []
        snp_pos_file = self.working_dir / "final_samples_snp_pos.csv"
        
        with open(snp_pos_file) as f:
            for line in f:
                fields = line.strip().split(",")
                if len(fields) > 1 and fields[1].isdigit():
                    snp_positions.append(int(fields[1]) - 1)
        
        # Generate full sequences
        snp_fasta = self.working_dir / "sample.SNPs_only.fasta"
        full_fasta = self.working_dir / "sample.wholegenome.fasta"
        
        with open(full_fasta, 'w') as outfile:
            for record in SeqIO.parse(snp_fasta, 'fasta'):
                # Start with reference sequence
                full_seq = list(self.reference_seq)
                
                # Apply SNPs
                for i, pos in enumerate(snp_positions):
                    if i < len(record.seq):
                        full_seq[pos] = record.seq[i]
                
                # Write sequence
                outfile.write(f">{record.id}\n{''.join(full_seq)}\n")


# ========================= Visualization =========================

class TrajectoryPlotter:
    """Plots trajectory data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def plot_seir_trajectory(self, data: pd.DataFrame, host_size: int, n_generations: int,
                            output_name: str = "SEIR_trajectory.png") -> None:
        """Plot SEIR trajectory."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data.plot(kind='line', ax=ax, linewidth=3, cmap='viridis')
        
        ax.set_xlabel('Generations')
        ax.set_ylabel('Number of hosts')
        ax.set_title('SEIR Trajectory')
        ax.set_xlim(0, n_generations)
        ax.set_ylim(0, host_size)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name)
        plt.close()
    
    def plot_strain_distribution(self, data: pd.DataFrame, n_generations: int,
                                output_name: str = "strain_trajectory.png") -> None:
        """Plot strain distribution over time."""
        # Normalize data
        normalized = data.div(data.sum(axis=1), axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        normalized.plot(kind='area', stacked=True, ax=ax, cmap='viridis')
        
        ax.set_xlabel('Generations')
        ax.set_ylabel('Proportion of strains')
        ax.set_title('Strain Distribution Through Time')
        ax.set_xlim(0, n_generations)
        ax.set_ylim(0, 1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Strains')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name)
        plt.close()
    
    def plot_aggregate_trajectories(self, all_data: List[pd.DataFrame], host_size: int,
                                   n_generations: int, plot_type: str = "seir",
                                   output_name: str = None) -> None:
        """Plot aggregate trajectories from multiple runs."""
        if not all_data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual runs with transparency
        for data in all_data:
            data.plot(kind='line', ax=ax, alpha=0.3, legend=False, cmap='viridis')
        
        # Calculate and plot average
        avg_data = pd.concat(all_data).groupby(level=0).mean()
        avg_data.plot(kind='line', ax=ax, linewidth=3, cmap='viridis')
        
        # Configure plot
        ax.set_xlabel('Generations')
        
        if plot_type == "seir":
            ax.set_ylabel('Number of hosts')
            ax.set_title('SEIR Trajectory (All Runs)')
            ax.set_ylim(0, host_size)
        else:
            ax.set_ylabel('Proportion')
            ax.set_title('Strain Distribution (All Runs)')
            ax.set_ylim(0, 1)
        
        ax.set_xlim(0, n_generations)
        
        plt.tight_layout()
        
        if output_name is None:
            output_name = f"all_{plot_type}_trajectory.png"
        
        plt.savefig(self.output_dir / output_name)
        plt.close()


# ========================= Main Processor =========================

class SimulationProcessor:
    """Main processor for simulation results."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.working_dir = Path(config.get("working_dir", "."))
        self.reference_path = Path(config.get("reference_path", ""))
        self.seed_host_match_path = Path(config.get("seed_host_match_path", ""))
        
        # Initialize components
        self.trait_config = self._create_trait_config()
        self.trait_calculator = TraitCalculator(self.trait_config)
        self.color_mapper = ColorMapper()
        self.trajectory_plotter = TrajectoryPlotter(self.working_dir)
    
    def _create_trait_config(self) -> TraitConfiguration:
        """Create trait configuration from config dict."""
        n_traits = self.config.get("n_traits", {})
        return TraitConfiguration(
            transmissibility_traits=n_traits.get("transmissibility", 0),
            drug_resistance_traits=n_traits.get("drug_resistance", 0),
            color_trait_index=self.config.get("color_trait_index", 0),
            use_sigmoid=self.config.get("use_sigmoid", False)
        )
    
    def process_run(self, run_id: int) -> bool:
        """Process a single simulation run."""
        run_dir = self.working_dir / str(run_id)
        
        try:
            # Load tree sequence data
            ts_data = TreeSequenceData.from_file(run_dir)
            ts_analyzer = TreeSequenceAnalyzer(ts_data)
            
            # Generate node labels
            labels = ts_analyzer.get_node_labels()
            
            # Generate Newick output
            newick_gen = NewickGenerator(run_dir)
            newick_gen.generate(ts_analyzer, labels, self.seed_host_match_path)
            
            # Calculate traits if configured
            trait_values = []
            if self.config.get("use_genetic_model", False):
                causal_gene_path = self.working_dir / "causal_gene_info.csv"
                self.trait_calculator.load_causal_genes(causal_gene_path)
                trait_values = self.trait_calculator.calculate_traits(ts_analyzer)
            
            # Determine node colors
            if self.trait_config.color_trait_index > 0 and trait_values:
                node_colors = self.color_mapper.color_by_trait(
                    trait_values[self.trait_config.color_trait_index - 1]
                )
            else:
                node_colors = self.color_mapper.color_by_seed(
                    ts_analyzer, self.seed_host_match_path
                )
            
            # Generate metadata
            metadata_gen = MetadataGenerator(run_dir / "transmission_tree_metadata.csv")
            metadata_gen.generate(ts_analyzer, trait_values, node_colors, self.trait_config)
            
            # Generate sequence output if configured
            if self.config.get("output_sequences", {}).get("vcf", False):
                seq_gen = SequenceOutputGenerator(run_dir, self.reference_path)
                seq_gen.generate_vcf(ts_data, labels)
                
                if self.config.get("output_sequences", {}).get("fasta", False):
                    seq_gen.generate_fasta(ts_data)
            
            # Plot trajectories
            self._plot_run_trajectories(run_dir)
            
            logger.info(f"Successfully processed run {run_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process run {run_id}: {e}")
            return False
    
    def _plot_run_trajectories(self, run_dir: Path) -> None:
        """Plot trajectories for a single run."""
        # Plot SEIR trajectory
        seir_file = run_dir / "SEIR_trajectory.csv.gz"
        if seir_file.exists():
            seir_data = pd.read_csv(seir_file, header=None, names=["S", "E", "I", "R"])
            
            # Add initial state
            seed_size = self.config.get("seed_size", 1)
            host_size = self.config.get("host_size", 100)
            n_generations = self.config.get("n_generations", 100)
            
            init_state = pd.DataFrame({
                "S": [host_size - seed_size],
                "E": [0],
                "I": [seed_size],
                "R": [0]
            })
            seir_data = pd.concat([init_state, seir_data]).reset_index(drop=True)
            
            plotter = TrajectoryPlotter(run_dir)
            plotter.plot_seir_trajectory(seir_data, host_size, n_generations)
        
        # Plot strain distribution
        strain_file = run_dir / "strain_trajectory.csv.gz"
        if strain_file.exists():
            seed_size = self.config.get("seed_size", 1)
            n_generations = self.config.get("n_generations", 100)
            
            strain_data = pd.read_csv(
                strain_file, 
                header=None, 
                names=[str(i) for i in range(seed_size)]
            )
            
            # Add initial state
            init_state = pd.DataFrame({str(i): [1] for i in range(seed_size)})
            strain_data = pd.concat([init_state, strain_data]).reset_index(drop=True)
            
            plotter = TrajectoryPlotter(run_dir)
            plotter.plot_strain_distribution(strain_data, n_generations)
    
    def process_all_runs(self, run_ids: List[int]) -> None:
        """Process all simulation runs and create aggregate plots."""
        successful_runs = []
        
        for run_id in run_ids:
            if self.process_run(run_id):
                successful_runs.append(run_id)
        
        # Create aggregate plots
        if successful_runs:
            self._create_aggregate_plots(successful_runs)
        
        logger.info(f"Processed {len(successful_runs)}/{len(run_ids)} runs successfully")
    
    def _create_aggregate_plots(self, run_ids: List[int]) -> None:
        """Create aggregate plots from multiple runs."""
        output_dir = self.working_dir / "output_trajectories"
        output_dir.mkdir(exist_ok=True)
        
        plotter = TrajectoryPlotter(output_dir)
        
        # Aggregate SEIR data
        seir_data_list = []
        strain_data_list = []
        
        for run_id in run_ids:
            run_dir = self.working_dir / str(run_id)
            
            # Load SEIR data
            seir_file = run_dir / "SEIR_trajectory.csv.gz"
            if seir_file.exists():
                data = pd.read_csv(seir_file, header=None, names=["S", "E", "I", "R"])
                
                # Add initial state
                seed_size = self.config.get("seed_size", 1)
                host_size = self.config.get("host_size", 100)
                
                init_state = pd.DataFrame({
                    "S": [host_size - seed_size],
                    "E": [0],
                    "I": [seed_size],
                    "R": [0]
                })
                data = pd.concat([init_state, data]).reset_index(drop=True)
                seir_data_list.append(data)
            
            # Load strain data
            strain_file = run_dir / "strain_trajectory.csv.gz"
            if strain_file.exists():
                seed_size = self.