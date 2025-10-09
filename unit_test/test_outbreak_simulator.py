import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json, os, sys
from unittest.mock import Mock, patch

curr_dir = os.path.dirname(__file__)
e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
if e3SIM_dir not in sys.path:
    sys.path.insert(0, e3SIM_dir)

from outbreak_simulator import (
    EpiModel,
    SubstitutionModel,
    LinkFunction,
    NetworkConfig,
    EvolutionConfig,
    EpidemiologyConfig,
    SeedInfo,
    GenomeElement,
    Postprocessing,
    SimulationConfig,
    ConfigValidator,
    ConfigParser,
    SlimScriptGenerator,
    SimulationRunner,
    SimulationOrchestrator,
)
from error_handling import CustomizedError


# ========================= Test Enums =========================

class TestEnums:
    """Test enum definitions."""
    
    def test_epi_model_values(self):
        assert EpiModel.SIR.value == "SIR"
        assert EpiModel.SEIR.value == "SEIR"
    
    def test_substitution_model_values(self):
        assert SubstitutionModel.MUT_RATE.value == "mut_rate"
        assert SubstitutionModel.MUT_RATE_MATRIX.value == "mut_rate_matrix"
    
    def test_link_function_values(self):
        assert LinkFunction.LOGIT.value == "logit"
        assert LinkFunction.CLOGLOG.value == "cloglog"


# ========================= Test ConfigValidator =========================

class TestConfigValidator:
    """Test configuration validation methods."""
    
    def test_validate_boolean_valid(self):
        ConfigValidator.validate_boolean(True, "test param")
        ConfigValidator.validate_boolean(False, "test param")
    
    def test_validate_boolean_invalid(self):
        with pytest.raises(CustomizedError, match="must be a boolean"):
            ConfigValidator.validate_boolean("true", "test param")
        with pytest.raises(CustomizedError, match="must be a boolean"):
            ConfigValidator.validate_boolean(1, "test param")
    
    def test_validate_integer_valid(self):
        ConfigValidator.validate_integer(5, "test param", min_val=1)
        ConfigValidator.validate_integer(100, "test param", min_val=1)
    
    def test_validate_integer_invalid_type(self):
        with pytest.raises(CustomizedError, match="must be an integer"):
            ConfigValidator.validate_integer(5.5, "test param")
        with pytest.raises(CustomizedError, match="must be an integer"):
            ConfigValidator.validate_integer("5", "test param")
    
    def test_validate_integer_below_minimum(self):
        with pytest.raises(CustomizedError, match="must be an integer >= 10"):
            ConfigValidator.validate_integer(5, "test param", min_val=10)
    
    def test_validate_float_valid(self):
        ConfigValidator.validate_float(0.5, "test param", min_val=0.0, max_val=1.0)
        ConfigValidator.validate_float(5, "test param")  # int should work
    
    def test_validate_float_invalid_type(self):
        with pytest.raises(CustomizedError, match="must be a number"):
            ConfigValidator.validate_float("0.5", "test param")
    
    def test_validate_float_out_of_range(self):
        with pytest.raises(CustomizedError, match="must be >= 0"):
            ConfigValidator.validate_float(-1.0, "test param", min_val=0.0)
        with pytest.raises(CustomizedError, match="must be <= 1"):
            ConfigValidator.validate_float(1.5, "test param", max_val=1.0)
    
    def test_validate_probability_valid(self):
        ConfigValidator.validate_probability(0.0, "test param")
        ConfigValidator.validate_probability(0.5, "test param")
        ConfigValidator.validate_probability(1.0, "test param")
    
    def test_validate_probability_strict(self):
        with pytest.raises(CustomizedError, match="must be strictly between 0 and 1"):
            ConfigValidator.validate_probability(0.0, "test param", strict=True)
        with pytest.raises(CustomizedError, match="must be strictly between 0 and 1"):
            ConfigValidator.validate_probability(1.0, "test param", strict=True)
        
        ConfigValidator.validate_probability(0.5, "test param", strict=True)
    
    def test_validate_mutation_matrix_valid(self):
        matrix = np.array([
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.3, 0.4],
            [0.2, 0.3, 0, 0.5],
            [0.3, 0.4, 0.5, 0]
        ])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name
        
        try:
            result = ConfigValidator.validate_and_write_mutation_matrix(matrix, temp_path)
            assert result is True
            df = pd.read_csv(temp_path, index_col=0)
            assert df.shape == (4, 4)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_validate_mutation_matrix_invalid_diagonal(self):
        matrix = np.array([
            [0.1, 0.1, 0.2, 0.3],
            [0.1, 0, 0.3, 0.4],
            [0.2, 0.3, 0, 0.5],
            [0.3, 0.4, 0.5, 0]
        ])
        result = ConfigValidator.validate_and_write_mutation_matrix(matrix, "dummy.csv")
        assert result is False
    
    def test_validate_mutation_matrix_negative_values(self):
        matrix = np.array([
            [0, -0.1, 0.2, 0.3],
            [0.1, 0, 0.3, 0.4],
            [0.2, 0.3, 0, 0.5],
            [0.3, 0.4, 0.5, 0]
        ])
        result = ConfigValidator.validate_and_write_mutation_matrix(matrix, "dummy.csv")
        assert result is False


# ========================= Test NetworkConfig =========================

class TestNetworkConfig:
    """Test NetworkConfig dataclass."""
    
    def test_valid_network_config(self, tmp_path):
        network_path = tmp_path / "contact_network.adjlist"
        network_path.write_text("1 2\n2 3\n")
        
        config = NetworkConfig(
            host_size=100,
            contact_network_path=network_path
        )
        assert config.host_size == 100
    
    def test_network_config_missing_file(self, tmp_path):
        with pytest.raises(CustomizedError, match="NetworkGenerator hasn't been run"):
            NetworkConfig(
                host_size=100,
                contact_network_path=tmp_path / "nonexistent.adjlist"
            )
    
    def test_network_config_invalid_host_size(self, tmp_path):
        network_path = tmp_path / "contact_network.adjlist"
        network_path.write_text("1 2\n")
        
        with pytest.raises(CustomizedError, match="must be an integer"):
            NetworkConfig(host_size=-10, contact_network_path=network_path)


# ========================= Test EvolutionConfig =========================

class TestEvolutionConfig:
    """Test EvolutionConfig dataclass."""
    
    def test_valid_evolution_config_mut_rate(self, tmp_path):
        config = EvolutionConfig(
            n_generation=100,
            subst_model_parameterization=SubstitutionModel.MUT_RATE,
            transition_matrix_path=tmp_path / "matrix.csv",
            mut_rate=0.001,
            within_host_reproduction=False,
            cap_withinhost=1
        )
        assert config.n_generation == 100
        assert config.mut_rate == 0.001
    
    def test_valid_evolution_config_mut_matrix(self, tmp_path):
        matrix = np.array([
            [0, 0.1, 0.2, 0.3],
            [0.1, 0, 0.3, 0.4],
            [0.2, 0.3, 0, 0.5],
            [0.3, 0.4, 0.5, 0]
        ])
        
        config = EvolutionConfig(
            n_generation=100,
            subst_model_parameterization=SubstitutionModel.MUT_RATE_MATRIX,
            transition_matrix_path=tmp_path / "matrix.csv",
            mut_rate_matrix=matrix,
            cap_withinhost=1
        )
        assert config.n_generation == 100
        assert (tmp_path / "matrix.csv").exists()
    
    def test_evolution_config_invalid_mut_matrix(self, tmp_path):
        bad_matrix = np.array([
            [0.1, 0.1, 0.2, 0.3],
            [0.1, 0, 0.3, 0.4],
            [0.2, 0.3, 0, 0.5],
            [0.3, 0.4, 0.5, 0]
        ])
        
        with pytest.raises(CustomizedError, match="does NOT meet the requirement"):
            EvolutionConfig(
                n_generation=100,
                subst_model_parameterization=SubstitutionModel.MUT_RATE_MATRIX,
                transition_matrix_path=tmp_path / "matrix.csv",
                mut_rate_matrix=bad_matrix,
                cap_withinhost=1
            )
    
    def test_within_host_reproduction_warnings(self, tmp_path, capsys):
        _ = EvolutionConfig(
            n_generation=100,
            subst_model_parameterization=SubstitutionModel.MUT_RATE,
            transition_matrix_path=tmp_path / "matrix.csv",
            mut_rate=0.001,
            within_host_reproduction=True,
            within_host_reproduction_rate=0.5,
            cap_withinhost=1
        )
        captured = capsys.readouterr()
        assert "Within-host capacity is 1" in captured.out
    
    def test_within_host_zero_rate_warning(self, tmp_path, capsys):
        _ = EvolutionConfig(
            n_generation=100,
            subst_model_parameterization=SubstitutionModel.MUT_RATE,
            transition_matrix_path=tmp_path / "matrix.csv",
            mut_rate=0.001,
            within_host_reproduction=True,
            within_host_reproduction_rate=0.0,
            cap_withinhost=5
        )
        captured = capsys.readouterr()
        assert "reproduction rate is 0" in captured.out


# ========================= Test EpidemiologyConfig =========================

class TestEpidemiologyConfig:
    """Test EpidemiologyConfig dataclass."""
    
    def test_valid_sir_config(self):
        config = EpidemiologyConfig(
            model="SIR",
            n_epoch=1,
            n_generation=100,
            transmissibility_effsize=[0],
            drug_resistance_effsize=[0],
            S_IE_rate=[0.1],
            I_R_rate=[0.05],
            R_S_rate=[0.01],
            latency_prob=[0],
            E_I_rate=[0],
            I_E_rate=[0],
            E_R_rate=[0],
            surviv_prob=[0.95],
            sample_rate=[0.1],
            recovery_prob_after_sampling=[0.5],
            n_massive_sample=0,
            massive_sample_generation=[],
            massive_sample_prob=[],
            massive_sample_recover_prob=[],
            cap_withinhost=1,
            slim_replicate_seed_file_path="",
            traits_num={"transmissibility": 0, "drug_resistance": 0}
        )
        assert config.model == EpiModel.SIR
        assert config.latency_prob == [0]
    
    def test_valid_seir_config(self):
        config = EpidemiologyConfig(
            model="SEIR",
            n_epoch=1,
            n_generation=100,
            transmissibility_effsize=[1],
            drug_resistance_effsize=[1],
            S_IE_rate=[0.1],
            I_R_rate=[0.05],
            R_S_rate=[0.01],
            latency_prob=[0.3],
            E_I_rate=[0.2],
            I_E_rate=[0.0],
            E_R_rate=[0.01],
            surviv_prob=[0.95],
            sample_rate=[0.1],
            recovery_prob_after_sampling=[0.5],
            n_massive_sample=0,
            massive_sample_generation=[],
            massive_sample_prob=[],
            massive_sample_recover_prob=[],
            cap_withinhost=1,
            slim_replicate_seed_file_path="",
            traits_num={"transmissibility": 2, "drug_resistance": 2}
        )
        assert config.model == EpiModel.SEIR
        assert config.latency_prob == [0.3]
    
    def test_invalid_model_name(self):
        with pytest.raises(CustomizedError, match="Invalid model"):
            EpidemiologyConfig(
                model="SIRS",
                n_epoch=1,
                n_generation=100,
                transmissibility_effsize=[0],
                drug_resistance_effsize=[0],
                S_IE_rate=[0.1],
                I_R_rate=[0.05],
                R_S_rate=[0.01],
                latency_prob=[0],
                E_I_rate=[0],
                I_E_rate=[0],
                E_R_rate=[0],
                surviv_prob=[0.95],
                sample_rate=[0.1],
                recovery_prob_after_sampling=[0.5],
                n_massive_sample=0,
                massive_sample_generation=[],
                massive_sample_prob=[],
                massive_sample_recover_prob=[],
                cap_withinhost=1,
                slim_replicate_seed_file_path="",
                traits_num={"transmissibility": 0, "drug_resistance": 0}
            )
    
    def test_epoch_changing_validation_missing_list(self):
        with pytest.raises(CustomizedError, match="epoch_changing_generation"):
            EpidemiologyConfig(
                model="SIR",
                n_epoch=2,
                n_generation=100,
                transmissibility_effsize=[0, 0],
                drug_resistance_effsize=[0, 0],
                S_IE_rate=[0.1, 0.2],
                I_R_rate=[0.05, 0.05],
                R_S_rate=[0.01, 0.01],
                latency_prob=[0, 0],
                E_I_rate=[0, 0],
                I_E_rate=[0, 0],
                E_R_rate=[0, 0],
                surviv_prob=[0.95, 0.95],
                sample_rate=[0.1, 0.1],
                recovery_prob_after_sampling=[0.5, 0.5],
                n_massive_sample=0,
                massive_sample_generation=[],
                massive_sample_prob=[],
                massive_sample_recover_prob=[],
                cap_withinhost=1,
                slim_replicate_seed_file_path="",
                traits_num={"transmissibility": 0, "drug_resistance": 0},
                epoch_changing_generation=[]
            )
    
    def test_epoch_changing_invalid_generation(self):
        with pytest.raises(CustomizedError, match="must be integers in range"):
            EpidemiologyConfig(
                model="SIR",
                n_epoch=2,
                n_generation=100,
                transmissibility_effsize=[0, 0],
                drug_resistance_effsize=[0, 0],
                S_IE_rate=[0.1, 0.2],
                I_R_rate=[0.05, 0.05],
                R_S_rate=[0.01, 0.01],
                latency_prob=[0, 0],
                E_I_rate=[0, 0],
                I_E_rate=[0, 0],
                E_R_rate=[0, 0],
                surviv_prob=[0.95, 0.95],
                sample_rate=[0.1, 0.1],
                recovery_prob_after_sampling=[0.5, 0.5],
                n_massive_sample=0,
                massive_sample_generation=[],
                massive_sample_prob=[],
                massive_sample_recover_prob=[],
                cap_withinhost=1,
                slim_replicate_seed_file_path="",
                traits_num={"transmissibility": 0, "drug_resistance": 0},
                epoch_changing_generation=[150]
            )
    
    def test_probability_not_list(self):
        with pytest.raises(CustomizedError, match="has to be a list"):
            EpidemiologyConfig(
                model="SIR",
                n_epoch=1,
                n_generation=100,
                transmissibility_effsize=[0],
                drug_resistance_effsize=[0],
                S_IE_rate=0.1,  # Should be list
                I_R_rate=[0.05],
                R_S_rate=[0.01],
                latency_prob=[0],
                E_I_rate=[0],
                I_E_rate=[0],
                E_R_rate=[0],
                surviv_prob=[0.95],
                sample_rate=[0.1],
                recovery_prob_after_sampling=[0.5],
                n_massive_sample=0,
                massive_sample_generation=[],
                massive_sample_prob=[],
                massive_sample_recover_prob=[],
                cap_withinhost=1,
                slim_replicate_seed_file_path="",
                traits_num={"transmissibility": 0, "drug_resistance": 0}
            )
    
    def test_probability_list_length_mismatch(self):
        with pytest.raises(CustomizedError, match="must have the same length"):
            EpidemiologyConfig(
                model="SIR",
                n_epoch=2,
                n_generation=100,
                transmissibility_effsize=[0, 0],
                drug_resistance_effsize=[0, 0],
                S_IE_rate=[0.1],
                I_R_rate=[0.05, 0.05],
                R_S_rate=[0.01, 0.01],
                latency_prob=[0, 0],
                E_I_rate=[0, 0],
                I_E_rate=[0, 0],
                E_R_rate=[0, 0],
                surviv_prob=[0.95, 0.95],
                sample_rate=[0.1, 0.1],
                recovery_prob_after_sampling=[0.5, 0.5],
                n_massive_sample=0,
                massive_sample_generation=[],
                massive_sample_prob=[],
                massive_sample_recover_prob=[],
                cap_withinhost=1,
                slim_replicate_seed_file_path="",
                traits_num={"transmissibility": 0, "drug_resistance": 0},
                epoch_changing_generation=[50]
            )
    
    def test_massive_sampling_generation_length_mismatch(self):
        with pytest.raises(CustomizedError, match="must be a list of length 2"):
            EpidemiologyConfig(
                model="SIR",
                n_epoch=1,
                n_generation=100,
                transmissibility_effsize=[0],
                drug_resistance_effsize=[0],
                S_IE_rate=[0.1],
                I_R_rate=[0.05],
                R_S_rate=[0.01],
                latency_prob=[0],
                E_I_rate=[0],
                I_E_rate=[0],
                E_R_rate=[0],
                surviv_prob=[0.95],
                sample_rate=[0.1],
                recovery_prob_after_sampling=[0.5],
                n_massive_sample=2,
                massive_sample_generation=[10],
                massive_sample_prob=[0.5, 0.5],
                massive_sample_recover_prob=[0.3, 0.3],
                cap_withinhost=1,
                slim_replicate_seed_file_path="",
                traits_num={"transmissibility": 0, "drug_resistance": 0}
            )
    
    def test_massive_sampling_prob_length_mismatch(self):
        with pytest.raises(CustomizedError, match="must be a list of length 2"):
            EpidemiologyConfig(
                model="SIR",
                n_epoch=1,
                n_generation=100,
                transmissibility_effsize=[0],
                drug_resistance_effsize=[0],
                S_IE_rate=[0.1],
                I_R_rate=[0.05],
                R_S_rate=[0.01],
                latency_prob=[0],
                E_I_rate=[0],
                I_E_rate=[0],
                E_R_rate=[0],
                surviv_prob=[0.95],
                sample_rate=[0.1],
                recovery_prob_after_sampling=[0.5],
                n_massive_sample=2,
                massive_sample_generation=[10, 20],
                massive_sample_prob=[0.5],
                massive_sample_recover_prob=[0.3, 0.3],
                cap_withinhost=1,
                slim_replicate_seed_file_path="",
                traits_num={"transmissibility": 0, "drug_resistance": 0}
            )
    
    def test_effect_size_exceeds_traits(self):
        with pytest.raises(CustomizedError, match="must be chosen from"):
            EpidemiologyConfig(
                model="SIR",
                n_epoch=1,
                n_generation=100,
                transmissibility_effsize=[3],
                drug_resistance_effsize=[0],
                S_IE_rate=[0.1],
                I_R_rate=[0.05],
                R_S_rate=[0.01],
                latency_prob=[0],
                E_I_rate=[0],
                I_E_rate=[0],
                E_R_rate=[0],
                surviv_prob=[0.95],
                sample_rate=[0.1],
                recovery_prob_after_sampling=[0.5],
                n_massive_sample=0,
                massive_sample_generation=[],
                massive_sample_prob=[],
                massive_sample_recover_prob=[],
                cap_withinhost=1,
                slim_replicate_seed_file_path="",
                traits_num={"transmissibility": 2, "drug_resistance": 0}
            )
    
    def test_super_infection_warning(self, capsys):
        config = EpidemiologyConfig(
            model="SIR",
            n_epoch=1,
            n_generation=100,
            transmissibility_effsize=[0],
            drug_resistance_effsize=[0],
            S_IE_rate=[0.1],
            I_R_rate=[0.05],
            R_S_rate=[0.01],
            latency_prob=[0],
            E_I_rate=[0],
            I_E_rate=[0],
            E_R_rate=[0],
            surviv_prob=[0.95],
            sample_rate=[0.1],
            recovery_prob_after_sampling=[0.5],
            n_massive_sample=0,
            massive_sample_generation=[],
            massive_sample_prob=[],
            massive_sample_recover_prob=[],
            cap_withinhost=1,
            slim_replicate_seed_file_path="",
            traits_num={"transmissibility": 0, "drug_resistance": 0},
            super_infection=True
        )
        captured = capsys.readouterr()
        assert "Super-infection is activated" in captured.out


# ========================= Test SeedInfo =========================

class TestSeedInfo:
    """Test SeedInfo dataclass."""
    
    def test_valid_seed_info_with_reference(self, tmp_path):
        seed_host_match = tmp_path / "seed_host_match.csv"
        seed_host_match.write_text("seed,host\n1,1\n")
        
        config = SeedInfo(
            seed_size=1,
            workding_dir=tmp_path,
            seed_host_matching_path=str(seed_host_match),
            use_reference=True
        )
        assert config.seed_size == 1
        assert config.use_reference is True
    
    def test_seed_info_with_originalvcfs(self, tmp_path):
        seed_host_match = tmp_path / "seed_host_match.csv"
        seed_host_match.write_text("seed,host\n1,1\n")
        
        vcf_dir = tmp_path / "originalvcfs"
        vcf_dir.mkdir()
        
        config = SeedInfo(
            seed_size=1,
            workding_dir=tmp_path,
            seed_host_matching_path=str(seed_host_match),
            use_reference=False
        )
        assert config.use_reference is False
    
    def test_seed_info_missing_vcf_dir(self, tmp_path):
        seed_host_match = tmp_path / "seed_host_match.csv"
        seed_host_match.write_text("seed,host\n1,1\n")
        
        with pytest.raises(CustomizedError, match="SeedGenerator hasn't been run"):
            SeedInfo(
                seed_size=1,
                workding_dir=tmp_path,
                seed_host_matching_path=str(seed_host_match),
                use_reference=False
            )
    
    def test_seed_info_missing_matching_file(self, tmp_path):
        with pytest.raises(CustomizedError, match="HostSeedMatcher hasn't been run"):
            SeedInfo(
                seed_size=1,
                workding_dir=tmp_path,
                seed_host_matching_path=str(tmp_path / "nonexistent.csv"),
                use_reference=True
            )


# ========================= Test GenomeElement =========================

class TestGenomeElement:
    """Test GenomeElement dataclass."""
    
    def test_valid_genome_element(self, tmp_path):
        ref_path = tmp_path / "reference.fasta"
        ref_path.write_text(">chr1\nATCG\n")
        
        causal_gene_path = tmp_path / "causal_gene_info.csv"
        causal_gene_path.write_text("gene,effect\ngene1,0.5\n")
        
        config = GenomeElement(
            ref_path=str(ref_path),
            use_genetic_model=True,
            workding_dir=str(tmp_path),
            traits_num={"transmissibility": 2, "drug_resistance": 1},
            link="logit",
            alpha_trans=[0.5, 0.3],
            alpha_drug=[0.2],
            causal_gene_path=causal_gene_path
        )
        assert config.use_genetic_model is True
        assert len(config.alpha_trans) == 2
    
    def test_genome_element_missing_ref(self, tmp_path):
        with pytest.raises(CustomizedError, match="doesn't exist"):
            GenomeElement(
                ref_path=str(tmp_path / "nonexistent.fasta"),
                use_genetic_model=False,
                workding_dir=str(tmp_path),
                traits_num={"transmissibility": 0, "drug_resistance": 0},
                link="logit",
                alpha_trans=[],
                alpha_drug=[],
                causal_gene_path=tmp_path / "causal.csv"
            )
    
    def test_genome_element_missing_causal_genes(self, tmp_path):
        ref_path = tmp_path / "reference.fasta"
        ref_path.write_text(">chr1\nATCG\n")
        
        with pytest.raises(CustomizedError, match="GeneticElementGenerator hasn't been run"):
            GenomeElement(
                ref_path=str(ref_path),
                use_genetic_model=True,
                workding_dir=str(tmp_path),
                traits_num={"transmissibility": 2, "drug_resistance": 1},
                link="logit",
                alpha_trans=[0.5, 0.3],
                alpha_drug=[0.2],
                causal_gene_path=tmp_path / "nonexistent.csv"
            )
    
    def test_genome_element_invalid_traits_num(self, tmp_path):
        ref_path = tmp_path / "reference.fasta"
        ref_path.write_text(">chr1\nATCG\n")
        
        with pytest.raises(CustomizedError, match="must be a dict"):
            GenomeElement(
                ref_path=str(ref_path),
                use_genetic_model=False,
                workding_dir=str(tmp_path),
                traits_num={"transmissibility": 0},
                link="logit",
                alpha_trans=[],
                alpha_drug=[],
                causal_gene_path=tmp_path / "causal.csv"
            )
    
    def test_genome_element_alpha_not_list(self, tmp_path):
        ref_path = tmp_path / "reference.fasta"
        ref_path.write_text(">chr1\nATCG\n")
        
        with pytest.raises(CustomizedError, match="has to be a list"):
            GenomeElement(
                ref_path=str(ref_path),
                use_genetic_model=False,
                workding_dir=str(tmp_path),
                traits_num={"transmissibility": 1, "drug_resistance": 0},
                link="logit",
                alpha_trans=0.5,  # Should be list
                alpha_drug=[],
                causal_gene_path=tmp_path / "causal.csv"
            )


# ========================= Test Postprocessing =========================

class TestPostprocessing:
    """Test Postprocessing dataclass."""
    
    def test_valid_postprocessing_enabled(self):
        config = Postprocessing(
            do_process=True,
            n_traits={"transmissibility": 2, "drug_resistance": 1},
            branch_color_trait=1,
            heatmap_trait="transmissibility",
            vcf=True,
            fasta=False
        )
        assert config.do_process is True
        assert config.branch_color_trait == 1
    
    def test_postprocessing_disabled(self, capsys):
        _ = Postprocessing(
            do_process=False,
            n_traits={"transmissibility": 0, "drug_resistance": 0},
            branch_color_trait=0,
            heatmap_trait="none",
            vcf=False,
            fasta=False
        )
        captured = capsys.readouterr()
        assert "not enabled" in captured.out
    
    def test_invalid_branch_color_trait(self):
        with pytest.raises(CustomizedError, match="should be an integer"):
            Postprocessing(
                do_process=True,
                n_traits={"transmissibility": 2, "drug_resistance": 1},
                branch_color_trait=10,
                heatmap_trait="none",
                vcf=False,
                fasta=False
            )
    
    def test_invalid_heatmap_trait(self):
        with pytest.raises(CustomizedError, match="not permitted"):
            Postprocessing(
                do_process=True,
                n_traits={"transmissibility": 2, "drug_resistance": 1},
                branch_color_trait=0,
                heatmap_trait="invalid_trait",
                vcf=False,
                fasta=False
            )
    
    def test_branch_color_by_seed(self, capsys):
        _ = Postprocessing(
            do_process=True,
            n_traits={"transmissibility": 2, "drug_resistance": 1},
            branch_color_trait=0,
            heatmap_trait="none",
            vcf=False,
            fasta=False
        )
        captured = capsys.readouterr()
        assert "colored by seed" in captured.out


# ========================= Test ConfigParser =========================

class TestConfigParser:
    """Test ConfigParser class."""
    
    def test_parse_minimal_config(self, tmp_path):
        # Create necessary files
        network_path = tmp_path / "contact_network.adjlist"
        network_path.write_text("1 2\n")
        
        ref_path = tmp_path / "reference.fasta"
        ref_path.write_text(">chr1\nATCG\n")
        
        seed_host_match = tmp_path / "seed_host_match.csv"
        seed_host_match.write_text("seed,host\n1,1\n")
        
        causal_gene_path = tmp_path / "causal_gene_info.csv"
        causal_gene_path.write_text("gene,effect\ngene1,0.5\n")
        
        config_dict = {
            "BasicRunConfiguration": {
                "cwdir": str(tmp_path),
                "n_replicates": 1
            },
            "NetworkModelParameters": {
                "host_size": 100
            },
            "EvolutionModel": {
                "n_generation": 100,
                "subst_model_parameterization": "mut_rate",
                "mut_rate": 0.001,
                "within_host_reproduction": False,
                "cap_withinhost": 1
            },
            "GenomeElement": {
                "ref_path": str(ref_path),
                "use_genetic_model": True,
                "traits_num": {"transmissibility": 2, "drug_resistance": 1},
                "trait_prob_link": {
                    "link": "logit",
                    "logit": {
                        "alpha_trans": [0.5, 0.3],
                        "alpha_drug": [0.2]
                    }
                }
            },
            "EpidemiologyModel": {
                "model": "SIR",
                "epoch_changing": {"n_epoch": 1},
                "super_infection": False,
                "genetic_architecture": {
                    "transmissibility": [0],
                    "drug_resistance": [0]
                },
                "transition_prob": {
                    "S_IE_prob": [0.1],
                    "I_R_prob": [0.05],
                    "R_S_prob": [0.01],
                    "latency_prob": [0],
                    "E_I_prob": [0],
                    "I_E_prob": [0],
                    "E_R_prob": [0],
                    "surviv_prob": [0.95],
                    "sample_prob": [0.1],
                    "recovery_prob_after_sampling": [0.5]
                },
                "massive_sampling": {
                    "event_num": 0,
                    "generation": [],
                    "sampling_prob": [],
                    "recovery_prob_after_sampling": []
                },
                "slim_replicate_seed_file_path": None
            },
            "SeedsConfiguration": {
                "seed_size": 1,
                "use_reference": True
            },
            "Postprocessing_options": {
                "do_postprocess": False,
                "tree_plotting": {
                    "branch_color_trait": 0,
                    "heatmap": "none"
                },
                "sequence_output": {
                    "vcf": False,
                    "fasta": False
                }
            }
        }
        
        parser = ConfigParser()
        config = parser.parse_config_dict(config_dict)
        
        assert isinstance(config, SimulationConfig)
        assert config.n_replicates == 1
        assert config.network.host_size == 100


# ========================= Test SlimScriptGenerator =========================

class TestSlimScriptGenerator:
    """Test SlimScriptGenerator class."""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock configuration."""
        network_path = tmp_path / "contact_network.adjlist"
        network_path.write_text("1 2\n")
        
        ref_path = tmp_path / "reference.fasta"
        ref_path.write_text(">chr1\nATCG\n")
        
        seed_host_match = tmp_path / "seed_host_match.csv"
        seed_host_match.write_text("seed,host\n1,1\n")
        
        causal_gene_path = tmp_path / "causal_gene_info.csv"
        causal_gene_path.write_text("gene,effect\ngene1,0.5\n")
        
        network_config = NetworkConfig(
            host_size=100,
            contact_network_path=network_path
        )
        
        evolution_config = EvolutionConfig(
            n_generation=100,
            subst_model_parameterization=SubstitutionModel.MUT_RATE,
            transition_matrix_path=tmp_path / "matrix.csv",
            mut_rate=0.001,
            cap_withinhost=1
        )
        
        genome_config = GenomeElement(
            ref_path=str(ref_path),
            use_genetic_model=True,
            workding_dir=str(tmp_path),
            traits_num={"transmissibility": 2, "drug_resistance": 1},
            link="logit",
            alpha_trans=[0.5, 0.3],
            alpha_drug=[0.2],
            causal_gene_path=causal_gene_path
        )
        
        epidemiology_config = EpidemiologyConfig(
            model="SIR",
            n_epoch=1,
            n_generation=100,
            transmissibility_effsize=[0],
            drug_resistance_effsize=[0],
            S_IE_rate=[0.1],
            I_R_rate=[0.05],
            R_S_rate=[0.01],
            latency_prob=[0],
            E_I_rate=[0],
            I_E_rate=[0],
            E_R_rate=[0],
            surviv_prob=[0.95],
            sample_rate=[0.1],
            recovery_prob_after_sampling=[0.5],
            n_massive_sample=0,
            massive_sample_generation=[],
            massive_sample_prob=[],
            massive_sample_recover_prob=[],
            cap_withinhost=1,
            slim_replicate_seed_file_path="",
            traits_num={"transmissibility": 2, "drug_resistance": 1}
        )
        
        seed_info = SeedInfo(
            seed_size=1,
            workding_dir=tmp_path,
            seed_host_matching_path=str(seed_host_match),
            use_reference=True
        )
        
        postprocess_config = Postprocessing(
            do_process=False,
            n_traits={"transmissibility": 2, "drug_resistance": 1},
            branch_color_trait=0,
            heatmap_trait="none",
            vcf=False,
            fasta=False
        )
        
        return SimulationConfig(
            working_dir=tmp_path,
            n_replicates=1,
            network=network_config,
            evolution=evolution_config,
            epidemiology=epidemiology_config,
            seed_info=seed_info,
            genome_config=genome_config,
            postprocess_config=postprocess_config
        )
    
    def test_generate_script_basic(self, mock_config, tmp_path):
        """Test basic script generation."""
        generator = SlimScriptGenerator(mock_config)
        generator.code_path = tmp_path / "slim_scripts"
        generator.code_path.mkdir()
        
        # Create dummy script files
        for component in [
            "trait_calc_function.slim",
            "initialization_pt1.slim",
            "read_config.slim",
            "initialization_pt2.slim",
            "genomic_element_init_effsize.slim",
            "initialization_pt3.slim",
            "mutation_effsize.slim",
            "block_control.slim",
            "seeds_read_in_noburnin.slim",
            "contact_network_read_in.slim",
            "self_reproduce.slim",
            "transmission_nogenetic.slim",
            "kill_old_pathogens.slim",
            "store_current_states.slim",
            "Infected_process_nogenetic.slim",
            "New_infection_process.slim",
            "Recovered_process.slim",
            "log.slim",
            "finish_simulation.slim"
        ]:
            (generator.code_path / component).write_text(f"// {component}\n")
        
        output_path = tmp_path / "test_script.slim"
        result = generator.generate_script(output_path)
        
        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_script_components_sir_no_genetic(self, mock_config):
        """Test script components for SIR without genetic model."""
        mock_config.genome_config.use_genetic_model = False
        mock_config.epidemiology.model = EpiModel.SIR
        
        generator = SlimScriptGenerator(mock_config)
        generator._add_initialization()
        generator._add_mutation_blocks()
        
        assert "genomic_element_init.slim" in generator.script_components
        assert "genomic_element_init_effsize.slim" not in generator.script_components
        assert "mutation_effsize.slim" not in generator.script_components
    
    def test_script_components_seir_with_genetic(self, mock_config):
        """Test script components for SEIR with genetic model."""
        mock_config.epidemiology.model = EpiModel.SEIR
        mock_config.genome_config.use_genetic_model = True
        
        generator = SlimScriptGenerator(mock_config)
        generator._add_initialization()
        generator._add_state_transitions()
        
        assert "genomic_element_init_effsize.slim" in generator.script_components
        assert "Exposed_process.slim" in generator.script_components
    
    def test_script_components_massive_sampling(self, mock_config):
        """Test script components include massive sampling."""
        mock_config.epidemiology.n_massive_sample = 2
        
        generator = SlimScriptGenerator(mock_config)
        generator._add_state_transitions()
        
        assert "massive_sampling.slim" in generator.script_components
    
    def test_script_components_super_infection(self, mock_config):
        """Test script components for super infection."""
        mock_config.epidemiology.super_infection = True
        mock_config.evolution.cap_withinhost = 5
        
        generator = SlimScriptGenerator(mock_config)
        generator._add_state_transitions()
        
        assert "New_infection_process_superinfection.slim" in generator.script_components
    
    def test_script_components_within_host_reproduction(self, mock_config):
        """Test script components include within-host reproduction."""
        mock_config.evolution.within_host_reproduction = True
        
        generator = SlimScriptGenerator(mock_config)
        generator._add_transmission_blocks()
        
        assert "within_host_reproduce.slim" in generator.script_components
    
    def test_script_components_multiple_epochs(self, mock_config):
        """Test script components for multiple epochs."""
        mock_config.epidemiology.n_epoch = 2
        
        generator = SlimScriptGenerator(mock_config)
        generator._add_control_blocks()
        
        assert "change_epoch.slim" in generator.script_components
    
    def test_script_components_recovery_process(self, mock_config):
        """Test script components include recovery when R_S_rate > 0."""
        mock_config.epidemiology.R_S_rate = [0.01]
        
        generator = SlimScriptGenerator(mock_config)
        generator._add_state_transitions()
        
        assert "Recovered_process.slim" in generator.script_components


# ========================= Test SimulationRunner =========================

class TestSimulationRunner:
    """Test SimulationRunner class."""
    
    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a minimal mock configuration."""
        network_path = tmp_path / "contact_network.adjlist"
        network_path.write_text("1 2\n")
        
        ref_path = tmp_path / "reference.fasta"
        ref_path.write_text(">chr1\nATCG\n")
        
        seed_host_match = tmp_path / "seed_host_match.csv"
        seed_host_match.write_text("seed,host\n1,1\n")
        
        causal_gene_path = tmp_path / "causal_gene_info.csv"
        causal_gene_path.write_text("gene,effect\ngene1,0.5\n")
        
        network_config = NetworkConfig(host_size=100, contact_network_path=network_path)
        evolution_config = EvolutionConfig(
            n_generation=100,
            subst_model_parameterization=SubstitutionModel.MUT_RATE,
            transition_matrix_path=tmp_path / "matrix.csv",
            mut_rate=0.001,
            cap_withinhost=1
        )
        genome_config = GenomeElement(
            ref_path=str(ref_path),
            use_genetic_model=False,
            workding_dir=str(tmp_path),
            traits_num={"transmissibility": 0, "drug_resistance": 0},
            link="logit",
            alpha_trans=[],
            alpha_drug=[],
            causal_gene_path=causal_gene_path
        )
        epidemiology_config = EpidemiologyConfig(
            model="SIR",
            n_epoch=1,
            n_generation=100,
            transmissibility_effsize=[0],
            drug_resistance_effsize=[0],
            S_IE_rate=[0.1],
            I_R_rate=[0.05],
            R_S_rate=[0.01],
            latency_prob=[0],
            E_I_rate=[0],
            I_E_rate=[0],
            E_R_rate=[0],
            surviv_prob=[0.95],
            sample_rate=[0.1],
            recovery_prob_after_sampling=[0.5],
            n_massive_sample=0,
            massive_sample_generation=[],
            massive_sample_prob=[],
            massive_sample_recover_prob=[],
            cap_withinhost=1,
            slim_replicate_seed_file_path="",
            traits_num={"transmissibility": 0, "drug_resistance": 0}
        )
        seed_info = SeedInfo(
            seed_size=1,
            workding_dir=tmp_path,
            seed_host_matching_path=str(seed_host_match),
            use_reference=True
        )
        postprocess_config = Postprocessing(
            do_process=False,
            n_traits={"transmissibility": 0, "drug_resistance": 0},
            branch_color_trait=0,
            heatmap_trait="none",
            vcf=False,
            fasta=False
        )
        
        return SimulationConfig(
            working_dir=tmp_path,
            n_replicates=2,
            network=network_config,
            evolution=evolution_config,
            epidemiology=epidemiology_config,
            seed_info=seed_info,
            genome_config=genome_config,
            postprocess_config=postprocess_config
        )
    
    def test_build_slim_command_without_seed(self, mock_config, tmp_path):
        """Test building SLiM command without seed."""
        runner = SimulationRunner(mock_config)
        
        (tmp_path / "slim.params").write_text("test")
        (tmp_path / "simulation.slim").write_text("test")
        
        cmd = runner._build_slim_command(run_id=1, seed=None)
        
        assert "slim" in cmd
        assert "-d" in cmd
        assert "runid=1" in cmd
        assert "-seed" not in cmd
    
    def test_build_slim_command_with_seed(self, mock_config, tmp_path):
        """Test building SLiM command with seed."""
        runner = SimulationRunner(mock_config)
        
        (tmp_path / "slim.params").write_text("test")
        (tmp_path / "simulation.slim").write_text("test")
        
        cmd = runner._build_slim_command(run_id=1, seed=12345)
        
        assert "slim" in cmd
        assert "-seed" in cmd
        assert "12345" in cmd
    
    def test_get_seed_for_run(self, mock_config, tmp_path):
        """Test getting seed for specific run."""
        seed_file = tmp_path / "seeds.csv"
        seed_df = pd.DataFrame({
            "run_id": [1, 2],
            "random_number_seed": [111, 222]
        })
        seed_df.to_csv(seed_file, index=False)
        
        mock_config.slim_seed_file = seed_file
        runner = SimulationRunner(mock_config)
        
        seed1 = runner._get_seed_for_run(1)
        seed2 = runner._get_seed_for_run(2)
        
        assert seed1 == 111
        assert seed2 == 222
    
    def test_get_seed_for_run_no_file(self, mock_config):
        """Test getting seed when no seed file exists."""
        runner = SimulationRunner(mock_config)
        seed = runner._get_seed_for_run(1)
        assert seed is None
    
    def test_get_seed_for_run_out_of_range(self, mock_config, tmp_path):
        """Test getting seed for run beyond file range."""
        seed_file = tmp_path / "seeds.csv"
        seed_df = pd.DataFrame({
            "run_id": [1],
            "random_number_seed": [111]
        })
        seed_df.to_csv(seed_file, index=False)
        
        mock_config.slim_seed_file = seed_file
        runner = SimulationRunner(mock_config)
        
        seed = runner._get_seed_for_run(5)
        assert seed is None
    
    # @patch('subprocess.run')
    # def test_run_single_simulation_success(self, mock_subprocess, mock_config, tmp_path):
    #     """Test successful single simulation run."""
    #     mock_subprocess.return_value = Mock(returncode=0, stderr=b"")
        
    #     (tmp_path / "slim.params").write_text("test")
    #     (tmp_path / "simulation.slim").write_text("test")
        
    #     runner = SimulationRunner(mock_config)
        
    #     run_dir = tmp_path / "1"
    #     run_dir.mkdir(exist_ok=True)
    #     (run_dir / "sample.csv.gz").write_text("dummy")
        
    #     result = runner._run_single_simulation(1)
        
    #     assert result is True
    #     assert mock_subprocess.called
    
    @patch('subprocess.run')
    def test_run_single_simulation_failure(self, mock_subprocess, mock_config, tmp_path):
        """Test failed single simulation run."""
        mock_subprocess.return_value = Mock(returncode=1, stderr=b"Error message")
        
        (tmp_path / "slim.params").write_text("test")
        (tmp_path / "simulation.slim").write_text("test")
        
        runner = SimulationRunner(mock_config)
        result = runner._run_single_simulation(1)
        
        assert result is False
    
    @patch('subprocess.run')
    def test_run_single_simulation_no_output(self, mock_subprocess, mock_config, tmp_path):
        """Test simulation that completes but produces no output."""
        mock_subprocess.return_value = Mock(returncode=0, stderr=b"")
        
        (tmp_path / "slim.params").write_text("test")
        (tmp_path / "simulation.slim").write_text("test")
        
        runner = SimulationRunner(mock_config)
        result = runner._run_single_simulation(1)
        
        assert result is False


# ========================= Test SimulationOrchestrator =========================

class TestSimulationOrchestrator:
    """Test SimulationOrchestrator class."""
    
    def test_writebinary(self, tmp_path):
        """Test binary string conversion."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{}')
        
        orchestrator = SimulationOrchestrator(config_path)
        
        assert orchestrator._writebinary(True) == "T"
        assert orchestrator._writebinary(False) == ""
        assert orchestrator._writebinary(1) == "T"
        assert orchestrator._writebinary(0) == ""
    
    def test_print_list_no_space(self, tmp_path):
        """Test list printing without spaces."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{}')
        
        orchestrator = SimulationOrchestrator(config_path)
        
        result = orchestrator._print_list_no_space([1, 2, 3])
        assert result == "1,2,3"
        
        result = orchestrator._print_list_no_space([0.1, 0.2, 0.3])
        assert result == "0.1,0.2,0.3"
        
        result = orchestrator._print_list_no_space([])
        assert result == ""
    
    def test_initialize_missing_config(self, tmp_path):
        """Test initialization with missing config file."""
        config_path = tmp_path / "nonexistent.json"
        orchestrator = SimulationOrchestrator(config_path)
        
        with pytest.raises(CustomizedError, match="not found"):
            orchestrator.initialize()


# ========================= Integration Tests =========================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_config_parsing_and_validation(self, tmp_path):
        """Test complete config parsing with all components."""
        network_path = tmp_path / "contact_network.adjlist"
        network_path.write_text("1 2\n2 3\n")
        
        ref_path = tmp_path / "reference.fasta"
        ref_path.write_text(">chr1\nATCGATCG\n")
        
        seed_host_match = tmp_path / "seed_host_match.csv"
        seed_host_match.write_text("seed,host\n1,1\n2,2\n")
        
        causal_gene_path = tmp_path / "causal_gene_info.csv"
        causal_gene_path.write_text("gene,effect\ngene1,0.5\n")
        
        config_data = {
            "BasicRunConfiguration": {
                "cwdir": str(tmp_path),
                "n_replicates": 3
            },
            "NetworkModelParameters": {
                "host_size": 100
            },
            "EvolutionModel": {
                "n_generation": 200,
                "subst_model_parameterization": "mut_rate",
                "mut_rate": 0.0005,
                "within_host_reproduction": True,
                "within_host_reproduction_rate": 0.3,
                "cap_withinhost": 3
            },
            "GenomeElement": {
                "ref_path": str(ref_path),
                "use_genetic_model": True,
                "traits_num": {"transmissibility": 3, "drug_resistance": 2},
                "trait_prob_link": {
                    "link": "logit",
                    "logit": {
                        "alpha_trans": [0.5, 0.4, 0.3],
                        "alpha_drug": [0.6, 0.4]
                    }
                }
            },
            "EpidemiologyModel": {
                "model": "SEIR",
                "epoch_changing": {
                    "n_epoch": 2,
                    "epoch_changing_generation": [100]
                },
                "super_infection": True,
                "genetic_architecture": {
                    "transmissibility": [2, 3],
                    "drug_resistance": [1, 2]
                },
                "transition_prob": {
                    "S_IE_prob": [0.1, 0.15],
                    "I_R_prob": [0.05, 0.06],
                    "R_S_prob": [0.01, 0.02],
                    "latency_prob": [0.3, 0.4],
                    "E_I_prob": [0.2, 0.25],
                    "I_E_prob": [0.0, 0.0],
                    "E_R_prob": [0.01, 0.02],
                    "surviv_prob": [0.95, 0.93],
                    "sample_prob": [0.1, 0.15],
                    "recovery_prob_after_sampling": [0.5, 0.6]
                },
                "massive_sampling": {
                    "event_num": 1,
                    "generation": [150],
                    "sampling_prob": [0.8],
                    "recovery_prob_after_sampling": [0.7]
                },
                "slim_replicate_seed_file_path": None
            },
            "SeedsConfiguration": {
                "seed_size": 2,
                "use_reference": True
            },
            "Postprocessing_options": {
                "do_postprocess": True,
                "tree_plotting": {
                    "branch_color_trait": 1,
                    "heatmap": "transmissibility"
                },
                "sequence_output": {
                    "vcf": True,
                    "fasta": True
                }
            }
        }
        
        config_file = tmp_path / "full_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        parser = ConfigParser()
        config = parser.parse_config_file(config_file)
        
        assert config.n_replicates == 3
        assert config.network.host_size == 100
        assert config.evolution.n_generation == 200
        assert config.evolution.within_host_reproduction is True
        assert config.evolution.cap_withinhost == 3
        assert config.epidemiology.model == EpiModel.SEIR
        assert config.epidemiology.n_epoch == 2
        assert config.epidemiology.super_infection is True
        assert len(config.epidemiology.epoch_changing_generation) == 1
        assert config.epidemiology.n_massive_sample == 1
        assert config.genome_config.use_genetic_model is True
        assert config.seed_info.seed_size == 2
        assert config.postprocess_config.do_process is True
        assert config.postprocess_config.vcf is True


if __name__ == "__main__":
    # Run tests with: python -m pytest test_outbreak_simulator.py -v
    pytest.main([__file__, "-v"])