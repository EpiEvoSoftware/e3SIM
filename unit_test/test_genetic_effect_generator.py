import pytest
import numpy as np
import pandas as pd
import os, sys
import tempfile
import shutil
from unittest.mock import Mock, patch, mock_open
from io import StringIO

curr_dir = os.path.dirname(__file__)
e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
if e3SIM_dir not in sys.path:
    sys.path.insert(0, e3SIM_dir)

from error_handling import CustomizedError

# Import the classes and functions to test
from genetic_effect_generator import (
    GeneticEffectConfig,
    EffectGenerator,
    effsize_generation_byconfig,
    DEFAULT_R_OHR,
    DEFAULT_R_CLR,
    DEFAULT_VTGT,
    DEFAULT_MU,
    EXP_BETAPRIOR
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def basic_trait_num():
    """Basic trait number configuration."""
    return {"transmissibility": 2, "drug_resistance": 1}


@pytest.fixture
def valid_config_randomly_generate(temp_dir, basic_trait_num):
    """Valid configuration for randomly_generate method."""
    csv_path = os.path.join(temp_dir, "candidates.csv")
    # Create a mock CSV file
    with open(csv_path, 'w') as f:
        f.write("start,end,trait_0,trait_1,trait_2\n")
        f.write("100,200,1,0,1\n")
        f.write("300,400,0,1,1\n")
    
    return GeneticEffectConfig(
        method="randomly_generate",
        wk_dir=temp_dir,
        num_init_seq=5,
        calibration=True,
        trait_num=basic_trait_num,
        random_seed=42,
        csv=csv_path,
        func="n",
        site_frac=[0.5, 0.3, 0.4],
        site_disp=100,
        taus=[0.1, 0.2, 0.15],
        var_target=[1.0, 1.0, 1.0],
        calibration_link=False,
        Rs=[],
        link="logit"
    )


@pytest.fixture
def valid_config_user_input(temp_dir, basic_trait_num):
    """Valid configuration for user_input method."""
    csv_path = os.path.join(temp_dir, "effects.csv")
    # Create effect size CSV
    with open(csv_path, 'w') as f:
        f.write("Sites,trait_0,trait_1,trait_2\n")
        f.write("150,0.5,0.0,0.3\n")
        f.write("350,0.0,0.4,0.2\n")
    
    return GeneticEffectConfig(
        method="user_input",
        wk_dir=temp_dir,
        num_init_seq=3,
        calibration=False,
        trait_num=basic_trait_num,
        random_seed=42,
        csv=csv_path
    )


# ============================================================================
# GeneticEffectConfig Tests
# ============================================================================

class TestGeneticEffectConfig:
    """Test suite for GeneticEffectConfig validation."""
    
    def test_valid_config_randomly_generate(self, valid_config_randomly_generate):
        """Test that valid randomly_generate config passes validation."""
        valid_config_randomly_generate.validate()  # Should not raise
    
    def test_valid_config_user_input(self, valid_config_user_input):
        """Test that valid user_input config passes validation."""
        valid_config_user_input.validate()  # Should not raise
    
    def test_invalid_working_directory(self, basic_trait_num):
        """Test validation fails for non-existent working directory."""
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir="/non/existent/path",
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42
        )
        with pytest.raises(CustomizedError, match="does not exist"):
            config.validate()
    
    def test_invalid_method(self, temp_dir, basic_trait_num):
        """Test validation fails for invalid method."""
        config = GeneticEffectConfig(
            method="invalid_method",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42
        )
        with pytest.raises(CustomizedError, match="isn't a valid method"):
            config.validate()
    
    def test_negative_seed_size(self, temp_dir, basic_trait_num):
        """Test validation fails for negative seed size."""
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=-1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42
        )
        with pytest.raises(CustomizedError, match="must be positive"):
            config.validate()
    
    def test_invalid_trait_num_keys(self, temp_dir):
        """Test validation fails when trait_num doesn't have exactly 2 keys."""
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num={"transmissibility": 1},
            random_seed=42
        )
        with pytest.raises(CustomizedError, match="exactly 2 kinds of traits'"):
            config.validate()
    
    def test_zero_total_traits(self, temp_dir):
        """Test validation fails when total traits is less than 1."""
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num={"transmissibility": 0, "drug_resistance": 0},
            random_seed=42
        )
        with pytest.raises(CustomizedError, match="at least 1"):
            config.validate()
    
    def test_site_frac_length_mismatch(self, temp_dir, basic_trait_num):
        """Test validation fails when site_frac length doesn't match trait count."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            func="n",
            site_frac=[0.1, 0.2],  # Should be 3
            taus=[0.1, 0.2, 0.3]
        )
        with pytest.raises(CustomizedError, match="same length"):
            config.validate()
    
    def test_site_frac_out_of_range(self, temp_dir, basic_trait_num):
        """Test validation fails when site_frac values are outside [0, 1]."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            func="n",
            site_frac=[0.5, 1.5, 0.3],  # 1.5 is invalid
            taus=[0.1, 0.2, 0.3]
        )
        with pytest.raises(CustomizedError, match=r"within \(0, 1\)"):
            config.validate()
    
    def test_negative_site_disp(self, temp_dir, basic_trait_num):
        """Test validation fails for negative site_disp."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            func="n",
            site_frac=[0.5, 0.3, 0.4],
            site_disp=-10,
            taus=[0.1, 0.2, 0.3]
        )
        with pytest.raises(CustomizedError, match="has to be positive"):
            config.validate()
    
    def test_invalid_func(self, temp_dir, basic_trait_num):
        """Test validation fails for invalid func parameter."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            site_frac=[0.5, 0.3, 0.4],
            site_disp=10,
            random_seed=42,
            func="invalid"
        )
        with pytest.raises(CustomizedError, match="isn't a valid method"):
            config.validate()
    
    def test_taus_length_mismatch(self, temp_dir, basic_trait_num):
        """Test validation fails when taus length doesn't match trait count for func='n'."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            func="n",
            site_frac=[0.5, 0.3, 0.4],
            site_disp=10,
            taus=[0.1, 0.2]  # Should be 3
        )
        with pytest.raises(CustomizedError, match="do not match"):
            config.validate()
    
    def test_calibration_link_invalid_link_type(self, temp_dir, basic_trait_num):
        """Test validation fails for invalid link type with calibration_link."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            func="n",
            taus=[0.1, 0.2, 0.3],
            site_frac=[0.5, 0.3, 0.4],
            site_disp=10,
            calibration_link=True,
            link="invalid_link"
        )
        with pytest.raises(CustomizedError, match="needs to be either"):
            config.validate()
    
    def test_calibration_link_default_Rs_logit(self, temp_dir, basic_trait_num):
        """Test that default Rs are set correctly for logit link."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            func="n",
            taus=[0.1, 0.2, 0.3],
            calibration_link=True,
            link="logit",
            Rs=[],
            site_frac=[0.5, 0.3, 0.4],
            site_disp=10,
        )
        config.validate()
        assert len(config.params["Rs"]) == 3
        assert all(r == DEFAULT_R_OHR for r in config.params["Rs"])
    
    def test_calibration_link_default_Rs_cloglog(self, temp_dir, basic_trait_num):
        """Test that default Rs are set correctly for cloglog link."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            func="n",
            taus=[0.1, 0.2, 0.3],
            calibration_link=True,
            link="cloglog",
            Rs=[],
            site_frac=[0.5, 0.3, 0.4],
            site_disp=10,
        )
        config.validate()
        assert len(config.params["Rs"]) == 3
        # First 2 should be DEFAULT_R_OHR, last 1 should be DEFAULT_R_CLR
        assert all(config.params["Rs"][:2] == DEFAULT_R_OHR)
        assert config.params["Rs"][2] == DEFAULT_R_CLR


# ============================================================================
# EffectGenerator Tests
# ============================================================================

class TestEffectGenerator:
    """Test suite for EffectGenerator class."""
    
    def test_initialization(self, valid_config_randomly_generate):
        """Test EffectGenerator initialization."""
        generator = EffectGenerator(valid_config_randomly_generate)
        assert generator.cfg == valid_config_randomly_generate
    
    def test_pointnormal_distribution(self, valid_config_randomly_generate):
        """Test that _pointnormal generates values with correct distribution."""
        generator = EffectGenerator(valid_config_randomly_generate)
        samples = generator._pointnormal(n=1000, tau=1.0)
        
        assert len(samples) == 1000
        # Check that mean is close to 0 and std is close to tau
        assert abs(np.mean(samples)) < 0.1
        assert abs(np.std(samples) - 1.0) < 0.1
    
    def test_laplace_distribution(self, valid_config_randomly_generate):
        """Test that _laplace generates values with correct distribution."""
        generator = EffectGenerator(valid_config_randomly_generate)
        samples = generator._laplace(n=1000, b=1.0)
        
        assert len(samples) == 1000
        # Check that mean is close to 0
        assert abs(np.mean(samples)) < 0.1
    
    def test_studentst_distribution(self, valid_config_randomly_generate):
        """Test that _studentst generates values with correct distribution."""
        generator = EffectGenerator(valid_config_randomly_generate)
        samples = generator._studentst(n=1000, scale=1.0, nv=3)
        
        assert len(samples) == 1000
        # Check that mean is close to 0
        assert abs(np.mean(samples)) < 0.2
    
    def test_read_candregion_csv(self, valid_config_randomly_generate):
        """Test reading candidate region CSV."""
        generator = EffectGenerator(valid_config_randomly_generate)
        candidates = generator._read_candregion_csv()
        
        assert isinstance(candidates, dict)
        assert len(candidates) == 3
        # Check that sites are correctly extracted
        assert 0 in candidates
        assert 1 in candidates
        assert 2 in candidates
    
    def test_read_candregion_csv_nonexistent(self, temp_dir, basic_trait_num):
        """Test error handling when CSV doesn't exist."""
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv="/non/existent/file.csv",
            func="n",
            taus=[0.1, 0.2, 0.3]
        )
        generator = EffectGenerator(config)
        
        with pytest.raises(CustomizedError, match="does not exist"):
            generator._read_candregion_csv()
    
    def test_read_effsize_csv(self, valid_config_user_input):
        """Test reading effect size CSV."""
        generator = EffectGenerator(valid_config_user_input)
        df = generator._read_effsize_csv()
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 4  # Sites + 3 traits
        assert "Sites" in df.columns
        assert all(f"trait_{i}" in df.columns for i in range(3))
    
    def test_select_sites_basic(self, valid_config_randomly_generate):
        """Test site selection from candidates."""
        generator = EffectGenerator(valid_config_randomly_generate)
        candidates = {0: [100, 101, 102], 1: [200, 201, 202], 2: [300, 301, 302]}
        
        df_sites = generator._select_sites(candidates, frac=[1.0, 1.0, 1.0], dispersion=100)
        
        assert isinstance(df_sites, pd.DataFrame)
        assert "Sites" in df_sites.columns
        assert all(f"trait_{i}" in df_sites.columns for i in range(3))
        # With frac=1.0, we should get sites for each trait
        assert df_sites.shape[0] > 0
    
    def test_select_sites_no_overlap(self, valid_config_randomly_generate):
        """Test that site selection prevents overlap between traits."""
        generator = EffectGenerator(valid_config_randomly_generate)
        candidates = {0: [100, 101, 102], 1: [100, 101, 102], 2: [100, 101, 102]}
        
        df_sites = generator._select_sites(candidates, frac=[0.5, 0.5, 0.5], dispersion=100)
        
        # Check that each site is assigned to at most one trait per row
        for _, row in df_sites.iterrows():
            trait_sum = sum(row[f"trait_{i}"] for i in range(3))
            assert trait_sum <= 3  # At most one per trait
    
    def test_sample_univariate_pointnormal(self, valid_config_randomly_generate):
        """Test univariate sampling with point normal."""
        generator = EffectGenerator(valid_config_randomly_generate)
        df_id = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [1, 1, 0],
            "trait_1": [0, 1, 1],
            "trait_2": [1, 0, 1]
        })
        
        df_eff = generator._sample_univariate(df_id)
        
        # Check that zeros remain zero and non-zeros are replaced
        assert df_eff.loc[2, "trait_0"] == 0
        assert df_eff.loc[0, "trait_1"] == 0
        assert df_eff.loc[1, "trait_2"] == 0
        
        # Non-zero values should be floats
        assert isinstance(df_eff.loc[0, "trait_0"], float)
        assert df_eff.loc[0, "trait_0"] != 0
    
    def test_rename_columns(self, valid_config_randomly_generate):
        """Test column renaming based on trait categories."""
        generator = EffectGenerator(valid_config_randomly_generate)
        df = pd.DataFrame({
            "Sites": [100, 101],
            "trait_0": [0.1, 0.2],
            "trait_1": [0.3, 0.4],
            "trait_2": [0.5, 0.6]
        })
        
        df_renamed = generator._rename_columns(df)
        
        assert "transmissibility_1" in df_renamed.columns
        assert "transmissibility_2" in df_renamed.columns
        assert "drug_resistance_1" in df_renamed.columns
    
    def test_compute_seed_traits_no_vcf(self, valid_config_randomly_generate):
        """Test seed trait computation when no VCF directory exists."""
        generator = EffectGenerator(valid_config_randomly_generate)
        df_eff = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [0.1, 0.2, 0.3],
            "trait_1": [0.4, 0.5, 0.6],
            "trait_2": [0.7, 0.8, 0.9]
        })
        
        seeds, _ = generator._compute_seed_traits(df_eff)
        
        # Should return all zeros when no VCF
        assert seeds.shape[0] == valid_config_randomly_generate.num_init_seq
        assert all(seeds.iloc[:, 1:].values.flatten() == 0)
    
    def test_variance_calc(self, valid_config_randomly_generate):
        """Test variance calculation."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_eff = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [0.1, 0.2, 0.3],
            "trait_1": [0.4, 0.5, 0.6],
            "trait_2": [0.7, 0.8, 0.9]
        })
        
        seeds_state = pd.DataFrame({
            "Sites": [100, 101, 102],
            "seed_0": [0, 1, 0],
            "seed_1": [1, 0, 1],
            "seed_2": [0, 1, 1]
        })
        
        var_emp = generator._variance_calc(df_eff, seeds_state)
        
        assert isinstance(var_emp, pd.Series)
        assert len(var_emp) == 3
        assert all(var_emp >= 0)
    
    def test_calibrate(self, valid_config_randomly_generate):
        """Test effect size calibration."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_eff = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [0.1, 0.2, 0.3],
            "trait_1": [0.4, 0.5, 0.6],
            "trait_2": [0.7, 0.8, 0.9]
        })
        
        seeds_state = pd.DataFrame({
            "Sites": [100, 101, 102],
            "seed_0": [0, 1, 0],
            "seed_1": [1, 0, 1],
            "seed_2": [0, 1, 1],
            "seed_3": [1, 1, 0],
            "seed_4": [0, 0, 1]
        })
        
        df_calibrated, var_emp = generator._calibrate(df_eff, seeds_state)
        
        assert isinstance(df_calibrated, pd.DataFrame)
        assert df_calibrated.shape == df_eff.shape
        assert isinstance(var_emp, pd.Series)
    
    def test_calibrate_linkslope_logit(self, valid_config_randomly_generate):
        """Test link slope calibration with logit link."""
        valid_config_randomly_generate.params["calibration_link"] = True
        valid_config_randomly_generate.params["link"] = "logit"
        valid_config_randomly_generate.params["Rs"] = [1.5, 1.5, 1.5]
        
        generator = EffectGenerator(valid_config_randomly_generate)
        
        var_em = np.array([1.0, 0.5, 0.8])
        
        result = generator._calibrate_linkslope(
            Rs=np.array([1.5, 1.5, 1.5]),
            link_type="logit",
            var_em=var_em,
            trait_num=valid_config_randomly_generate.trait_num
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert "logit" in result
    
    def test_calibrate_linkslope_cloglog(self, valid_config_randomly_generate):
        """Test link slope calibration with cloglog link."""
        valid_config_randomly_generate.params["calibration_link"] = True
        valid_config_randomly_generate.params["link"] = "cloglog"
        valid_config_randomly_generate.params["Rs"] = [1.5, 1.5, 0.667]
        
        generator = EffectGenerator(valid_config_randomly_generate)
        
        var_em = np.array([1.0, 0.5, 0.8])
        
        result = generator._calibrate_linkslope(
            Rs=np.array([1.5, 1.5, 0.667]),
            link_type="cloglog",
            var_em=var_em,
            trait_num=valid_config_randomly_generate.trait_num
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert "cloglog" in result
    
    def test_write_outputs(self, valid_config_randomly_generate):
        """Test writing output files."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_eff = pd.DataFrame({
            "Sites": [100, 101],
            "transmissibility_1": [0.1, 0.2],
            "transmissibility_2": [0.3, 0.4],
            "drug_resistance_1": [0.5, 0.6]
        })
        
        seeds = pd.DataFrame({
            "Seed_ID": [0, 1],
            "transmissibility_1": [0.5, 0.6],
            "transmissibility_2": [0.7, 0.8],
            "drug_resistance_1": [0.9, 1.0]
        })
        
        generator._write_outputs(df_eff, seeds)
        
        # Check that files were created
        assert os.path.exists(os.path.join(valid_config_randomly_generate.wk_dir, "causal_gene_info.csv"))
        assert os.path.exists(os.path.join(valid_config_randomly_generate.wk_dir, "seeds_trait_values.csv"))
        
        # Verify content
        df_read = pd.read_csv(os.path.join(valid_config_randomly_generate.wk_dir, "causal_gene_info.csv"))
        assert df_read.shape == df_eff.shape


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_run_user_input_method(self, valid_config_user_input):
        """Test running the full pipeline with user_input method."""
        generator = EffectGenerator(valid_config_user_input)
        error = generator.run()
        
        assert error is None
        # Check output files exist
        assert os.path.exists(os.path.join(valid_config_user_input.wk_dir, "causal_gene_info.csv"))
        assert os.path.exists(os.path.join(valid_config_user_input.wk_dir, "seeds_trait_values.csv"))
    
    def test_run_randomly_generate_method(self, valid_config_randomly_generate):
        """Test running the full pipeline with randomly_generate method."""
        generator = EffectGenerator(valid_config_randomly_generate)
        error = generator.run()
        
        assert error is None
        # Check output files exist
        assert os.path.exists(os.path.join(valid_config_randomly_generate.wk_dir, "causal_gene_info.csv"))
        assert os.path.exists(os.path.join(valid_config_randomly_generate.wk_dir, "seeds_trait_values.csv"))


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_candidate_csv(self, temp_dir, basic_trait_num):
        """Test handling of empty candidate CSV."""
        csv_path = os.path.join(temp_dir, "empty.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        candidates = generator._read_candregion_csv()
        
        # Should return empty lists for each trait
        assert all(len(candidates[i]) == 0 for i in range(3))
    
    def test_swapped_start_end_in_csv(self, temp_dir, basic_trait_num):
        """Test handling when end < start in CSV."""
        csv_path = os.path.join(temp_dir, "swapped.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("200,100,1,0,1\n")  # Swapped
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        # Should handle swapped values gracefully
        candidates = generator._read_candregion_csv()
        assert len(candidates[0]) > 0
    
    def test_zero_variance_calibration(self, valid_config_randomly_generate):
        """Test calibration with zero variance traits."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        # Create data with zero variance for trait_2
        df_eff = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [0.1, 0.2, 0.3],
            "trait_1": [0.4, 0.5, 0.6],
            "trait_2": [0.0, 0.0, 0.0]
        })
        
        seeds_state = pd.DataFrame({
            "Sites": [100, 101, 102],
            "seed_0": [0, 1, 0],
            "seed_1": [1, 0, 1],
            "seed_2": [0, 1, 1]
        })
        
        # Should handle zero variance gracefully
        df_calibrated, var_emp = generator._calibrate(df_eff, seeds_state)
        assert df_calibrated is not None


# ============================================================================
# Parametrized Tests
# ============================================================================

class TestParametrized:
    """Parametrized tests for various distributions and configurations."""
    
    @pytest.mark.parametrize("func,params", [
        ("n", {"taus": [0.1, 0.2, 0.3]}),
        ("l", {"bs": [0.1, 0.2, 0.3]}),
        ("st", {"s": [0.1, 0.2, 0.3], "nv": 3})
    ])
    def test_sample_different_functions(self, temp_dir, basic_trait_num, func, params):
        """Test sampling with different distribution functions."""
        csv_path = os.path.join(temp_dir, "candidates.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("100,200,1,0,1\n")
            f.write("300,400,0,1,1\n")
        
        config_params = {
            "method": "randomly_generate",
            "wk_dir": temp_dir,
            "num_init_seq": 3,
            "calibration": False,
            "trait_num": basic_trait_num,
            "random_seed": 42,
            "csv": csv_path,
            "func": func,
            "site_frac": [0.5, 0.5, 0.5]
        }
        config_params.update(params)
        
        config = GeneticEffectConfig(**config_params)
        generator = EffectGenerator(config)
        
        df_id = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [1, 1, 0],
            "trait_1": [0, 1, 1],
            "trait_2": [1, 0, 1]
        })
        
        df_eff = generator._sample_univariate(df_id)
        
        # Check that sampling occurred correctly
        assert df_eff.loc[0, "trait_0"] != 0
        assert df_eff.loc[1, "trait_0"] != 0
        assert df_eff.loc[2, "trait_0"] == 0
    
    @pytest.mark.parametrize("link_type", ["logit", "cloglog"])
    def test_calibrate_linkslope_different_links(self, valid_config_randomly_generate, link_type):
        """Test link slope calibration with different link types."""
        valid_config_randomly_generate.params["link"] = link_type
        
        generator = EffectGenerator(valid_config_randomly_generate)
        
        var_em = np.array([1.0, 0.5, 0.8])
        Rs = np.array([1.5, 1.5, 0.667]) if link_type == "cloglog" else np.array([1.5, 1.5, 1.5])
        
        result = generator._calibrate_linkslope(
            Rs=Rs,
            link_type=link_type,
            var_em=var_em,
            trait_num=valid_config_randomly_generate.trait_num
        )
        
        assert result is not None
        assert link_type in result
    
    @pytest.mark.parametrize("n_seeds", [1, 5, 10])
    def test_different_seed_numbers(self, temp_dir, basic_trait_num, n_seeds):
        """Test with different numbers of seeds."""
        csv_path = os.path.join(temp_dir, "effects.csv")
        with open(csv_path, 'w') as f:
            f.write("Sites,trait_0,trait_1,trait_2\n")
            f.write("150,0.5,0.0,0.3\n")
        
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=n_seeds,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path
        )
        
        generator = EffectGenerator(config)
        df_eff = generator._read_effsize_csv()
        seeds, seeds_state = generator._compute_seed_traits(df_eff)
        
        assert seeds.shape[0] == n_seeds
        assert seeds_state.shape[1] == n_seeds + 1  # +1 for Sites column
    


# ============================================================================
# Mock and Patch Tests
# ============================================================================

class TestMockingAndPatching:
    """Tests using mocking and patching for external dependencies."""
    
    def test_compute_seed_traits_with_vcf(self, valid_config_randomly_generate):
        """Test seed trait computation with mocked VCF files."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_eff = pd.DataFrame({
            "Sites": [100, 150, 200],
            "trait_0": [0.1, 0.2, 0.3],
            "trait_1": [0.4, 0.5, 0.6],
            "trait_2": [0.7, 0.8, 0.9]
        })
        
        # Create mock VCF directory and files
        vcf_dir = os.path.join(valid_config_randomly_generate.wk_dir, "originalvcfs")
        os.makedirs(vcf_dir, exist_ok=True)
        
        # Create mock VCF files
        for i in range(valid_config_randomly_generate.num_init_seq):
            vcf_path = os.path.join(vcf_dir, f"seed_{i}.vcf")
            with open(vcf_path, 'w') as f:
                f.write("##fileformat=VCFv4.2\n")
                f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n")
                f.write(f"chr1\t150\t.\tA\tG\t100\tPASS\t.\tGT\t1/1\n")
        
        seeds, _ = generator._compute_seed_traits(df_eff) # seed X trait, site X seed
        
        # All seeds should have the same trait value from position 150
        assert seeds.shape[0] == valid_config_randomly_generate.num_init_seq
        assert all(seeds["trait_0"] == df_eff.loc[df_eff["Sites"] == 150, "trait_0"].values[0])
    
    @patch('builtins.print')
    def test_warning_messages(self, mock_print, temp_dir, basic_trait_num):
        """Test that warning messages are printed correctly."""
        csv_path = os.path.join(temp_dir, "empty.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        generator._read_candregion_csv()
        
        # Check that warning was printed
        warning_calls = [call for call in mock_print.call_args_list 
                        if len(call[0]) > 0 and "WARNING" in str(call[0][0])]
        assert len(warning_calls) > 0


# ============================================================================
# Numerical Accuracy Tests
# ============================================================================

class TestNumericalAccuracy:
    """Tests for numerical accuracy and precision."""
    
    def test_variance_calculation_accuracy(self, valid_config_randomly_generate):
        """Test that variance calculation is numerically accurate."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        # Create known data with calculable variance
        df_eff = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [1.0, 2.0, 3.0],
            "trait_1": [0.5, 0.5, 0.5],
            "trait_2": [1.0, -1.0, 0.0]
        })
        
        seeds_state = pd.DataFrame({
            "Sites": [100, 101, 102],
            "seed_0": [1, 0, 0],
            "seed_1": [0, 1, 0],
            "seed_2": [0, 0, 1],
            "seed_3": [1, 1, 0],
            "seed_4": [1, 0, 1]
        })
        
        var_emp = generator._variance_calc(df_eff, seeds_state)
        
        # All variances should be non-negative
        assert all(var_emp >= 0)
        
        # trait_1 has all same values, so contribution is only from AF variation
        # Should have lower variance than trait_0 which has varying effects
        assert var_emp["trait_1"] <= var_emp["trait_0"]
    
    def test_calibration_preserves_direction(self, valid_config_randomly_generate):
        """Test that calibration preserves the sign of effect sizes."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_eff = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [0.1, -0.2, 0.3],
            "trait_1": [-0.4, 0.5, -0.6],
            "trait_2": [0.7, 0.8, -0.9]
        })
        
        seeds_state = pd.DataFrame({
            "Sites": [100, 101, 102],
            "seed_0": [1, 0, 1],
            "seed_1": [0, 1, 0],
            "seed_2": [1, 1, 1],
            "seed_3": [0, 0, 1],
            "seed_4": [1, 0, 0]
        })
        
        df_calibrated, _ = generator._calibrate(df_eff, seeds_state)
        
        # Check that signs are preserved
        for col in ["trait_0", "trait_1", "trait_2"]:
            original_signs = np.sign(df_eff[col])
            calibrated_signs = np.sign(df_calibrated[col])
            assert all(original_signs == calibrated_signs)
    
    def test_effect_sizes_reasonable_range(self, valid_config_randomly_generate):
        """Test that generated effect sizes are in reasonable range."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        # Generate many samples
        samples_n = generator._pointnormal(n=1000, tau=0.1)
        samples_l = generator._laplace(n=1000, b=0.1)
        samples_st = generator._studentst(n=1000, scale=0.1, nv=3)
        
        # Most values should be within reasonable range (e.g., 3 standard deviations)
        assert np.percentile(np.abs(samples_n), 95) < 0.3
        assert np.percentile(np.abs(samples_l), 95) < 0.5
        # Student's t has heavier tails, so more lenient
        assert np.percentile(np.abs(samples_st), 95) < 1.0


# ============================================================================
# File I/O Tests
# ============================================================================

class TestFileIO:
    """Tests for file input/output operations."""
    
    def test_write_and_read_outputs(self, valid_config_randomly_generate):
        """Test that written files can be read back correctly."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_eff = pd.DataFrame({
            "Sites": [100, 101, 102],
            "transmissibility_1": [0.1, 0.2, 0.3],
            "transmissibility_2": [0.4, 0.5, 0.6],
            "drug_resistance_1": [0.7, 0.8, 0.9]
        })
        
        seeds = pd.DataFrame({
            "Seed_ID": [0, 1, 2],
            "transmissibility_1": [0.5, 0.6, 0.7],
            "transmissibility_2": [0.8, 0.9, 1.0],
            "drug_resistance_1": [1.1, 1.2, 1.3]
        })
        
        generator._write_outputs(df_eff, seeds)
        
        # Read back and verify
        df_eff_read = pd.read_csv(os.path.join(valid_config_randomly_generate.wk_dir, "causal_gene_info.csv"))
        seeds_read = pd.read_csv(os.path.join(valid_config_randomly_generate.wk_dir, "seeds_trait_values.csv"))
        
        pd.testing.assert_frame_equal(df_eff, df_eff_read)
        pd.testing.assert_frame_equal(seeds, seeds_read)
    
    def test_csv_with_different_encodings(self, temp_dir, basic_trait_num):
        """Test reading CSV files with different encodings."""
        csv_path = os.path.join(temp_dir, "effects_utf8.csv")
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Sites,trait_0,trait_1,trait_2\n")
            f.write("150,0.5,0.0,0.3\n")
        
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path
        )
        
        generator = EffectGenerator(config)
        df = generator._read_effsize_csv()
        
        assert df.shape[0] == 1
        assert df.shape[1] == 4
    
    def test_malformed_csv_handling(self, temp_dir, basic_trait_num):
        """Test handling of malformed CSV files."""
        csv_path = os.path.join(temp_dir, "malformed.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("not_a_number,200,1,0,1\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=1,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        
        # Should raise an error due to non-numeric values
        with pytest.raises(Exception):
            generator._read_candregion_csv()


# ============================================================================
# Random Seed Reproducibility Tests
# ============================================================================

class TestReproducibility:
    """Tests for random seed reproducibility."""
    
    def test_same_seed_same_results(self, temp_dir, basic_trait_num):
        """Test that same random seed produces same results."""
        csv_path = os.path.join(temp_dir, "candidates.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("100,200,1,1,1\n")
        
        config1 = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=12345,
            csv=csv_path,
            func="n",
            site_frac=[0.5, 0.5, 0.5],
            taus=[0.1, 0.2, 0.3]
        )
        
        config2 = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=12345,
            csv=csv_path,
            func="n",
            site_frac=[0.5, 0.5, 0.5],
            taus=[0.1, 0.2, 0.3]
        )
        
        gen1 = EffectGenerator(config1)
        gen2 = EffectGenerator(config2)
        
        samples1 = gen1._pointnormal(n=100, tau=0.1)
        # Re-seed for second generator
        np.random.seed(12345)
        samples2 = gen2._pointnormal(n=100, tau=0.1)
        
        np.testing.assert_array_equal(samples1, samples2)
    
    def test_different_seed_different_results(self, temp_dir, basic_trait_num):
        """Test that different random seeds produce different results."""
        csv_path = os.path.join(temp_dir, "candidates.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("100,200,1,1,1\n")
        
        config1 = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=12345,
            csv=csv_path,
            func="n",
            site_frac=[0.5, 0.5, 0.5],
            taus=[0.1, 0.2, 0.3]
        )
        
        config2 = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=54321,
            csv=csv_path,
            func="n",
            site_frac=[0.5, 0.5, 0.5],
            taus=[0.1, 0.2, 0.3]
        )
        
        gen1 = EffectGenerator(config1)
        gen2 = EffectGenerator(config2)
        
        samples1 = gen1._pointnormal(n=100, tau=0.1)
        samples2 = gen2._pointnormal(n=100, tau=0.1)
        
        # Results should be different
        assert not np.array_equal(samples1, samples2)

# ============================================================================
# Performance and Stress Tests
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    def test_large_number_of_sites(self, temp_dir, basic_trait_num):
        """Test handling of large number of sites."""
        csv_path = os.path.join(temp_dir, "large.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            # Large genomic region
            f.write("1,10000,1,1,1\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=10,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            site_frac=[0.01, 0.01, 0.01],
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        candidates = generator._read_candregion_csv()
        
        # Should handle 10000 sites
        assert all(len(candidates[i]) == 10000 for i in range(3))
    
    def test_many_traits(self, temp_dir):
        """Test with many traits."""
        trait_num = {"transmissibility": 5, "drug_resistance": 5}
        
        csv_path = os.path.join(temp_dir, "many_traits.csv")
        with open(csv_path, 'w') as f:
            header = "start,end," + ",".join([f"trait_{i}" for i in range(10)])
            f.write(header + "\n")
            f.write("100,200," + ",".join(["1"] * 10) + "\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            site_frac=[0.5] * 10,
            taus=[0.1] * 10
        )
        
        generator = EffectGenerator(config)
        candidates = generator._read_candregion_csv()
        
        assert len(candidates) == 10


# ============================================================================
# Boundary Value Tests
# ============================================================================

class TestBoundaryValues:
    """Tests for boundary values and edge conditions."""
    
    def test_site_frac_zero(self, temp_dir, basic_trait_num):
        """Test with site_frac = 0."""
        csv_path = os.path.join(temp_dir, "candidates.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("100,200,1,1,1\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            site_frac=[0.0, 0.0, 0.0],
            taus=[0.1, 0.2, 0.3]
        )
        
        # config.validate()  # Should not raise
        with pytest.raises(CustomizedError, match = r"within \(0, 1\)"):
            config.validate()

    def test_site_frac_one(self, temp_dir, basic_trait_num):
        """Test with site_frac = 1.0."""
        csv_path = os.path.join(temp_dir, "candidates.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("100,105,1,1,1\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            site_frac=[1.0, 1.0, 1.0],
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        candidates = generator._read_candregion_csv()
        df_sites = generator._select_sites(candidates, frac=[1.0, 1.0, 1.0], dispersion=100)
        
        # With frac=1.0, should select sites for each trait
        assert df_sites.shape[0] > 0
    
    def test_very_small_tau(self, valid_config_randomly_generate):
        """Test with very small tau values."""
        generator = EffectGenerator(valid_config_randomly_generate)
        samples = generator._pointnormal(n=1000, tau=0.001)
        
        # Variance should be very small
        assert np.var(samples) < 0.01
    
    def test_very_large_tau(self, valid_config_randomly_generate):
        """Test with very large tau values."""
        generator = EffectGenerator(valid_config_randomly_generate)
        samples = generator._pointnormal(n=1000, tau=10.0)
        
        # Variance should be large
        assert np.var(samples) > 50
    
    def test_single_site(self, temp_dir, basic_trait_num):
        """Test with only one site."""
        csv_path = os.path.join(temp_dir, "single.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("100,100,1,0,1\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            site_frac=[1.0, 1.0, 1.0],
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        candidates = generator._read_candregion_csv()
        
        # Should have exactly 1 site for trait_0 and trait_2
        assert len(candidates[0]) == 1
        assert len(candidates[1]) == 0
        assert len(candidates[2]) == 1


# ============================================================================
# Data Type and Structure Tests
# ============================================================================

class TestDataTypes:
    """Tests for data types and structures."""
    
    def test_effect_dataframe_structure(self, valid_config_user_input):
        """Test that effect DataFrame has correct structure."""
        generator = EffectGenerator(valid_config_user_input)
        df = generator._read_effsize_csv()
        
        # Check column types
        assert pd.api.types.is_integer_dtype(df["Sites"]) or pd.api.types.is_float_dtype(df["Sites"])
        for col in df.columns[1:]:
            assert pd.api.types.is_numeric_dtype(df[col])
    
    def test_seeds_dataframe_structure(self, valid_config_randomly_generate):
        """Test that seeds DataFrame has correct structure."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_eff = pd.DataFrame({
            "Sites": [100, 101],
            "trait_0": [0.1, 0.2],
            "trait_1": [0.3, 0.4],
            "trait_2": [0.5, 0.6]
        })
        
        seeds, _ = generator._compute_seed_traits(df_eff)
        
        # Check structure
        assert "Seed_ID" in seeds.columns
        assert seeds.shape[1] == 4  # Seed_ID + 3 traits
        assert all(pd.api.types.is_numeric_dtype(seeds[col]) for col in seeds.columns[1:])
    
    def test_numpy_array_conversion(self, valid_config_randomly_generate):
        """Test conversion between pandas and numpy."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_eff = pd.DataFrame({
            "Sites": [100, 101, 102],
            "trait_0": [0.1, 0.2, 0.3],
            "trait_1": [0.4, 0.5, 0.6],
            "trait_2": [0.7, 0.8, 0.9]
        })
        
        # Convert to numpy
        eff_array = df_eff.iloc[:, 1:].to_numpy(dtype=float)
        
        assert isinstance(eff_array, np.ndarray)
        assert eff_array.shape == (3, 3)
        assert eff_array.dtype == np.float64


# ============================================================================
# Configuration from Dict Tests (for effsize_generation_byconfig)
# ============================================================================

class TestConfigFromDict:
    """Tests for creating configuration from dictionary."""
    
    def test_valid_config_dict(self, temp_dir):
        """Test creating configuration from valid dictionary."""
        csv_path = os.path.join(temp_dir, "effects.csv")
        with open(csv_path, 'w') as f:
            f.write("Sites,trait_0,trait_1\n")
            f.write("150,0.5,0.3\n")
        
        all_config = {
            "BasicRunConfiguration": {
                "cwdir": temp_dir,
                "random_number_seed": 42
            },
            "SeedsConfiguration": {
                "seed_size": 3
            },
            "GenomeElement": {
                "traits_num": {
                    "transmissibility": 1,
                    "drug_resistance": 1
                },
                "effect_size": {
                    "method": "user_input",
                    "filepath": {
                        "csv_path": csv_path
                    },
                    "calibration": {
                        "do_calibration": False,
                        "V_target": []
                    },
                    "causalsites_params": {
                        "method": "beta_binomial",
                        "pis": [],
                        "Ks": []
                    },
                    "effsize_params": {
                        "effsize_function": "n",
                        "normal": {
                            "taus": [0.1, 0.2]
                        },
                        "laplace": {
                            "bs": []
                        },
                        "studentst": {
                            "nv": 3,
                            "s": []
                        }
                    }
                },
                "trait_prob_link": {
                    "calibration": False,
                    "Rs": [],
                    "link": "logit"
                }
            }
        }
        
        error = effsize_generation_byconfig(all_config)
        
        assert error is None


# ============================================================================
# Special Cases and Regression Tests
# ============================================================================

class TestSpecialCases:
    """Tests for special cases and known issues."""
    
    def test_overlapping_genomic_regions(self, temp_dir, basic_trait_num):
        """Test handling of overlapping genomic regions in CSV."""
        csv_path = os.path.join(temp_dir, "overlap.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("100,200,1,0,0\n")
            f.write("150,250,0,1,0\n")  # Overlaps with previous
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            site_frac=[0.5, 0.5, 0.5],
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        candidates = generator._read_candregion_csv()
        
        # Should handle overlapping regions correctly
        # Sites 150-200 can be in both trait_0 and trait_1
        assert len(candidates[0]) > 0
        assert len(candidates[1]) > 0
    
    def test_non_contiguous_genomic_regions(self, temp_dir, basic_trait_num):
        """Test with non-contiguous genomic regions."""
        csv_path = os.path.join(temp_dir, "non_contiguous.csv")
        with open(csv_path, 'w') as f:
            f.write("start,end,trait_0,trait_1,trait_2\n")
            f.write("100,110,1,0,0\n")
            f.write("500,510,1,0,0\n")
            f.write("1000,1010,1,0,0\n")
        
        config = GeneticEffectConfig(
            method="randomly_generate",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path,
            func="n",
            site_frac=[1.0, 0.5, 0.5],
            taus=[0.1, 0.2, 0.3]
        )
        
        generator = EffectGenerator(config)
        candidates = generator._read_candregion_csv()
        
        # Should have all sites from all three regions
        expected_sites = set(range(100, 111)) | set(range(500, 511)) | set(range(1000, 1011))
        assert set(candidates[0]) == expected_sites
    
    def test_all_zeros_in_effect_sizes(self, temp_dir, basic_trait_num):
        """Test handling when all effect sizes are zero."""
        csv_path = os.path.join(temp_dir, "zeros.csv")
        with open(csv_path, 'w') as f:
            f.write("Sites,trait_0,trait_1,trait_2\n")
            f.write("100,0.0,0.0,0.0\n")
            f.write("101,0.0,0.0,0.0\n")
        
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path
        )
        
        generator = EffectGenerator(config)
        df = generator._read_effsize_csv()
        
        # Should read successfully even with all zeros
        assert df.shape[0] == 2
        assert all(df.iloc[:, 1:].values.flatten() == 0)
    
    def test_extremely_large_effect_sizes(self, temp_dir, basic_trait_num):
        """Test handling of extremely large effect sizes."""
        csv_path = os.path.join(temp_dir, "large_effects.csv")
        with open(csv_path, 'w') as f:
            f.write("Sites,trait_0,trait_1,trait_2\n")
            f.write("100,1e6,1e6,1e6\n")
        
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path
        )
        
        generator = EffectGenerator(config)
        df = generator._read_effsize_csv()
        seeds, _ = generator._compute_seed_traits(df)
        
        # Should handle large numbers without overflow
        assert not np.any(np.isinf(seeds.iloc[:, 1:].values))
        assert not np.any(np.isnan(seeds.iloc[:, 1:].values))
    
    def test_negative_effect_sizes(self, temp_dir, basic_trait_num):
        """Test that negative effect sizes are handled correctly."""
        csv_path = os.path.join(temp_dir, "negative.csv")
        with open(csv_path, 'w') as f:
            f.write("Sites,trait_0,trait_1,trait_2\n")
            f.write("100,-0.5,-0.3,-0.7\n")
            f.write("101,-0.2,-0.4,-0.1\n")
        
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num=basic_trait_num,
            random_seed=42,
            csv=csv_path
        )
        
        generator = EffectGenerator(config)
        df = generator._read_effsize_csv()
        
        # Negative values should be preserved
        assert all(df.iloc[:, 1:].values.flatten() < 0)


# ============================================================================
# Link Function Tests
# ============================================================================

class TestLinkFunctions:
    """Tests specifically for link function calibration."""
    
    def test_logit_link_calibration(self, temp_dir):
        """Test logit link calibration with known values."""
        trait_num = {"transmissibility": 2, "drug_resistance": 1}
        
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=5,
            calibration=False,
            trait_num=trait_num,
            random_seed=42,
            csv="dummy.csv",
            calibration_link=True,
            link="logit",
            Rs=[1.5, 2.0, 1.8]
        )
        
        generator = EffectGenerator(config)
        
        var_em = np.array([1.0, 1.0, 1.0])
        
        result = generator._calibrate_linkslope(
            Rs=np.array([1.5, 2.0, 1.8]),
            link_type="logit",
            var_em=var_em,
            trait_num=trait_num
        )
        
        # Check that result contains expected values
        assert "logit" in result
        assert "alpha_trans" in result
        assert "alpha_drug" in result
    
    def test_cloglog_link_calibration(self, temp_dir):
        """Test cloglog link calibration."""
        trait_num = {"transmissibility": 2, "drug_resistance": 1}
        
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=5,
            calibration=False,
            trait_num=trait_num,
            random_seed=42,
            csv="dummy.csv",
            calibration_link=True,
            link="cloglog",
            Rs=[1.5, 2.0, 0.667]
        )
        
        generator = EffectGenerator(config)
        
        var_em = np.array([1.0, 1.0, 1.0])
        
        result = generator._calibrate_linkslope(
            Rs=np.array([1.5, 2.0, 0.667]),
            link_type="cloglog",
            var_em=var_em,
            trait_num=trait_num
        )
        
        assert "cloglog" in result
        assert "alpha_trans" in result
        assert "alpha_drug" in result
    
    def test_link_calibration_with_zero_variance(self, temp_dir):
        """Test link calibration when variance is zero."""
        trait_num = {"transmissibility": 1, "drug_resistance": 1}
        
        config = GeneticEffectConfig(
            method="user_input",
            wk_dir=temp_dir,
            num_init_seq=5,
            calibration=False,
            trait_num=trait_num,
            random_seed=42,
            csv="dummy.csv",
            calibration_link=True,
            link="logit",
            Rs=[1.5, 1.5]
        )
        
        generator = EffectGenerator(config)
        
        var_em = np.array([0.0, 1.0])  # First trait has zero variance
        
        result = generator._calibrate_linkslope(
            Rs=np.array([1.5, 1.5]),
            link_type="logit",
            var_em=var_em,
            trait_num=trait_num
        )
        
        # Should handle zero variance gracefully
        assert result is None


# ============================================================================
# Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility and helper functions."""
    
    def test_rename_preserves_data(self, valid_config_randomly_generate):
        """Test that renaming preserves data values."""
        generator = EffectGenerator(valid_config_randomly_generate)
        
        df_original = pd.DataFrame({
            "Sites": [100, 101],
            "trait_0": [0.1, 0.2],
            "trait_1": [0.3, 0.4],
            "trait_2": [0.5, 0.6]
        })
        
        df_renamed = generator._rename_columns(df_original.copy())
        
        # Check that data values are preserved
        assert df_renamed.iloc[0, 1] == 0.1
        assert df_renamed.iloc[0, 2] == 0.3
        assert df_renamed.iloc[0, 3] == 0.5
    
    def test_build_effect_df_user_input(self, valid_config_user_input):
        """Test _build_effect_df with user_input method."""
        generator = EffectGenerator(valid_config_user_input)
        df = generator._build_effect_df()
        
        assert isinstance(df, pd.DataFrame)
        assert "Sites" in df.columns
        assert df.shape[0] > 0
    
    def test_build_effect_df_randomly_generate(self, valid_config_randomly_generate):
        """Test _build_effect_df with randomly_generate method."""
        generator = EffectGenerator(valid_config_randomly_generate)
        df = generator._build_effect_df()
        
        assert isinstance(df, pd.DataFrame)
        assert "Sites" in df.columns
        # Should have trait columns
        assert any("trait_" in col for col in df.columns)


# ============================================================================
# Run Configuration Tests
# ============================================================================

if __name__ == "__main__":
    """
    Run all tests with pytest.
    
    Usage:
        pytest test_effect_generator.py -v
        pytest test_effect_generator.py -v -k "TestGeneticEffectConfig"
        pytest test_effect_generator.py -v --cov=effect_generator
    """
    pytest.main([__file__, "-v"])
