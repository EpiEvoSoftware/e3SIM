import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the modules to test
# Note: Adjust these imports based on your actual module structure
from genetic_effect_generator import (
    GeneticEffectConfig,
    EffectGenerator,
    CustomizedError,
    effsize_generation_byconfig
)


class TestGeneticEffectConfig:
    """Test suite for GeneticEffectConfig class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.basic_config = {
            "method": "gff",
            "wk_dir": self.temp_dir,
            "num_init_seq": 5,
            "calibration": False,
            "trait_num": {"transmissibility": 1, "drug_resistance": 1},
            "random_seed": 42,
            "pis": [0.5, 0.5]
        }
    
    def teardown_method(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_valid_config_initialization(self):
        """Test valid configuration initialization"""
        config = GeneticEffectConfig(**self.basic_config, site_method="p", func="n", taus=[0.1, 0.2])
        assert config.method == "gff"
        assert config.num_init_seq == 5
        assert config.random_seed == 42
        assert config.params["site_method"] == "p"
    
    def test_invalid_working_directory(self):
        """Test validation with non-existent working directory"""
        invalid_config = self.basic_config.copy()
        invalid_config["wk_dir"] = "/nonexistent/path"
        config = GeneticEffectConfig(**invalid_config)
        
        with pytest.raises(CustomizedError, match="Working directory"):
            config.validate()
    
    def test_invalid_method(self):
        """Test validation with invalid method"""
        invalid_config = self.basic_config.copy()
        invalid_config["method"] = "invalid"
        config = GeneticEffectConfig(**invalid_config)
        
        with pytest.raises(CustomizedError, match="isn't a valid method"):
            config.validate()
    
    def test_invalid_num_init_seq(self):
        """Test validation with invalid seed size"""
        invalid_config = self.basic_config.copy()
        invalid_config["num_init_seq"] = 0
        config = GeneticEffectConfig(**invalid_config)
        
        with pytest.raises(CustomizedError, match="Seed size must be positive"):
            config.validate()
    
    def test_invalid_trait_num_keys(self):
        """Test validation with wrong number of trait keys"""
        invalid_config = self.basic_config.copy()
        invalid_config["trait_num"] = {"transmissibility": 1}
        config = GeneticEffectConfig(**invalid_config)
        
        with pytest.raises(CustomizedError, match="specify exactly 2 traits"):
            config.validate()
    
    def test_invalid_trait_num_sum(self):
        """Test validation with trait sum less than 1"""
        invalid_config = self.basic_config.copy()
        invalid_config["trait_num"] = {"transmissibility": 0, "drug_resistance": 0}
        config = GeneticEffectConfig(**invalid_config)
        
        with pytest.raises(CustomizedError, match="sums up to at least 1"):
            config.validate()
    
    def test_invalid_site_method(self):
        """Test validation with invalid site method"""
        config = GeneticEffectConfig(**self.basic_config, site_method="invalid")
        
        with pytest.raises(CustomizedError, match="isn't a valid method for resampling"):
            config.validate()
    
    def test_pis_length_mismatch(self):
        """Test validation with mismatched pis length"""
        invalid_config = self.basic_config.copy()
        invalid_config["pis"] = [0.5]  # Should be 2 elements
        config = GeneticEffectConfig(**invalid_config, site_method="p")
        
        with pytest.raises(CustomizedError, match="success probability list"):
            config.validate()
    
    def test_invalid_pis_values(self):
        """Test validation with invalid pi values"""
        invalid_config = self.basic_config.copy()
        invalid_config["pis"] = [0.5, 1.5]  # 1.5 is out of range
        config = GeneticEffectConfig(**invalid_config, site_method="p")
        
        with pytest.raises(CustomizedError, match="probability.*has to be within"):
            config.validate()
    
    def test_ks_validation(self):
        """Test validation with Ks parameter"""
        config = GeneticEffectConfig(**self.basic_config, site_method="n", Ks=[10, 20])
        config.validate()  # Should not raise
        
        # Test with wrong length
        config = GeneticEffectConfig(**self.basic_config, site_method="n", Ks=[10])
        with pytest.raises(CustomizedError, match="causal site number list"):
            config.validate()
        
        # Test with non-integer
        config = GeneticEffectConfig(**self.basic_config, site_method="n", Ks=[10.5, 20])
        with pytest.raises(CustomizedError, match="has to be an integer"):
            config.validate()
    
    def test_calibration_link_validation(self):
        """Test calibration link validation"""
        config = GeneticEffectConfig(
            **self.basic_config,
            site_method = "p", # added after test failure
            calibration_link=True,
            link="invalid",
            Rs=[]
        )
        
        with pytest.raises(CustomizedError, match="needs to be either 'logit' or 'cloglog'"):
            config.validate()


class TestEffectGenerator:
    """Test suite for EffectGenerator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.basic_config = GeneticEffectConfig(
            method="gff",
            wk_dir=self.temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num={"transmissibility": 1, "drug_resistance": 1},
            random_seed=42,
            pis=[0.5, 0.5],
            site_method = "p",
            func = "n",
            taus =[1, 2]
        )
        self.generator = EffectGenerator(self.basic_config)
    
    def teardown_method(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('genetic_effect_generator.EffectGenerator._read_gff_sites')
    @patch('genetic_effect_generator.EffectGenerator._compute_seed_traits')
    @patch('genetic_effect_generator.EffectGenerator._write_outputs')
    def test_run_successful(self, mock_write, mock_compute, mock_read_gff):
        """Test successful run of effect generator"""
        mock_read_gff.return_value = [100, 200, 300]
        mock_compute.return_value = (
            pd.DataFrame({"Seed_ID": [0, 1, 2], "trait_0": [0.1, 0.2, 0.3], "trait_1": [0.4, 0.5, 0.6]}),
            pd.DataFrame({"Sites": [100, 200], "seed_0": [1, 0], "seed_1": [1, 1], "seed_2": [0, 1]})
        )
        
        result = self.generator.run()
        assert result is None
        mock_write.assert_called_once()
    
    def test_read_gff_sites_valid(self):
        """Test reading GFF file with valid content"""
        gff_content = """##gff-version 3
#sequence-region chr1 1 1000
chr1	source	gene	100	200	.	+	.	ID=gene1
chr1	source	gene	300	350	.	+	.	ID=gene2
"""
        gff_path = os.path.join(self.temp_dir, "test.gff")
        with open(gff_path, 'w') as f:
            f.write(gff_content)
        
        self.generator.cfg.params["gff"] = gff_path
        sites = self.generator._read_gff_sites()
        
        expected_sites = list(range(100, 201)) + list(range(300, 351))
        assert sites == expected_sites
    
    def test_read_gff_sites_nonexistent(self):
        """Test reading non-existent GFF file"""
        self.generator.cfg.params["gff"] = "/nonexistent/file.gff"
        
        with pytest.raises(CustomizedError, match="not a valid file path"):
            self.generator._read_gff_sites()
    
    def test_read_effsize_csv_valid(self):
        """Test reading valid effect size CSV"""
        csv_content = """Sites,trait1,trait2
100,0.1,0.2
200,0.3,0.4
300,0.5,0.6"""
        csv_path = os.path.join(self.temp_dir, "effects.csv")
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        self.generator.cfg.params["csv"] = csv_path
        df = self.generator._read_effsize_csv()
        
        assert df.shape == (3, 3)
        assert list(df.columns) == ["Sites", "trait_0", "trait_1"]
        assert df["Sites"].tolist() == [100, 200, 300]
    
    def test_read_effsize_csv_nonexistent(self):
        """Test reading non-existent CSV file"""
        self.generator.cfg.params["csv"] = "/nonexistent/file.csv"
        
        with pytest.raises(CustomizedError, match="does not exist"):
            self.generator._read_effsize_csv()
    
    def test_select_sites_bernoulli(self):
        """Test site selection with Bernoulli trials"""
        np.random.seed(42)
        candidates = list(range(100, 110))
        df = self.generator._select_sites(candidates, "p", [0.3, 0.5], None)
        
        assert "Sites" in df.columns
        assert "trait_0" in df.columns
        assert "trait_1" in df.columns
        assert all(df["trait_0"].isin([0, 1]))
        assert all(df["trait_1"].isin([0, 1]))
    
    def test_select_sites_uniform(self):
        """Test site selection with uniform sampling"""
        np.random.seed(42)
        candidates = list(range(100, 120))
        df = self.generator._select_sites(candidates, "n", None, [5, 3])
        
        assert "Sites" in df.columns
        assert df["trait_0"].sum() == 5
        assert df["trait_1"].sum() == 3
    
    def test_select_sites_k_too_large(self):
        """Test site selection when K exceeds candidate sites"""
        candidates = [100, 101, 102]
        
        with pytest.raises(CustomizedError, match="larger than the candidate site list"):
            self.generator._select_sites(candidates, "n", None, [5, 2])
    
    def test_pointnormal_sampling(self):
        """Test point normal distribution sampling"""
        np.random.seed(42)
        samples = self.generator._pointnormal(100, 0.5)
        
        assert len(samples) == 100
        assert abs(np.mean(samples)) < 0.1  # Should be close to 0
        assert 0.4 < np.std(samples) < 0.6  # Should be close to 0.5
    
    def test_laplace_sampling(self):
        """Test Laplace distribution sampling"""
        np.random.seed(42)
        samples = self.generator._laplace(100, 0.5)
        
        assert len(samples) == 100
        assert abs(np.mean(samples)) < 0.1  # Should be close to 0
    
    def test_studentst_sampling(self):
        """Test Student's t distribution sampling"""
        np.random.seed(42)
        samples = self.generator._studentst(100, scale=2.0, nv=5)
        
        assert len(samples) == 100
        # Check that samples are scaled appropriately
        assert np.std(samples) > 1.5  # Should be affected by scale
    
    def test_sample_univariate(self):
        """Test univariate sampling of effect sizes"""
        df_id = pd.DataFrame({
            "Sites": [100, 200, 300],
            "trait_0": [1, 0, 1],
            "trait_1": [0, 1, 1]
        })
        
        self.generator.cfg.params["func"] = "n"
        self.generator.cfg.params["taus"] = [0.1, 0.2]
        
        result = self.generator._sample_univariate(df_id)
        
        # Check that zeros remain zeros
        assert result.loc[result["Sites"] == 200, "trait_0"].values[0] == 0
        assert result.loc[result["Sites"] == 100, "trait_1"].values[0] == 0
        
        # Check that ones are replaced with sampled values
        assert result.loc[result["Sites"] == 100, "trait_0"].values[0] != 1
        assert result.loc[result["Sites"] == 200, "trait_1"].values[0] != 1
    
    def test_compute_seed_traits_no_vcf(self):
        """Test computing seed traits without VCF directory"""
        df_eff = pd.DataFrame({
            "Sites": [100, 200],
            "trait_0": [0.1, 0.2],
            "trait_1": [0.3, 0.4]
        })
        
        result, _ = self.generator._compute_seed_traits(df_eff)
        
        assert result.shape == (3, 3)  # 3 seeds, 3 columns
        assert all(result["trait_0"] == 0)
        assert all(result["trait_1"] == 0)
    
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_compute_seed_traits_with_vcf(self, mock_listdir, mock_exists):
        """Test computing seed traits with VCF files"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["seed1.vcf", "seed2.vcf", "seed3.vcf"]
        
        vcf_content = """##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	100	.	A	T	.	.	.
chr1	200	.	G	C	.	.	."""
        
        df_eff = pd.DataFrame({
            "Sites": [100, 200, 300],
            "trait_0": [0.1, 0.2, 0.3],
            "trait_1": [0.4, 0.5, 0.6]
        })
        
        with patch('builtins.open', mock_open(read_data=vcf_content)):
            result, _ = self.generator._compute_seed_traits(df_eff)
        
        assert result.shape[1] == 3  # Seed_ID + 2 traits
        assert "Seed_ID" in result.columns
    
    def test_calibrate(self):
        """Test effect size calibration"""
        df_eff = pd.DataFrame({
            "Sites": [100, 200],
            "trait_0": [0.1, 0.2],
            "trait_1": [0.3, 0.4]
        })
        
        seeds_state = pd.DataFrame({
            "Sites": [100, 200],
            "seed_0": [1, 0],
            "seed_1": [1, 1],
            "seed_2": [0, 1]
        })
        
        self.generator.cfg.params["var_target"] = [1.0, 1.0]
        
        df_calibrated, var_empirical = self.generator._calibrate(df_eff, seeds_state)
        
        assert df_calibrated.shape == df_eff.shape
        assert "Sites" in df_calibrated.columns
        assert len(var_empirical) == 2
    
    def test_rename_columns(self):
        """Test column renaming"""
        df = pd.DataFrame({
            "Sites": [100, 200],
            "trait_0": [0.1, 0.2],
            "trait_1": [0.3, 0.4]
        })
        
        renamed = self.generator._rename_columns(df)
        
        assert "transmissibility_1" in renamed.columns
        assert "drug_resistance_1" in renamed.columns
        assert renamed.columns[0] == "Sites"
    
    def test_calibrate_linkslope_logit(self):
        """Test link slope calibration for logit"""
        Rs = np.array([1.5, 2.0])
        var_em = np.array([1.0, 0.5])
        trait_num = {"transmissibility": 1, "drug_resistance": 1}
        
        with patch('builtins.print'):
            result = self.generator._calibrate_linkslope(Rs, "logit", var_em, trait_num)
        
        assert isinstance(result, str)
        config = json.loads(result)
        assert config["link"] == "logit"
        assert "logit" in config
    
    def test_calibrate_linkslope_cloglog(self):
        """Test link slope calibration for cloglog"""
        Rs = np.array([1.5, 2.0])
        var_em = np.array([1.0, 0.5])
        trait_num = {"transmissibility": 1, "drug_resistance": 1}
        
        with patch('builtins.print'):
            result = self.generator._calibrate_linkslope(Rs, "cloglog", var_em, trait_num)
        
        assert isinstance(result, str)
        config = json.loads(result)
        assert config["link"] == "cloglog"
    
    def test_calibrate_linkslope_invalid(self):
        """Test link slope calibration with invalid link type"""
        Rs = np.array([1.5, 2.0])
        var_em = np.array([1.0, 0.5])
        trait_num = {"transmissibility": 1, "drug_resistance": 1}
        
        with pytest.raises(CustomizedError, match="Unknown link_type"):
            self.generator._calibrate_linkslope(Rs, "invalid", var_em, trait_num)
    
    def test_write_outputs(self):
        """Test writing output files"""
        df_eff = pd.DataFrame({
            "Sites": [100, 200],
            "trait_0": [0.1, 0.2],
            "trait_1": [0.3, 0.4]
        })
        
        seeds = pd.DataFrame({
            "Seed_ID": [0, 1],
            "trait_0": [0.5, 0.6],
            "trait_1": [0.7, 0.8]
        })
        
        self.generator._write_outputs(df_eff, seeds)
        
        assert os.path.exists(os.path.join(self.temp_dir, "causal_gene_info.csv"))
        assert os.path.exists(os.path.join(self.temp_dir, "seeds_trait_values.csv"))
        
        # Verify content
        df_loaded = pd.read_csv(os.path.join(self.temp_dir, "causal_gene_info.csv"))
        assert df_loaded.shape == df_eff.shape


class TestEffsizeGenerationByConfig:
    """Test suite for effsize_generation_byconfig function"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "BasicRunConfiguration": {
                "cwdir": self.temp_dir,
                "random_number_seed": 42
            },
            "SeedsConfiguration": {
                "seed_size": 5
            },
            "GenomeElement": {
                "effect_size": {
                    "method": "csv",
                    "filepath": {
                        "csv_path": "test.csv",
                        "gff_path": ""
                    },
                    "calibration": {
                        "do_calibration": False,
                        "V_target": []
                    },
                    "causalsites_params": {
                        "method": "p",
                        "pis": [0.5, 0.5],
                        "Ks": []
                    },
                    "effsize_params": {
                        "effsize_function": "n",
                        "normal": {"taus": [0.1, 0.2]},
                        "laplace": {"bs": []},
                        "studentst": {"nv": 3, "s": []}
                    }
                },
                "traits_num": {
                    "transmissibility": 1,
                    "drug_resistance": 1
                },
                "trait_prob_link": {
                    "calibration": False,
                    "Rs": [],
                    "link": "logit"
                }
            }
        }
    
    def teardown_method(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('genetic_effect_generator.EffectGenerator.run')
    def test_effsize_generation_byconfig_success(self, mock_run):
        """Test successful generation from config"""
        mock_run.return_value = None
        
        # Create a dummy CSV file
        csv_path = os.path.join(self.temp_dir, "test.csv")
        with open(csv_path, 'w') as f:
            f.write("Sites,trait1,trait2\n100,0.1,0.2\n")
        self.config["GenomeElement"]["effect_size"]["filepath"]["csv_path"] = csv_path
        
        result = effsize_generation_byconfig(self.config)
        
        assert result is None
        mock_run.assert_called_once()
    
    def test_effsize_generation_byconfig_error(self):
        """Test error handling in generation from config"""
        # Use invalid config to trigger error
        self.config["GenomeElement"]["effect_size"]["method"] = "invalid"
        
        result = effsize_generation_byconfig(self.config)
        
        assert result is not None  # Should return error


class TestIntegration:
    """Integration tests for the complete workflow"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('genetic_effect_generator.EffectGenerator._read_gff_sites')
    def test_complete_workflow_gff(self, mock_read_gff):
        """Test complete workflow with GFF method"""
        mock_read_gff.return_value = list(range(100, 200))
        
        config = GeneticEffectConfig(
            method="gff",
            wk_dir=self.temp_dir,
            num_init_seq=3,
            calibration=True,
            trait_num={"transmissibility": 1, "drug_resistance": 1},
            random_seed=42,
            pis=[0.3, 0.4],
            site_method="p",
            func="n",
            taus=[0.1, 0.2],
            var_target=[1.0, 1.0],
            gff="dummy.gff"
        )
        
        generator = EffectGenerator(config)
        result = generator.run()
        
        assert result is None
        assert os.path.exists(os.path.join(self.temp_dir, "causal_gene_info.csv"))
        assert os.path.exists(os.path.join(self.temp_dir, "seeds_trait_values.csv"))
    
    def test_complete_workflow_csv(self):
        """Test complete workflow with CSV method"""
        # Create test CSV
        csv_path = os.path.join(self.temp_dir, "effects.csv")
        with open(csv_path, 'w') as f:
            f.write("Sites,trait1,trait2\n100,0.1,0.2\n200,0.3,0.4\n")
        
        config = GeneticEffectConfig(
            method="csv",
            wk_dir=self.temp_dir,
            num_init_seq=3,
            calibration=False,
            trait_num={"transmissibility": 1, "drug_resistance": 1},
            random_seed=42,
            pis=[],
            csv=csv_path
        )
        
        generator = EffectGenerator(config)
        result = generator.run()
        
        assert result is None
        assert os.path.exists(os.path.join(self.temp_dir, "causal_gene_info.csv"))


# Additional test utilities
@pytest.fixture
def mock_config():
    """Fixture providing a mock configuration"""
    temp_dir = tempfile.mkdtemp()
    yield {
        "method": "gff",
        "wk_dir": temp_dir,
        "num_init_seq": 5,
        "calibration": False,
        "trait_num": {"transmissibility": 1, "drug_resistance": 1},
        "random_seed": 42,
        "pis": [0.5, 0.5],
        "func": "n",
        "taus": [1, 2],
        "site_method": "p"
    }
    shutil.rmtree(temp_dir)


@pytest.mark.parametrize("method,expected", [
    ("csv", True),
    ("gff", True),
    ("invalid", False)
])
def test_method_validation(method, expected, mock_config):
    """Parametrized test for method validation"""
    mock_config["method"] = method
    config = GeneticEffectConfig(**mock_config)
    
    if expected:
        try:
            config.validate()
            assert True
        except CustomizedError:
            assert False
    else:
        with pytest.raises(CustomizedError):
            config.validate()

if __name__ == "__main__":
    # Run tests with: python -m pytest test_genetic_effect_generator.py -v
    pytest.main([__file__, "-v"])




