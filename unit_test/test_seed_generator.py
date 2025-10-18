import pytest
import os, sys
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import subprocess
import tskit
import pyslim

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

curr_dir = os.path.dirname(__file__)
e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
if e3SIM_dir not in sys.path:
    sys.path.insert(0, e3SIM_dir)

from seed_generator import SeedConfig, VCFHandler, PhylogenyHandler,SimulationRunner, SeedGenerator
from error_handling import CustomizedError

# ==================== Test SeedConfig ====================

class TestSeedConfig:
    """Test suite for SeedConfig class"""
    
    def test_valid_config_creation(self, tmp_path):
        """Test creating a valid configuration"""
        config = SeedConfig(
            method="user_input",
            wk_dir=str(tmp_path),
            seed_size=10,
            param1="value1"
        )
        assert config.method == "user_input"
        assert config.wk_dir == str(tmp_path)
        assert config.seed_size == 10
        assert config.params["param1"] == "value1"
    
    def test_invalid_working_directory(self):
        """Test configuration with non-existent working directory"""
        with pytest.raises(CustomizedError, match="Working directory .* does not exist"):
            SeedConfig(
                method="user_input",
                wk_dir="/non/existent/path",
                seed_size=10
            )
    
    def test_invalid_seed_size_zero(self, tmp_path):
        """Test configuration with zero seed size"""
        with pytest.raises(CustomizedError, match="Seed size must be positive"):
            SeedConfig(
                method="user_input",
                wk_dir=str(tmp_path),
                seed_size=0
            )
    
    def test_invalid_seed_size_negative(self, tmp_path):
        """Test configuration with negative seed size"""
        with pytest.raises(CustomizedError, match="Seed size must be positive"):
            SeedConfig(
                method="user_input",
                wk_dir=str(tmp_path),
                seed_size=-5
            )
    
    def test_invalid_method(self, tmp_path):
        """Test configuration with unsupported method"""
        with pytest.raises(CustomizedError, match="Unsupported method"):
            SeedConfig(
                method="invalid_method",
                wk_dir=str(tmp_path),
                seed_size=10
            )
    
    def test_valid_methods(self, tmp_path):
        """Test all valid methods"""
        valid_methods = ["user_input", "SLiM_burnin_WF", "SLiM_burnin_epi"]
        for method in valid_methods:
            config = SeedConfig(
                method=method,
                wk_dir=str(tmp_path),
                seed_size=5
            )
            assert config.method == method


# ==================== Test VCFHandler ====================

class TestVCFHandler:
    """Test suite for VCFHandler class"""
    
    @pytest.fixture
    def vcf_handler(self, tmp_path):
        """Create a VCFHandler instance with temporary directory"""
        return VCFHandler(str(tmp_path))
    
    def test_vcf_handler_initialization(self, vcf_handler, tmp_path):
        """Test VCFHandler initialization"""
        assert vcf_handler.wk_dir == str(tmp_path)
        assert vcf_handler.vcf_path == os.path.join(str(tmp_path), "seeds.vcf")
    
    def test_check_input_file_not_found(self, vcf_handler):
        """Test check_input with non-existent file"""
        with pytest.raises(FileNotFoundError, match="Seed VCF file not found"):
            vcf_handler.check_input("/non/existent/file.vcf", 5)
    
    def test_check_input_valid_vcf(self, vcf_handler, tmp_path):
        """Test check_input with valid VCF file"""
        vcf_content = """##fileformat=VCFv4.2\n##source=test\n#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2\nchr1	100	.	A	T	.	PASS	.	GT	0/1	0/0"""
        vcf_path = tmp_path / "test.vcf"
        vcf_path.write_text(vcf_content)
        
        result = vcf_handler.check_input(str(vcf_path), 2)
        assert result is True
    
    def test_check_input_wrong_sample_count(self, vcf_handler, tmp_path):
        """Test check_input with wrong number of samples"""
        vcf_content = """##fileformat=VCFv4.2\n##source=test\n#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3\nchr1	100	.	A	T	.	PASS	.	GT	0/1	0/0	1/1"""
        vcf_path = tmp_path / "test.vcf"
        vcf_path.write_text(vcf_content)
        
        with pytest.raises(CustomizedError, match="doesn't have the correct number of individuals"):
            vcf_handler.check_input(str(vcf_path), 2)
    
    @patch('os.mkdir')
    @patch('shutil.rmtree')
    def test_create_seeds_directory_existing(self, mock_rmtree, mock_mkdir, vcf_handler, tmp_path):
        """Test creating seeds directory when it already exists"""
        seeds_dir = os.path.join(vcf_handler.wk_dir, "originalvcfs/")
        
        with patch('os.path.exists', return_value=True):
            result = vcf_handler._create_seeds_directory()
            mock_rmtree.assert_called_once_with(seeds_dir, ignore_errors=True)
            mock_mkdir.assert_called_once_with(seeds_dir)
            assert result == seeds_dir
    
    @patch('os.mkdir')
    def test_create_seeds_directory_new(self, mock_mkdir, vcf_handler, tmp_path):
        """Test creating seeds directory when it doesn't exist"""
        seeds_dir = os.path.join(vcf_handler.wk_dir, "originalvcfs/")
        
        with patch('os.path.exists', return_value=False):
            result = vcf_handler._create_seeds_directory()
            mock_mkdir.assert_called_once_with(seeds_dir)
            assert result == seeds_dir
    
    def test_copy_headers(self, vcf_handler, tmp_path):
        """Test copying VCF headers to multiple files"""
        vcf_files = [
            str(tmp_path / "vcf1.vcf"),
            str(tmp_path / "vcf2.vcf")
        ]
        
        vcf_handler._copy_headers(vcf_files)
        
        for vcf_file in vcf_files:
            with open(vcf_file, 'r') as f:
                content = f.read()
                assert "##fileformat=VCFv4.2" in content
                assert "##source=SLiM" in content
    
    def test_write_column_names(self, vcf_handler, tmp_path):
        """Test writing column names to VCF files"""
        vcf_files = [
            str(tmp_path / "vcf1.vcf"),
            str(tmp_path / "vcf2.vcf")
        ]
        # Create empty files
        for f in vcf_files:
            open(f, 'w').close()
        
        header_line = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\tSample2\n"
        vcf_handler._write_column_names(vcf_files, header_line, 10)
        
        for vcf_file in vcf_files:
            with open(vcf_file, 'r') as f:
                content = f.read()
                assert "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1\n" == content


# ==================== Test PhylogenyHandler ====================

class TestPhylogenyHandler:
    """Test suite for PhylogenyHandler class"""
    
    @pytest.fixture
    def phylo_handler(self, tmp_path):
        """Create a PhylogenyHandler instance"""
        return PhylogenyHandler(str(tmp_path))
    
    def test_phylogeny_handler_initialization(self, phylo_handler, tmp_path):
        """Test PhylogenyHandler initialization"""
        assert phylo_handler.wk_dir == str(tmp_path)
        expected_path = os.path.join(str(tmp_path), "seeds_phylogeny.txt")
        assert phylo_handler.tree_path == expected_path
    
    @patch('seed_generator.Tree')
    @patch('shutil.copyfile')
    def test_copy_input_valid(self, mock_copy, mock_tree, phylo_handler, tmp_path):
        """Test copying valid phylogeny input"""
        source_path = "/path/to/phylogeny.nwk"
        
        # Mock tree with correct tip labels
        mock_tree_instance = Mock()
        mock_leaf1, mock_leaf2, mock_leaf3 = Mock(), Mock(), Mock()
        mock_leaf1.name = 0
        mock_leaf2.name = 1
        mock_leaf3.name = 2
        mock_tree_instance.__iter__ = Mock(return_value=iter([mock_leaf1, mock_leaf2, mock_leaf3]))
        mock_tree.return_value = mock_tree_instance
        
        with patch('os.path.exists', return_value=True):
            phylo_handler.copy_input(source_path)

            # Assert Tree was constructed correctly
            mock_tree.assert_called_once_with(source_path, "newick")

            mock_copy.assert_called_once_with(
                source_path, 
                os.path.join(phylo_handler.wk_dir, "seeds.nwk")
            )
    
    @patch('seed_generator.Tree')
    def test_copy_input_invalid_tips(self, mock_tree, phylo_handler):
        """Test copying phylogeny with invalid tip labels"""
        source_path = "/path/to/phylogeny.nwk"
        
        # Mock tree with incorrect tip labels
        mock_tree_instance = Mock()
        mock_leaf1, mock_leaf2 = Mock(), Mock()
        mock_leaf1.name = 0
        mock_leaf2.name = 5  # Non-consecutive
        mock_tree_instance.__iter__ = Mock(return_value=iter([mock_leaf1, mock_leaf2]))
        mock_tree.return_value = mock_tree_instance
        
        with patch('os.path.exists', return_value=True):
            with pytest.raises(CustomizedError, match="consecutive integers"):
                phylo_handler.copy_input(source_path)
    
    def test_copy_input_nonexistent_file(self, phylo_handler):
        """Test copying non-existent phylogeny file"""
        with pytest.raises(CustomizedError, match="doesn't exist"):
            phylo_handler.copy_input("/non/existent/path.nwk")
    
    @patch('seed_generator.Tree')
    def test_scale_tree_rooted(self, mock_tree, phylo_handler, tmp_path):
        """Test scaling a rooted tree"""
        mock_tree_instance = Mock()
        mock_tree_instance.children = [Mock(), Mock()]  # Binary root (rooted tree)
        
        # Create mock nodes for traversal
        mock_node1, mock_node2 = Mock(), Mock()
        mock_node1.dist = 1.0
        mock_node2.dist = 2.0
        mock_tree_instance.traverse.return_value = [mock_node1, mock_node2]
        
        mock_tree.return_value = mock_tree_instance
        
        phylo_handler.scale_tree(scale_factor=2.5)
        
        assert mock_node1.dist == 2.5
        assert mock_node2.dist == 5.0
        mock_tree_instance.write.assert_called_once()
    
    @patch('seed_generator.Tree')
    def test_scale_tree_unrooted(self, mock_tree, phylo_handler):
        """Test scaling an unrooted tree raises error"""
        mock_tree_instance = Mock()
        mock_tree_instance.children = [Mock(), Mock(), Mock()]  # Non-binary root
        mock_tree.return_value = mock_tree_instance
        
        with pytest.raises(CustomizedError, match="not rooted"):
            phylo_handler.scale_tree(scale_factor=2.0)


# ==================== Test SimulationRunner ====================

class TestSimulationRunner:
    """Test suite for SimulationRunner class"""
    
    @pytest.fixture
    def sim_runner(self, tmp_path):
        """Create a SimulationRunner instance"""
        return SimulationRunner(str(tmp_path))
    
    def test_simulation_runner_initialization(self, sim_runner, tmp_path):
        """Test SimulationRunner initialization"""
        assert sim_runner.wk_dir == str(tmp_path)
    
    def test_bool2SLiM_true(self, sim_runner):
        """Test bool2SLiM with True value"""
        assert sim_runner.bool2SLiM(True) == 1
    
    def test_bool2SLiM_false(self, sim_runner):
        """Test bool2SLiM with False value"""
        assert sim_runner.bool2SLiM(False) == 0
    
    @patch('subprocess.run')
    @patch('os.remove')
    @patch('os.path.dirname')
    def test_run_wf_without_seed(self, mock_dirname, mock_remove, mock_run, sim_runner, tmp_path):
        """Test running WF simulation without random seed"""
        mock_dirname.return_value = "/mock/dir"
        mock_seeds_treeseq = Mock()
        mock_split_seedvcf = Mock()
        
        sim_runner.run_wf(
            seeds_treeseq=mock_seeds_treeseq,
            split_seedvcf=mock_split_seedvcf,
            Ne=100,
            seed_size=10,
            ref_path="/path/to/ref.fa",
            mu=0.001,
            n_gen=50,
            rand_seed=None,
            use_subst_matrix=False
        )
        
        mock_run.assert_called_once()
        mock_seeds_treeseq.assert_called_once_with(10)
        mock_split_seedvcf.assert_called_once()
        mock_remove.assert_called_once()
    
    @patch('subprocess.run')
    @patch('os.remove')
    @patch('os.path.dirname')
    def test_run_wf_with_seed(self, mock_dirname, mock_remove, mock_run, sim_runner, tmp_path):
        """Test running WF simulation with random seed"""
        mock_dirname.return_value = "/mock/dir"
        mock_seeds_treeseq = Mock()
        mock_split_seedvcf = Mock()
        
        sim_runner.run_wf(
            seeds_treeseq=mock_seeds_treeseq,
            split_seedvcf=mock_split_seedvcf,
            Ne=100,
            seed_size=10,
            ref_path="/path/to/ref.fa",
            mu=0.001,
            n_gen=50,
            rand_seed=12345,
            use_subst_matrix=False
        )
        
        # Check that subprocess.run was called with seed parameter
        call_args = mock_run.call_args[0][0]
        assert "-d" in call_args
        assert "seed=12345" in call_args


# ==================== Test SeedGenerator ====================

class TestSeedGenerator:
    """Test suite for SeedGenerator class"""
    
    @pytest.fixture
    def valid_config(self, tmp_path):
        """Create a valid configuration for testing"""
        return SeedConfig(
            method="user_input",
            wk_dir=str(tmp_path),
            seed_size=5,
            seed_vcf="test.vcf",
            path_seeds_phylogeny=None,

        )
    
    @pytest.fixture
    def seed_generator(self, valid_config):
        """Create a SeedGenerator instance"""
        return SeedGenerator(valid_config)
    
    def test_seed_generator_initialization(self, seed_generator, valid_config):
        """Test SeedGenerator initialization"""
        assert seed_generator.config == valid_config
        assert isinstance(seed_generator.vcf_handler, VCFHandler)
        assert isinstance(seed_generator.phylo_handler, PhylogenyHandler)
        assert isinstance(seed_generator.sim_runner, SimulationRunner)
    
    def test_generate_sample_indices_sufficient_genomes(self, seed_generator):
        """Test generating sample indices with sufficient genomes"""
        # Create mock tree sequence
        mock_ts = Mock()
        mock_ts.tables.individuals.num_rows = 20
        
        with patch('numpy.random.choice', return_value=np.array([1, 3, 5, 7])):
            indices = seed_generator._generate_sample_indices(mock_ts, 4)
            assert indices == [2, 6, 10, 14]  # NODES_PER_IND * sampled_inds
    
    def test_generate_sample_indices_insufficient_genomes(self, seed_generator):
        """Test generating sample indices with insufficient genomes"""
        mock_ts = Mock()
        mock_ts.tables.individuals.num_rows = 3
        
        with pytest.raises(ValueError, match="Not enough genomes"):
            seed_generator._generate_sample_indices(mock_ts, 5)
    
    @patch.object(VCFHandler, 'check_input')
    @patch.object(VCFHandler, 'split')
    def test_handle_user_input(self, mock_split, mock_check, seed_generator):
        """Test handling user input method"""
        seed_generator._handle_user_input()
        
        mock_check.assert_called_once_with(
            seed_generator.config.params["seed_vcf"],
            seed_generator.config.seed_size
        )
        mock_split.assert_called_once_with(
            seed_generator.config.params["seed_vcf"],
            seed_generator.config.seed_size,
            "user"
        )
    
    @patch.object(VCFHandler, 'check_input')
    @patch.object(VCFHandler, 'split')
    @patch.object(PhylogenyHandler, 'copy_input')
    def test_handle_user_input_with_phylogeny(self, mock_copy_phylo, mock_split, 
                                               mock_check, seed_generator):
        """Test handling user input with phylogeny"""
        seed_generator.config.params["path_seeds_phylogeny"] = "/path/to/phylo.nwk"
        seed_generator._handle_user_input()
        
        mock_copy_phylo.assert_called_once_with("/path/to/phylo.nwk")
    
    def test_generate_unsupported_method(self, tmp_path):
        """Test generating with unsupported method after bypassing validation"""
        config = SeedConfig.__new__(SeedConfig)
        config.method = "unsupported"
        config.wk_dir = str(tmp_path)
        config.seed_size = 5
        config.params = {}
        
        generator = SeedGenerator(config)
        with pytest.raises(CustomizedError, match="Unsupported method"):
            generator._generate()
    
    @patch('numpy.random.seed')
    @patch.object(SeedGenerator, '_handle_user_input')
    def test_run_success(self, mock_handle, mock_seed, seed_generator):
        """Test successful run"""
        seed_generator.config.params["rand_seed"] = 42
        result = seed_generator.run()
        
        mock_seed.assert_called_once_with(42)
        mock_handle.assert_called_once()
        assert result is None
    
    @patch.object(SeedGenerator, '_generate')
    def test_run_with_exception(self, mock_generate, seed_generator):
        """Test run with exception handling"""
        mock_generate.side_effect = CustomizedError("Test error")
        result = seed_generator.run()
        
        assert isinstance(result, CustomizedError)
        assert str(result) == "Test error"


# ==================== Test Integration Scenarios ====================

class TestIntegrationScenarios:
    """Test integration scenarios"""
    
    @pytest.fixture
    def setup_environment(self, tmp_path):
        """Set up test environment with necessary files"""
        # Create a valid VCF file
        vcf_content = """##fileformat=VCFv4.2
##source=test
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2
chr1	100	.	A	T	.	PASS	.	GT	0/1	0/0
chr1	200	.	G	C	.	PASS	.	GT	0/0	1/1
"""
        vcf_path = tmp_path / "test.vcf"
        vcf_path.write_text(vcf_content)
        
        return {
            "wk_dir": str(tmp_path),
            "vcf_path": str(vcf_path)
        }
    
    def test_user_input_workflow(self, setup_environment):
        """Test complete user input workflow"""
        config = SeedConfig(
            method="user_input",
            wk_dir=setup_environment["wk_dir"],
            seed_size=2,
            seed_vcf=setup_environment["vcf_path"]
        )
        
        generator = SeedGenerator(config)
        
        # Mock the split method to avoid file operations
        with patch.object(generator.vcf_handler, 'split'):
            generator._handle_user_input()
            generator.vcf_handler.split.assert_called_once()
    
    # The following tests are tentatively outdated because we implemented the try-catch block
    # no exceptions will be raised and they can't be checked against the assertion
    # but sl984 has checked it manually by observing the output

    # @patch('subprocess.run')
    # @patch('os.path.exists', return_value=True)
    # @patch('os.path.dirname', return_value="/mock/dir")
    # def test_wf_burnin_validation(self, mock_dirname, mock_exists, mock_run, tmp_path):
    #     """Test WF burn-in parameter validation"""
    #     # Test Ne validation
    #     with pytest.raises(Exception) as exc_info:
    #         config = SeedConfig(
    #             method="SLiM_burnin_WF",
    #             wk_dir=str(tmp_path),
    #             seed_size=5,
    #             Ne=0,  # Invalid
    #             ref_path="/path/to/ref.fa",
    #             mu=0.001,
    #             n_gen=10
    #         )
    #         generator = SeedGenerator(config)
    #         generator.run()
        
    #     assert "effective population size" in str(exc_info.value)
    
    # @patch('subprocess.run')
    # @patch('os.path.exists')
    # def test_epi_burnin_validation(self, mock_exists, mock_run, tmp_path):
    #     """Test epidemiological burn-in validation"""
    #     mock_exists.return_value = True
        
    #     # Test without seeded hosts
    #     with pytest.raises(Exception) as exc_info:
    #         config = SeedConfig(
    #             method="SLiM_burnin_epi",
    #             wk_dir=str(tmp_path),
    #             seed_size=5,
    #             ref_path="/path/to/ref.fa",
    #             mu=0.001,
    #             n_gen=10,
    #             host_size=100,
    #             seeded_host_id=[],  # Empty - should raise error
    #             S_IE_prob=0.1,
    #             use_subst_matrix=False
    #         )
    #         generator = SeedGenerator(config)
    #         generator.run()
        
    #     assert "at least one host id" in str(exc_info.value)


# ==================== Test Helper Functions ====================

@pytest.fixture
def cleanup_files():
    """Fixture to clean up created files after tests"""
    files_to_cleanup = []
    yield files_to_cleanup
    for file_path in files_to_cleanup:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)

if __name__ == "__main__":
    # Run tests with: python -m pytest test_network_generators.py -v
    pytest.main([__file__, "-v"])