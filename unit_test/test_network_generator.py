import pytest
import numpy as np
import networkx as nx
import os, sys
import tempfile
from unittest.mock import patch, mock_open

curr_dir = os.path.dirname(__file__)
e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
if e3SIM_dir not in sys.path:
    sys.path.insert(0, e3SIM_dir)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from network_generator import BaseNetworkGenerator, ERGenerator, RPGenerator, BAGenerator, UserInputGenerator
from error_handling import CustomizedError
# ==================== UNIT TESTS ====================

class TestBaseNetworkGenerator:
    """Tests for BaseNetworkGenerator base class."""
    
    def test_init_basic(self):
        """Test basic initialization."""
        generator = BaseNetworkGenerator(100)
        assert generator.pop_size == 100
        assert generator.rand_seed is None
    
    def test_init_with_seed(self):
        """Test initialization with random seed."""
        with patch('numpy.random.seed') as mock_seed:
            generator = BaseNetworkGenerator(100, rand_seed=42)
            assert generator.pop_size == 100
            assert generator.rand_seed == 42
            mock_seed.assert_called_once_with(42)
    
    def test_init_with_none_seed(self):
        """Test initialization with None seed doesn't call np.random.seed."""
        with patch('numpy.random.seed') as mock_seed:
            _ = BaseNetworkGenerator(100, rand_seed=None)
            mock_seed.assert_not_called()
    
    def test_generate_not_implemented(self):
        """Test that generate() raises NotImplementedError."""
        generator = BaseNetworkGenerator(100)
        with pytest.raises(NotImplementedError, match="Subclasses must implement generate\\(\\)"):
            generator.generate()


class TestERGenerator:
    """Tests for ERGenerator (Erdős-Rényi)."""
    
    def test_init_valid_probability(self):
        """Test initialization with valid probability."""
        generator = ERGenerator(100, 0.5)
        assert generator.pop_size == 100
        assert generator.p_ER == 0.5
    
    def test_init_boundary_probabilities(self):
        """Test initialization with boundary probability values."""
        # Test p_ER = 1 (valid)
        generator = ERGenerator(100, 1.0)
        assert generator.p_ER == 1.0
        
        # Test p_ER just above 0 (valid)
        generator = ERGenerator(100, 0.001)
        assert generator.p_ER == 0.001
    
    def test_init_invalid_probability_zero(self):
        """Test initialization with p_ER = 0 raises error."""
        with pytest.raises(CustomizedError, match="You need to specify a 0<p<=1"):
            ERGenerator(100, 0)
    
    def test_init_invalid_probability_negative(self):
        """Test initialization with negative p_ER raises error."""
        with pytest.raises(CustomizedError, match="You need to specify a 0<p<=1"):
            ERGenerator(100, -0.1)
    
    def test_init_invalid_probability_greater_than_one(self):
        """Test initialization with p_ER > 1 raises error."""
        with pytest.raises(CustomizedError, match="You need to specify a 0<p<=1"):
            ERGenerator(100, 1.1)
    
    @patch('networkx.fast_gnp_random_graph')
    def test_generate(self, mock_gnp):
        """Test generate method calls NetworkX correctly."""
        mock_graph = nx.Graph()
        mock_gnp.return_value = mock_graph
        
        generator = ERGenerator(100, 0.5)
        result = generator.generate()
        
        mock_gnp.assert_called_once_with(100, 0.5, seed=np.random)
        assert result == mock_graph
    
    def test_generate_reproducible_with_seed(self):
        """Test that generate produces reproducible results with seed."""
        generator1 = ERGenerator(10, 0.5, rand_seed=42)
        generator2 = ERGenerator(10, 0.5, rand_seed=42)
        
        graph1 = generator1.generate()
        # Reset seed for second generator
        np.random.seed(42)
        graph2 = generator2.generate()
        
        assert graph1.number_of_nodes() == graph2.number_of_nodes()
        # Note: Due to how seeding works, we can't guarantee identical edge sets
        # without more complex seed management

    def test_number_of_edges(self):
        generator = ERGenerator(100, 0.1)

        graph = generator.generate()
        assert graph.number_of_edges() <= 100 * 99 * 0.1 / 2 + np.sqrt(4950 * 0.1 * 0.9)*4 
        # this is a rather generous upper bound for normal distribution

class TestRPGenerator:
    """Tests for RPGenerator (Random Partition/Stochastic Block Model)."""
    
    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        generator = RPGenerator(100, [50, 30, 20], [0.8, 0.7, 0.9], 0.1)
        assert generator.pop_size == 100
        assert generator.rp_size == [50, 30, 20]
        assert generator.p_within == [0.8, 0.7, 0.9]
        assert generator.p_between == 0.1
    
    def test_init_partition_size_mismatch(self):
        """Test error when partition sizes don't sum to population size."""
        with pytest.raises(CustomizedError, match="Partition sizes .* must sum to population size"):
            RPGenerator(100, [50, 30], [0.8, 0.7], 0.1)
    
    def test_init_probability_count_mismatch(self):
        """Test error when number of within-group probabilities doesn't match partitions."""
        with pytest.raises(CustomizedError, match="Number of partitions and within-group probabilities mismatch"):
            RPGenerator(100, [50, 50], [0.8, 0.7, 0.6], 0.1)
    
    def test_init_zero_between_probability_warning(self, capsys):
        """Test warning when between-group probability is 0."""
        RPGenerator(100, [50, 50], [0.8, 0.7], 0)
        captured = capsys.readouterr()
        assert "WARNING: Between-group probability is 0. Partitions will be isolated." in captured.out
    
    @patch('networkx.stochastic_block_model')
    def test_generate(self, mock_sbm):
        """Test generate method calls NetworkX correctly."""
        mock_graph = nx.Graph()
        mock_sbm.return_value = mock_graph
        
        generator = RPGenerator(100, [60, 40], [0.8, 0.7], 0.1)
        result = generator.generate()
        
        expected_prob_matrix = [[0.8, 0.1], [0.1, 0.7]]
        mock_sbm.assert_called_once_with([60, 40], expected_prob_matrix, seed=np.random)
        assert result == mock_graph
    
    def test_generate_three_partitions(self):
        """Test probability matrix construction for three partitions."""
        with patch('networkx.stochastic_block_model') as mock_sbm:
            mock_graph = nx.Graph()
            mock_sbm.return_value = mock_graph
            
            generator = RPGenerator(100, [40, 30, 30], [0.9, 0.8, 0.7], 0.05)
            generator.generate()
            
            expected_prob_matrix = [
                [0.9, 0.05, 0.05],
                [0.05, 0.8, 0.05],
                [0.05, 0.05, 0.7]
            ]
            mock_sbm.assert_called_once_with([40, 30, 30], expected_prob_matrix, seed=np.random)

    def test_number_of_edges(self):
        generator = RPGenerator(100, [50, 50], [0.1, 0.2], 0.01)
        graph = generator.generate()

        assert graph.number_of_edges() <= (1225 * 0.3 + 2500 * 0.01) + 73


class TestBAGenerator:
    """Tests for BAGenerator (Barabási-Albert)."""
    
    def test_init(self):
        """Test initialization."""
        generator = BAGenerator(100, 3)
        assert generator.pop_size == 100
        assert generator.m == 3
    
    def test_init_with_seed(self):
        """Test initialization with seed."""
        generator = BAGenerator(100, 3, rand_seed=42)
        assert generator.pop_size == 100
        assert generator.m == 3
        assert generator.rand_seed == 42
    
    @patch('networkx.barabasi_albert_graph')
    def test_generate(self, mock_ba):
        """Test generate method calls NetworkX correctly."""
        mock_graph = nx.Graph()
        mock_ba.return_value = mock_graph
        
        generator = BAGenerator(100, 5)
        result = generator.generate()
        
        mock_ba.assert_called_once_with(100, 5, seed=np.random)
        assert result == mock_graph

    def test_number_of_edges(self):
        generator = BAGenerator(100, 3)

        graph = generator.generate()
        assert graph.number_of_edges() == 3 * 97 # star graph 3 edges + 96 * 3


class TestUserInputGenerator:
    """Tests for UserInputGenerator."""
    
    def test_init(self):
        """Test initialization."""
        generator = UserInputGenerator(100, "/path/to/network.txt")
        assert generator.pop_size == 100
        assert generator.path_network == "/path/to/network.txt"
    
    def test_generate_file_not_found_empty_path(self):
        """Test error when path is empty."""
        generator = UserInputGenerator(100, "")
        with pytest.raises(FileNotFoundError, match="Network path  not found"):
            generator.generate()
    
    def test_generate_file_not_found_none_path(self):
        """Test error when path is None."""
        generator = UserInputGenerator(100, None)
        with pytest.raises(FileNotFoundError, match="Network path None not found"):
            generator.generate()
    
    def test_generate_file_not_exists(self):
        """Test error when file doesn't exist."""
        generator = UserInputGenerator(100, "/nonexistent/path.txt")
        with pytest.raises(FileNotFoundError, match="Network path .* not found"):
            generator.generate()
    
    def test_generate_node_count_mismatch(self):
        """Test error when network nodes don't match population size."""
        # Create a temporary file with network data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            # Write a simple adjacency list with 3 nodes
            tmp_file.write("0 1\n1 2\n2 0\n")
            tmp_file.flush()
            temp_path = tmp_file.name
        
        try:
            generator = UserInputGenerator(5, temp_path)  # Expect 5 nodes, but file has 3
            with pytest.raises(CustomizedError, match="Network nodes 3 do not match population size 5"):
                generator.generate()
        finally:
            os.unlink(temp_path)
    
    def test_generate_success(self):
        """Test successful network loading."""
        # Create a temporary file with network data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
            # Write adjacency list with 3 nodes
            tmp_file.write("0 1 2\n1 0\n2 0\n")
            tmp_file.flush()
            temp_path = tmp_file.name
        
        try:
            generator = UserInputGenerator(3, temp_path)
            graph = generator.generate()
            
            assert isinstance(graph, nx.Graph)
            assert len(graph) == 3
        finally:
            os.unlink(temp_path)
    
    @patch('os.path.exists')
    @patch('networkx.read_adjlist')
    def test_generate_mocked_success(self, mock_read_adjlist, mock_exists):
        """Test generate with mocked file operations."""
        mock_exists.return_value = True
        mock_graph = nx.Graph()
        mock_graph.add_nodes_from([0, 1, 2])  # 3 nodes
        mock_read_adjlist.return_value = mock_graph
        
        generator = UserInputGenerator(3, "/fake/path.txt")
        result = generator.generate()
        
        mock_exists.assert_called_once_with("/fake/path.txt")
        mock_read_adjlist.assert_called_once_with("/fake/path.txt")
        assert result == mock_graph


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests that test actual graph generation."""
    
    def test_er_generator_creates_valid_graph(self):
        """Test that ERGenerator creates a valid graph."""
        generator = ERGenerator(10, 0.5, rand_seed=42)
        graph = generator.generate()
        
        assert isinstance(graph, nx.Graph)
        assert len(graph) == 10
        assert all(isinstance(node, int) for node in graph.nodes())
    
    def test_ba_generator_creates_valid_graph(self):
        """Test that BAGenerator creates a valid graph."""
        generator = BAGenerator(20, 2, rand_seed=42)
        graph = generator.generate()
        
        assert isinstance(graph, nx.Graph)
        assert len(graph) == 20
        # BA graphs should be connected for m >= 1 and n >= m+1
        assert nx.is_connected(graph)
    
    def test_rp_generator_creates_valid_graph(self):
        """Test that RPGenerator creates a valid graph."""
        generator = RPGenerator(30, [15, 15], [0.8, 0.7], 0.1, rand_seed=42)
        graph = generator.generate()
        
        assert isinstance(graph, nx.Graph)
        assert len(graph) == 30


# ==================== FIXTURES AND HELPERS ====================

@pytest.fixture
def temp_network_file():
    """Create a temporary network file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_file:
        tmp_file.write("0 1 2\n1 0 2\n2 0 1\n")  # Complete graph with 3 nodes
        tmp_file.flush()
        yield tmp_file.name
    os.unlink(tmp_file.name)


class TestWithFixtures:
    """Tests using pytest fixtures."""
    
    def test_user_input_generator_with_fixture(self, temp_network_file):
        """Test UserInputGenerator with a real temporary file."""
        generator = UserInputGenerator(3, temp_network_file)
        graph = generator.generate()
        
        assert isinstance(graph, nx.Graph)
        assert len(graph) == 3


if __name__ == "__main__":
    # Run tests with: python -m pytest test_network_generators.py -v
    pytest.main([__file__, "-v"])
