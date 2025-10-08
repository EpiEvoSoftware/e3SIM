import pytest
import numpy as np
import networkx as nx
import json
import tempfile
import os, sys
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

curr_dir = os.path.dirname(__file__)
e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
if e3SIM_dir not in sys.path:
    sys.path.insert(0, e3SIM_dir)
    
# Assuming the main module is named seed_host_match.py
from seed_host_matcher import (
    NetworkPreprocessor,
    HostMatcher,
    SeedHostMatcher,
    FileManager,
    MatchingOrchestrator,
    CustomizedError
)


class TestNetworkPreprocessor:
    """Test suite for NetworkPreprocessor class."""
    
    @pytest.fixture
    def sample_network(self):
        """Create a sample network for testing."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 4)])
        return G
    
    def test_initialization(self, sample_network):
        """Test preprocessor initialization."""
        preprocessor = NetworkPreprocessor(sample_network)
        assert preprocessor.network == sample_network
        assert preprocessor._sorted_nodes is None
        assert preprocessor._sorted_degrees is None
    
    def test_get_sorted_nodes(self, sample_network):
        """Test getting nodes sorted by degree."""
        preprocessor = NetworkPreprocessor(sample_network)
        sorted_nodes = preprocessor.get_sorted_nodes()
        
        # Node degrees: 0->3, 2->3, 3->3, 1->2, 4->1
        expected_order = [0, 2, 3, 1, 4]  # Nodes sorted by degree (descending)
        np.testing.assert_array_equal(sorted_nodes, expected_order)
    
    def test_get_sorted_degrees(self, sample_network):
        """Test getting sorted degrees."""
        preprocessor = NetworkPreprocessor(sample_network)
        sorted_degrees = preprocessor.get_sorted_degrees()
        
        expected_degrees = [3, 3, 3, 2, 1]
        np.testing.assert_array_equal(sorted_degrees, expected_degrees)
    
    def test_lazy_loading(self, sample_network):
        """Test that sorting is computed only once (lazy loading)."""
        preprocessor = NetworkPreprocessor(sample_network)
        
        # First call should compute
        nodes1 = preprocessor.get_sorted_nodes()
        degrees1 = preprocessor.get_sorted_degrees()
        
        # Second call should return cached values
        nodes2 = preprocessor.get_sorted_nodes()
        degrees2 = preprocessor.get_sorted_degrees()
        
        assert nodes1 is nodes2
        assert degrees1 is degrees2


class TestHostMatcher:
    """Test suite for HostMatcher class."""
    
    @pytest.fixture
    def sample_preprocessor(self):
        """Create a sample preprocessor with mock data."""
        preprocessor = Mock(spec=NetworkPreprocessor)
        # Mock network with 10 nodes: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (sorted by degree)
        preprocessor.get_sorted_nodes.return_value = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) # but actually you can't have 10 edge in disease transmission
        preprocessor.get_sorted_degrees.return_value = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) # solely for testing purpose
        return preprocessor
    
    def test_match_random(self, sample_preprocessor):
        """Test random host matching."""
        matcher = HostMatcher(sample_preprocessor)
        
        # Match a random host
        host = matcher.match_random()
        assert host in sample_preprocessor.get_sorted_nodes()
        assert host in matcher.taken_hosts
        
        # Match another random host
        host2 = matcher.match_random()
        assert host2 != host
        assert host2 in matcher.taken_hosts
    
    def test_match_random_no_available_hosts(self, sample_preprocessor):
        """Test random matching when all hosts are taken."""
        matcher = HostMatcher(sample_preprocessor)
        
        # Take all hosts
        for node in sample_preprocessor.get_sorted_nodes():
            matcher.taken_hosts.add(node)
        
        with pytest.raises(CustomizedError, match="No available hosts"):
            matcher.match_random()
    
    def test_match_ranking_valid(self, sample_preprocessor):
        """Test ranking-based host matching with valid rank."""
        matcher = HostMatcher(sample_preprocessor)
        
        # Match to rank 1 (highest degree node)
        host = matcher.match_ranking(1)
        assert host == 9
        assert host in matcher.taken_hosts
        
        # Match to rank 5
        host = matcher.match_ranking(5)
        assert host == 5
        assert host in matcher.taken_hosts
    
    def test_match_ranking_invalid_rank(self, sample_preprocessor):
        """Test ranking with invalid ranks."""
        matcher = HostMatcher(sample_preprocessor)
        
        # Rank too high
        with pytest.raises(CustomizedError, match="exceeds network size"):
            matcher.match_ranking(11)
        
        # Rank too low
        with pytest.raises(CustomizedError, match="must be >= 1"):
            matcher.match_ranking(0)
        
        # Non-integer rank
        with pytest.raises(CustomizedError, match="must be integer"):
            matcher.match_ranking("5")
    
    def test_match_ranking_already_taken(self, sample_preprocessor):
        """Test ranking when host is already taken."""
        matcher = HostMatcher(sample_preprocessor)
        
        # Take rank 1
        matcher.match_ranking(1)
        
        # Try to take rank 1 again
        with pytest.raises(CustomizedError, match="already taken"):
            matcher.match_ranking(1)
    
    def test_match_percentile_valid(self, sample_preprocessor):
        """Test percentile-based host matching."""
        matcher = HostMatcher(sample_preprocessor)
        
        # Match to top 50% (percentile 0-50)
        host = matcher.match_percentile([0, 50])
        expected_hosts = [9, 8, 7, 6, 5]  # Top 50% of 10 nodes
        assert host in expected_hosts
        assert host in matcher.taken_hosts
    
    def test_match_percentile_invalid_format(self, sample_preprocessor):
        """Test percentile with invalid format."""
        matcher = HostMatcher(sample_preprocessor)
        
        # Not a list
        with pytest.raises(CustomizedError, match="must be list of 2 integers"):
            matcher.match_percentile(50)
        
        # Wrong length
        with pytest.raises(CustomizedError, match="must be list of 2 integers"):
            matcher.match_percentile([0, 50, 100])
        
        # Non-integer values
        with pytest.raises(CustomizedError, match="must be integers"):
            matcher.match_percentile([0.5, 50.5])
    
    def test_match_percentile_invalid_range(self, sample_preprocessor):
        """Test percentile with invalid range."""
        matcher = HostMatcher(sample_preprocessor)
        
        # Out of bounds
        with pytest.raises(CustomizedError, match="must be between 0 and 100"):
            matcher.match_percentile([-10, 50])
        
        with pytest.raises(CustomizedError, match="must be between 0 and 100"):
            matcher.match_percentile([50, 110])
        
        # Invalid range (high <= low)
        with pytest.raises(CustomizedError, match="Invalid percentile range"):
            matcher.match_percentile([50, 50])
        
        with pytest.raises(CustomizedError, match="Invalid percentile range"):
            matcher.match_percentile([60, 40])
    
    def test_get_available_hosts(self, sample_preprocessor):
        """Test getting available hosts."""
        matcher = HostMatcher(sample_preprocessor)
        all_nodes = sample_preprocessor.get_sorted_nodes()
        
        # All hosts available initially
        available = matcher._get_available_hosts(all_nodes)
        np.testing.assert_array_equal(available, all_nodes)
        
        # Take some hosts
        matcher.taken_hosts = {9, 7, 5}
        available = matcher._get_available_hosts(all_nodes)
        expected = np.array([8, 6, 4, 3, 2, 1, 0])
        np.testing.assert_array_equal(sorted(available), sorted(expected))


class TestSeedHostMatcher:
    """Test suite for SeedHostMatcher class."""
    
    @pytest.fixture
    def sample_network(self):
        """Create a sample network."""
        G = nx.Graph()
        G.add_edges_from([(i, j) for i in range(5) for j in range(i+1, 5)])
        return G
    
    def test_initialization(self, sample_network):
        """Test SeedHostMatcher initialization."""
        matcher = SeedHostMatcher(sample_network)
        assert matcher.network == sample_network
        assert isinstance(matcher.preprocessor, NetworkPreprocessor)
        assert isinstance(matcher.matcher, HostMatcher)
    
    def test_match_all_seeds_random(self, sample_network):
        """Test matching all seeds with random method."""
        matcher = SeedHostMatcher(sample_network)
        
        match_methods = {"0": "random", "1": "random", "2": "random"}
        match_params = {"0": None, "1": None, "2": None}
        
        result = matcher.match_all_seeds(match_methods, match_params, 3)
        
        assert len(result) == 3
        assert all(seed in result for seed in [0, 1, 2])
        assert len(set(result.values())) == 3  # All hosts unique
    
    def test_match_all_seeds_ranking(self, sample_network):
        """Test matching with ranking method."""
        matcher = SeedHostMatcher(sample_network)
        
        match_methods = {"0": "ranking", "1": "ranking"}
        match_params = {"0": 1, "1": 2}
        
        result = matcher.match_all_seeds(match_methods, match_params, 2)
        
        assert len(result) == 2
        # In complete graph, all nodes have same degree, so ranking is arbitrary
        assert len(set(result.values())) == 2
    
    def test_match_all_seeds_percentile(self, sample_network):
        """Test matching with percentile method."""
        matcher = SeedHostMatcher(sample_network)
        
        match_methods = {"0": "percentile"}
        match_params = {"0": [0, 100]}
        
        result = matcher.match_all_seeds(match_methods, match_params, 1)
        
        assert len(result) == 1
        assert 0 in result
    
    def test_match_all_seeds_mixed_methods(self, sample_network):
        """Test matching with mixed methods."""
        matcher = SeedHostMatcher(sample_network)
        
        match_methods = {"0": "ranking", "1": "percentile", "2": "random"}
        match_params = {"0": 1, "1": [0, 50], "2": None}
        
        result = matcher.match_all_seeds(match_methods, match_params, 3)
        
        assert len(result) == 3
        assert len(set(result.values())) == 3
    
    def test_match_all_seeds_too_many(self, sample_network):
        """Test matching more seeds than network size."""
        matcher = SeedHostMatcher(sample_network)
        
        match_methods = {str(i): "random" for i in range(10)}
        match_params = {str(i): None for i in range(10)}
        
        with pytest.raises(CustomizedError, match="Cannot match"):
            matcher.match_all_seeds(match_methods, match_params, 10)
    
    def test_match_single_seed_unknown_method(self, sample_network):
        """Test matching with unknown method."""
        matcher = SeedHostMatcher(sample_network)
        
        with pytest.raises(CustomizedError, match="Unknown method"):
            matcher._match_single_seed("unknown_method", None)
    
    def test_group_seeds_by_method(self, sample_network):
        """Test grouping seeds by method."""
        matcher = SeedHostMatcher(sample_network)
        
        match_methods = {
            "0": "random",
            "1": "ranking",
            "2": "random",
            "3": "percentile",
            "4": "ranking"
        }
        
        groups = matcher._group_seeds_by_method(match_methods, 5)
        
        assert groups["random"] == [0, 2]
        assert groups["ranking"] == [1, 4]
        assert groups["percentile"] == [3]


class TestFileManager:
    """Test suite for FileManager class."""
    
    def test_save_matching_csv(self, tmp_path):
        """Test saving matching to CSV."""
        matching = {0: 5, 1: 3, 2: 7, 3: 1}
        file_path = tmp_path / "test_match.csv"
        
        FileManager.save_matching_csv(matching, str(file_path))
        
        # Read back and verify
        df = pd.read_csv(file_path)
        assert len(df) == 4
        assert list(df.columns) == ['seed', 'host_id']
        # Should be sorted by host_id
        assert list(df['host_id']) == [1, 3, 5, 7]
        assert list(df['seed']) == [3, 1, 0, 2]
    
    def test_read_network(self, tmp_path):
        """Test reading network from file."""
        # Create a test network file
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        network_path = tmp_path / "test_network.adjlist"
        nx.write_adjlist(G, str(network_path))
        
        # Read back
        loaded_network = FileManager.read_network(str(network_path))
        assert loaded_network.number_of_nodes() == 3
        assert loaded_network.number_of_edges() == 2
    
    def test_read_network_not_found(self):
        """Test reading non-existent network file."""
        with pytest.raises(CustomizedError, match="not found"):
            FileManager.read_network("non_existent.adjlist")
    
    def test_read_user_matching_json(self, tmp_path):
        """Test reading user matching from JSON."""
        matching = {0: 5, 1: 3, 2: 7}
        file_path = tmp_path / "test_match.json"
        
        with open(file_path, 'w') as f:
            json.dump(matching, f)
        
        result = FileManager.read_user_matching_file(str(file_path))
        assert result == matching
    
    def test_read_user_matching_csv(self, tmp_path):
        """Test reading user matching from CSV."""
        df = pd.DataFrame({'seed': [0, 1, 2], 'host_id': [5, 3, 7]})
        file_path = tmp_path / "test_match.csv"
        df.to_csv(file_path, index=False)
        
        result = FileManager.read_user_matching_file(str(file_path))
        assert result == {0: 5, 1: 3, 2: 7}
    
    def test_read_user_matching_invalid_format(self, tmp_path):
        """Test reading matching with invalid format."""
        file_path = tmp_path / "test_match.txt"
        file_path.write_text("invalid content")
        
        with pytest.raises(CustomizedError, match="must be JSON or CSV"):
            FileManager.read_user_matching_file(str(file_path))
    
    def test_read_user_matching_not_found(self):
        """Test reading non-existent matching file."""
        with pytest.raises(CustomizedError, match="not found"):
            FileManager.read_user_matching_file("non_existent.json")
    
    def test_read_user_matching_invalid_json(self, tmp_path):
        """Test reading invalid JSON file."""
        file_path = tmp_path / "bad.json"
        file_path.write_text("{invalid json}")
        
        with pytest.raises(CustomizedError, match="Invalid JSON"):
            FileManager.read_user_matching_file(str(file_path))


class TestMatchingOrchestrator:
    """Test suite for MatchingOrchestrator class."""
    
    @pytest.fixture
    def temp_working_dir(self, tmp_path):
        """Create a temporary working directory with a network file."""
        G = nx.Graph()
        G.add_edges_from([(i, j) for i in range(5) for j in range(i+1, 5)])
        network_path = tmp_path / "contact_network.adjlist"
        nx.write_adjlist(G, str(network_path))
        return str(tmp_path)
    
    def test_initialization(self, temp_working_dir):
        """Test orchestrator initialization."""
        orchestrator = MatchingOrchestrator(temp_working_dir, random_seed=42)
        assert orchestrator.working_dir == temp_working_dir
    
    def test_run_matching_user_input(self, temp_working_dir, tmp_path):
        """Test running with user input method."""
        # Create user matching file
        matching = {0: 3, 1: 2, 2: 4}
        match_file = tmp_path / "user_match.json"
        with open(match_file, 'w') as f:
            json.dump(matching, f)
        
        orchestrator = MatchingOrchestrator(temp_working_dir)
        result, error = orchestrator.run_matching(
            method="user_input",
            num_seeds=3,
            path_matching=str(match_file)
        )
        
        assert error is None
        assert result == matching
    
    def test_run_matching_user_input_no_path(self, temp_working_dir):
        """Test user input without path."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        result, error = orchestrator.run_matching(
            method="user_input",
            num_seeds=3,
            path_matching=""
        )
        
        assert result is None
        assert isinstance(error, CustomizedError)
        assert "Path to matching file required" in str(error)
    
    def test_run_matching_randomly_generate(self, temp_working_dir):
        """Test running with random generation."""
        orchestrator = MatchingOrchestrator(temp_working_dir, random_seed=42)
        result, error = orchestrator.run_matching(
            method="randomly_generate",
            num_seeds=3,
            match_scheme="",
            match_scheme_param=""
        )
        
        assert error is None
        assert len(result) == 3
        assert all(i in result for i in [0, 1, 2])
        
        # Check CSV was created
        csv_path = Path(temp_working_dir) / "seed_host_match.csv"
        assert csv_path.exists()
    
    def test_run_matching_invalid_method(self, temp_working_dir):
        """Test running with invalid method."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        result, error = orchestrator.run_matching(
            method="invalid_method",
            num_seeds=3
        )
        
        assert result is None
        assert isinstance(error, CustomizedError)
        assert "Invalid method" in str(error)
    
    def test_parse_match_scheme_dict(self, temp_working_dir):
        """Test parsing match scheme from dict."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        scheme = {"0": "random", "1": "ranking"}
        
        result = orchestrator._parse_match_scheme(scheme, 2)
        assert result == scheme
    
    def test_parse_match_scheme_empty(self, temp_working_dir):
        """Test parsing empty match scheme."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        
        result = orchestrator._parse_match_scheme("", 3)
        assert result == {"0": "random", "1": "random", "2": "random"}
    
    def test_parse_match_scheme_json_string(self, temp_working_dir):
        """Test parsing match scheme from JSON string."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        scheme_str = '{"0": "ranking", "1": "percentile"}'
        
        result = orchestrator._parse_match_scheme(scheme_str, 2)
        assert result == {"0": "ranking", "1": "percentile"}
    
    def test_parse_match_scheme_invalid_json(self, temp_working_dir):
        """Test parsing invalid JSON match scheme."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        
        with pytest.raises(CustomizedError, match="Invalid JSON"):
            orchestrator._parse_match_scheme("{invalid}", 2)
    
    def test_parse_match_params_dict(self, temp_working_dir):
        """Test parsing match params from dict."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        params = {"0": None, "1": 1}
        
        result = orchestrator._parse_match_params(params, {"0": "random", "1": "ranking"}, 2)
        assert result == params
    
    def test_parse_match_params_empty(self, temp_working_dir):
        """Test parsing empty match params."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        
        result = orchestrator._parse_match_params("", {"0": "random"}, 1)
        assert result == {"0": None}
    
    def test_parse_match_params_json_string(self, temp_working_dir):
        """Test parsing match params from JSON string."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        params_str = '{"0": 1, "1": [0, 50]}'
        
        result = orchestrator._parse_match_params(
            params_str, 
            {"0": "ranking", "1": "percentile"}, 
            2
        )
        assert result == {"0": 1, "1": [0, 50]}
    
    def test_parse_match_params_invalid_json_all_random(self, temp_working_dir):
        """Test parsing invalid JSON params when all methods are random."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        
        # Should default to None params when all random
        result = orchestrator._parse_match_params(
            "{invalid}", 
            {"0": "random", "1": "random"}, 
            2
        )
        assert result == {"0": None, "1": None}
    
    def test_parse_match_params_invalid_json_not_all_random(self, temp_working_dir):
        """Test parsing invalid JSON params when not all methods are random."""
        orchestrator = MatchingOrchestrator(temp_working_dir)
        
        with pytest.raises(CustomizedError, match="Invalid JSON"):
            orchestrator._parse_match_params(
                "{invalid}", 
                {"0": "ranking", "1": "random"}, 
                2
            )


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def setup_environment(self, tmp_path):
        """Set up a complete testing environment."""
        # Create network
        G = nx.Graph()
        edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5), 
                 (5, 0), (1, 3), (2, 4), (3, 5)]
        G.add_edges_from(edges)
        
        # Save network
        network_path = tmp_path / "contact_network.adjlist"
        nx.write_adjlist(G, str(network_path))
        
        return str(tmp_path)
    
    def test_complete_random_workflow(self, setup_environment):
        """Test complete workflow with random matching."""
        orchestrator = MatchingOrchestrator(setup_environment, random_seed=123)
        
        result, error = orchestrator.run_matching(
            method="randomly_generate",
            num_seeds=4,
            match_scheme='{"0": "random", "1": "random", "2": "random", "3": "random"}',
            match_scheme_param='{"0": null, "1": null, "2": null, "3": null}'
        )
        
        assert error is None
        assert len(result) == 4
        assert len(set(result.values())) == 4  # All unique hosts
        
        # Verify CSV file was created
        csv_path = Path(setup_environment) / "seed_host_match.csv"
        assert csv_path.exists()
        
        df = pd.read_csv(csv_path)
        assert len(df) == 4
        assert set(df['seed']) == {0, 1, 2, 3}
    
    def test_complete_mixed_methods_workflow(self, setup_environment):
        """Test complete workflow with mixed matching methods."""
        orchestrator = MatchingOrchestrator(setup_environment, random_seed=456)
        
        match_scheme = {
            "0": "ranking",
            "1": "percentile",
            "2": "random"
        }
        match_params = {
            "0": 1,
            "1": [0, 50],
            "2": None
        }
        
        result, error = orchestrator.run_matching(
            method="randomly_generate",
            num_seeds=3,
            match_scheme=match_scheme,
            match_scheme_param=match_params
        )
        
        assert error is None
        assert len(result) == 3
        assert len(set(result.values())) == 3
    
    def test_error_handling_in_workflow(self, setup_environment):
        """Test error handling in the complete workflow."""
        orchestrator = MatchingOrchestrator(setup_environment)
        
        # Try to match more seeds than nodes
        result, error = orchestrator.run_matching(
            method="randomly_generate",
            num_seeds=10,  # Network only has 6 nodes
            match_scheme="",
            match_scheme_param=""
        )
        
        assert result is None
        assert error is not None
        assert "Cannot match" in str(error)


class TestBackwardCompatibility:
    """Test backward compatibility functions."""
    
    @patch('e3SIM_codes.seed_host_matcher.SeedHostMatcher')
    def test_match_all_hosts(self, mock_matcher_class):
        """Test backward compatible match_all_hosts function."""
        from e3SIM_codes.seed_host_matcher import match_all_hosts
        
        mock_network = Mock()
        mock_matcher = Mock()
        mock_matcher_class.return_value = mock_matcher
        mock_matcher.match_all_seeds.return_value = {0: 3, 1: 2}
        
        result = match_all_hosts(
            mock_network,
            {"0": "random"},
            {"0": None},
            1
        )
        
        mock_matcher_class.assert_called_once_with(mock_network)
        mock_matcher.match_all_seeds.assert_called_once()
        assert result == {0: 3, 1: 2}
    
    @patch('e3SIM_codes.seed_host_matcher.FileManager')
    def test_write_match(self, mock_file_manager):
        """Test backward compatible write_match function."""
        from e3SIM_codes.seed_host_matcher import write_match
        
        match_dict = {0: 3, 1: 2}
        wk_dir = "/test/dir"
        
        result = write_match(match_dict, wk_dir)
        
        expected_path = os.path.join(wk_dir, "seed_host_match.csv")
        mock_file_manager.save_matching_csv.assert_called_once_with(match_dict, expected_path)
        assert result == expected_path
    
    @patch('e3SIM_codes.seed_host_matcher.FileManager')
    def test_read_network_compat(self, mock_file_manager):
        """Test backward compatible read_network function."""
        from e3SIM_codes.seed_host_matcher import read_network
        
        mock_network = Mock()
        mock_file_manager.read_network.return_value = mock_network
        
        result = read_network("test.adjlist")
        
        mock_file_manager.read_network.assert_called_once_with("test.adjlist")
        assert result == mock_network
    
    @patch('e3SIM_codes.seed_host_matcher.FileManager')
    def test_read_user_matchingfile(self, mock_file_manager):
        """Test backward compatible read_user_matchingfile function."""
        from e3SIM_codes.seed_host_matcher import read_user_matchingfile
        
        mock_matching = {0: 3}
        mock_file_manager.read_user_matching_file.return_value = mock_matching
        
        result = read_user_matchingfile("test.json")
        
        mock_file_manager.read_user_matching_file.assert_called_once_with("test.json")
        assert result == mock_matching

if __name__ == "__main__":
    # Run tests with: python -m pytest test_seed_host_matcher.py -v
    pytest.main([__file__, "-v"])