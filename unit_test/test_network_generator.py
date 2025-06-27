import sys
import os
import networkx as nx
import numpy as np
import pytest

curr_dir = os.path.dirname(__file__)
e3SIM_dir = os.path.join(curr_dir, '../e3SIM_codes')
if e3SIM_dir not in sys.path:
    sys.path.insert(0, e3SIM_dir)

from network_generator import write_network, read_input_network, run_network_generation

def test_write_network():
    g = nx.erdos_renyi_graph(100, 0.5)
    file_location = write_network(g, curr_dir)
    g_ = nx.read_adjlist(file_location)
    assert(nx.is_isomorphic(g, g_))
    os.remove(file_location)
  
def test_read_input_network():
    g = nx.erdos_renyi_graph(100, 0.5)
    file_location = write_network(g, curr_dir)
    g_ = read_input_network(file_location, 100)
    assert(nx.is_isomorphic(g, g_))
    os.remove(file_location)

def test_read_input_network_bad_path():
    with pytest.raises(Exception):
        g = read_input_network(curr_dir, 100)

    with pytest.raises(Exception):
        g = read_input_network("", 100)

    with pytest.raises(Exception):
        g = read_input_network("/abcdef/efg", 100)

def test_read_input_network_bad_popsize():
    with pytest.raises(Exception):
        g = nx.erdos_renyi_graph(100, 0.5)
        file_location = write_network(g, curr_dir)
        g_ = read_input_network(file_location, 50)
        os.remove(file_location)

def test_run_network_generation_bad_method():
    g, err = run_network_generation(
        pop_size=100,
        wk_dir=curr_dir,
        method="make_graph",
        model="ER",
        p_ER=0.1)
    assert(err != None)
    
def test_run_network_generation_success_ER():
    g, err = run_network_generation(
        pop_size=100,
        wk_dir=curr_dir,
        method="randomly_generate",
        model="ER",
        p_ER=0.1)

    assert(err == None)
    file_location = os.path.join(curr_dir, "contact_network.adjlist")
    g_ = nx.read_adjlist(file_location)
    assert(nx.is_isomorphic(g, g_))
    assert(g.number_of_nodes() == 100)
    # Number of edges within 4 standard deviations
    assert(g.number_of_edges() <= 495 + np.sqrt(4950 * 0.1 * 0.9)*4) # this is a rather generous upper bound for normal distribution
    os.remove(file_location)

def test_run_network_generation_success_BA():
    g, err = run_network_generation(
        pop_size=100,
        wk_dir=curr_dir,
        method="randomly_generate",
        model="BA",
        m=3)

    assert(err == None)
    file_location = os.path.join(curr_dir, "contact_network.adjlist")
    g_ = nx.read_adjlist(file_location)
    assert(nx.is_isomorphic(g, g_))
    assert(g.number_of_nodes() == 100)
    assert(g.number_of_edges() == 3*97) # star graph 3 edges + 96 * 3
    os.remove(file_location)

def test_run_network_generation_success_RP():
    g, err = run_network_generation(
        pop_size=100,
        wk_dir=curr_dir,
        method="randomly_generate",
        model="RP",
        rp_size = [50, 50],
        p_within = [0.1, 0.2],
        p_between = 0.01)

    assert(err == None)
    file_location = os.path.join(curr_dir, "contact_network.adjlist")
    g_ = nx.read_adjlist(file_location)
    assert(nx.is_isomorphic(g, g_))
    assert(g.number_of_nodes() == 100)
    # within 4 standard deviations
    assert(g.number_of_edges() <= (1225 * 0.3 + 2500 * 0.01) + 73) # independent binomial distributions
    os.remove(file_location)
