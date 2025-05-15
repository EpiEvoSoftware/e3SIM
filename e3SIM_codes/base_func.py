import json, os, argparse
from Bio import SeqIO
from error_handling import CustomizedError

def recursive_update(default, user):
    for k, v in user.items():
        if isinstance(v, dict) and k in default and isinstance(default[k], dict):
            recursive_update(default[k], v)
        else:
            default[k] = v

def read_params(path_config, default_config):
    """
    Reads configuration parameters from JSON files and merges with default parameters defined in he template dictionary.

    Parameters:
        path_config (str): Full path to configuration file.
        default_config (str): Relative path to the template (e.g., "base_params.json", "slim_only_template.json")
    """
    # Read default template
    default_config = open(os.path.join(os.path.dirname(__file__), "config_template", default_config), "r")
    default_param_dict = json.loads(default_config.read())
    # Read the user defined parameters
    config = open(path_config, "r")
    usr_param_dict = json.loads(config.read())
    # Update the template with user information
    # default_param_dict.update(usr_param_dict)
    recursive_update(default_param_dict, usr_param_dict)

    return default_param_dict

def read_params_simonly(path_config):
    # Remeber to get rid of
    default_config = open(os.path.join(os.path.dirname(__file__), "slim_only_template.json"), "r")
    pass


def str2bool(v):
    """
    Returns boolean value represented by the given str / bool.

    Parameters:
        v (str / bool): A string representing the boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower().replace(" ", "") in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower().replace(" ", "") in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def dump_json(json2dump):
    """
    Writes a JSON object to the SLiM config file.

    Parameters:
        json2dump: A dictionary rerpesenting the JSON object to be dumped.
    """
    out_config_path = os.path.join(json2dump["BasicRunConfiguration"]["cwdir"], "params.json")
    with open(out_config_path, 'w') as out_config:
        json.dump(json2dump, out_config, indent=2)

def check_config(config_dict):
    pass

def format_subst_mtx(mu_matrix, diag_zero=True):
    """
    From a string object of the mutation probability, to the list type probability rate matrix.

    Parameters:
        mu_matrix: A string object of the mutation probability, having the format '{"A":[,,,], "C":[,,,],"G":[,,,],"T":[,,,]}'.
    """
    default_matrix = [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]]
    try:
        mu_matrix = json.loads(mu_matrix)
    except json.decoder.JSONDecodeError:
        raise CustomizedError(f"The mutation matrix {mu_matrix} (-mu_matrix) is "
    		   "not a valid json format")
    alleles = ["A", "C", "G", "T"]
    from_allele = 0
    for allele in alleles:
        if allele not in mu_matrix:
            print(f"WARNING: The allele {allele} is not specified in the substitution rate matrix")
        elif len(mu_matrix[allele]) != 4:
            raise CustomizedError(f"The transition probability (from) each allele should be a list of 4, but the allele {allele} is not.")
        else:
            to_allele = 0
            for j in mu_matrix[allele]:
                if type(j) != float and type(j) != int:
                    raise CustomizedError(f"The provided substitution probability from {allele} contains non-floats, please provide a float for each probability")
                elif j<0 or j>1:
                    raise CustomizedError(f"The provided substitution probability from {allele} should be between 0 and 1")
                elif from_allele == to_allele and j!=0:
                    if diag_zero==True:
                        print(f"WARNING: The probability {allele}>{allele} should be zero following SLiM's notation, though non-zero value is provided, it will be ignored.")
                    to_allele = to_allele + 1
                else:
                    default_matrix[from_allele][to_allele] = j
                    to_allele = to_allele + 1
            if sum(default_matrix[from_allele])>1:
                raise CustomizedError(f"The sum of the provided substitution probability from {allele} should be smaller than 1, please check your entry.")
            if sum(default_matrix[from_allele])==0:
                print(f"WARNING: The sum of the provided substitution probability from {allele} is 0, meaning that allele \"{allele}\" will never mutate. Please check if this is not what you desired")
            if diag_zero==False:
                default_matrix[from_allele][from_allele] = 1 - sum(default_matrix[from_allele])
            from_allele = from_allele + 1
    return(default_matrix)



def check_ref_format(ref_path):
    ref_seq = SeqIO.parse(open(ref_path), 'fasta')
    seq_num = 0
    for fasta in ref_seq:
        seq_num = seq_num + 1
        name, sequence = fasta.id, str(fasta.seq).upper()
    if seq_num>1:
        raise CustomizedError("The reference genome file provided contains more than 1 sequences!")
    else:
        count_A = sequence.count("A")
        count_C = sequence.count("C")
        count_G = sequence.count("G")
        count_T = sequence.count("T")
        seq_len = len(sequence)
        print([count_A, count_C, count_G, count_T])
        if sum([count_A, count_C, count_G, count_T]) != seq_len:
            raise CustomizedError("The reference genome file provided contains characters that are not A, C, G, T!")
        return([count_A/seq_len, count_C/seq_len, count_G/seq_len, count_T/seq_len])









