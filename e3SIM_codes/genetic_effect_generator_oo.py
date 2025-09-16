from base_func import *
from error_handling import CustomizedError
import numpy as np
import ast
import pandas as pd
import scipy
from scipy.stats import multivariate_t
from mv_laplace import MvLaplaceSampler
import traceback
import argparse, statistics, os, json
from Bio import SeqIO

START_IDX = 0
END_IDX = 1
E_SIZE = 2

GFF_START = 3
GFF_END = 4
GFF_INFO = 8


################## REWRITE ########################

# Logit and inverse logit functions
################### These might be used for the calibration but not currently
def logit(p):
    return scipy.stats.logit(p)

def inv_logit(p):
    return scipy.stats.expit(p)


def logit_link(default_val, alpha, trait_val):
    """
    Returns the probability of transmission or survival based on default values 
    and trait values using the logit link

    Parameters:
        default_val (float): Base values (beta for transmissibility and s for survival)
        alpha (float): slope on the link scale
        trait_val (float): Trait values on the link scale
    """
    # CHANGED, originally different from the supplementary specification
    return inv_logit(logit(default_val) + alpha * trait_val)



def trans_loglog_link(beta, alpha, trait_val):
    """
    Returns the probability of transmission based on base transmissibility and 
    trait values using the log-log link

    Parameters:
        beta (float): Base transmissibiliy
        alpha (float): slope on the link scale
        trait_val (float): Trait values on the link scale
    """
    return 1 - np.exp(np.log(1 - beta) * np.exp(alpha * trait_val))


def surviv_loglog_link(s, alpha, trait_val):
    """
    Returns the probability of transmission based on base transmissibility and trait values
    using the log-log link

    Parameters:
        s (float): Base survival probability in treatment
        alpha (float): slope on the link scale
        trait_val (float): Trait values on the link scale
    """
    return np.exp(np.log(s) * np.exp(-alpha * trait_val))


################### END.


######## THESE ARE USED!

# ------------- Genetic Effect Configuration ----------------------- #
class GeneticEffectConfig:
    def __init__(self, method, wk_dir, n_seed, func, calibration,pleiotropy, trait_num, random_seed, pis, **kwargs):
        self.method = method # # 'gff' | 'csv'
        self.wk_dir = wk_dir 
        self.n_seed = n_seed # number of seeds to generate
        self.calibration = calibration # whether to do calibration
        self.trait_num = trait_num # number of traits
        self.func = func # n | st | l
        self.pleiotropy = pleiotropy
        self.random_seed = random_seed
        self.pis = pis
        self.params = kwargs # gff, csv, cov, nv, bs, redo, taus, s, s_cov
        # self._validate()

    def validate(self):
        if not os.path.exists(self.wk_dir):
            raise CustomizedError(f"Working directory {self.wk_dir} does not exist.")
        if self.method not in ("csv", "gff"):
            raise CustomizedError(f"{self.method} isn't a valid method. Please provide a permitted method. "
                            "(csv/gff)")
        if self.n_seed <= 0:
            raise CustomizedError("Seed size must be positive.")
        if len(self.trait_num.keys()) != 2:
            raise CustomizedError("Please specify exactly 2 traits quantities in a list (-trait_n for transmissibility and drug resistance)")
        if sum(self.trait_num.values()) < 1:
            raise CustomizedError("Please provide a list of trait quantities (-trait_n) that sums up to at least 1")
        if self.func not in ("n", "l", "st"):
            raise CustomizedError(f"{self.func} isn't a valid method for sampling traits. Please choose a permitted method."
                                "(n/l/st)")
        if not self.pleiotropy:
            if self.func == "n" and len(self.params.get("taus")) != sum(self.trait_num.values()):
                raise CustomizedError(f"The given length of the variance (-taus) {self.params.get("taus")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the point normal mode.")
            if self.func == "l" and len(self.params.get("bs")) != sum(self.trait_num.values()):
                raise CustomizedError(f"The given length of the scales (-bs) {self.params.get("bs")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the laplace mode.")
            if self.func == "st" and len(self.params.get("s")) != sum(self.trait_num.values()):
                raise CustomizedError(f"The given length of the scales (-s) {self.params.get("s")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the laplace mode.")

class Mapper:
    def __init__(self, **kwargs):
        raise NotImplementedError
        # TO DO: Implement the mapping genetic values to event probabilities mapper, 
        # but maybe this will be part of outbreak simulation given original logic?

class Calibrator:
    def __init__(self, **kwargs):
        raise NotImplementedError
        # To Do: Implement

# ------------- Genetic Effect Generation ----------------------- #
class EffectGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        np.random.seed(cfg.random_seed)

    def run(self):
        try:
            self.cfg.validate()
            df_eff = self._build_effect_df()
            seeds = self._compute_seed_traits(df_eff)
            if self.cfg.calibration:
                df_eff = self._calibrate(df_eff, seeds)
            df_eff_named = self._rename_columns(df_eff)
            seeds_named = self._rename_columns(seeds)
            self._write_outputs(df_eff_named, seeds_named)
            print("********************************************************************\n"
                "                  GENETIC ARCHITECTURES GENERATED\n"
                "********************************************************************",
                flush=True)
            return None
        except Exception as e:
            print(f"Genetic effects generation - An error occured: {e}.", flush = True)
            traceback.print_exc()
            error_message = e
            return error_message

        
    # ---------- Build effect df ----------
    def _build_effect_df(self) -> pd.DataFrame:
        if self.cfg.method == "gff":
            candidates = self._read_gff_sites()
            df_sites = self._select_sites(candidates, self.cfg.pis)
            return self._sample(df_sites)
        elif self.cfg.method == "csv":
            df = self._read_effsize_csv()
            df = self._apply_redo(df)
            return self._sample(df)
        else:
            raise CustomizedError("method must be 'gff' or 'csv'")

    # ---------- IO ----------
    def _read_gff_sites(self):
        """
        Returns causal sites provided by the gff.
        """
        cand_causal_sites = []
        gff_path = self.cfg.params.get("gff")
        if not os.path.exists(gff_path):
            raise CustomizedError(f"The provided path to gff file ('{gff_path}') is not a valid file path")

        with open(gff_path, "r") as gff:
            for line in gff:
                # skip the file header
                if line.startswith("#") or line.startswith("\n") :
                    continue        
                else:
                    # parse the line to fill in the dict_causal_genes_dict
                    fields = line.rstrip("\n").split("\t")
                    # info = dict(item.split("=") for item in fields[GFF_INFO].split(";"))
                    cand_causal_sites.extend(list(range(int(fields[GFF_START]), int(fields[GFF_END]) + 1)))
        
        return cand_causal_sites
    
    def _read_effsize_csv(self):
        """
        Read effect sizes from a csv file

        Returns:
            just_read: Dataframe storing transmissibility and drug resistance information.
        """
        csv_path = self.cfg.params.get("csv")
        if not os.path.exists(csv_path):
            raise CustomizedError(f"The provided effect size csv file '{csv_path}'"
                        f" does not exist.")

        just_read = pd.read_csv(csv_path)
        if self.cfg.trait_num["transmissibility"] + self.cfg.trait_num["drug_resistance"] != just_read.shape[1] - 1:
            print(f"WARNING: The number of traits in the provided csv '{csv_path}'"
                        f" does not match the number of traits required '{self.cfg.trait_num}'.")
            just_read = just_read.iloc[:, : self.cfg.trait_num["transmissibility"] + self.cfg.trait_num["drug_resistance"] + 1]
        just_read.columns = sum([["Sites"], 
                         [f"trait_{i}" for i in range(self.cfg.trait_num["transmissibility"] + self.cfg.trait_num["drug_resistance"])]], [])
        return just_read

    def _write_outputs(self, df_eff, seeds):
        """
        Write causal_gene_info.csv and seeds_trait_values.csv into wkdir.
        """
        df_eff.to_csv(os.path.join(self.cfg.wk_dir, "causal_gene_info.csv"), index = False)
        seeds.to_csv(os.path.join(self.cfg.wk_dir, "seeds_trait_values.csv"), index = False)
    
    # ---------- Selection & redo ----------
    def _select_sites(self, candidates, pis):
        """
        Returns causal sites chosen from the regions

        Parameters:
            candidates (list[int]): Candidate causal sites.
            pis (list[float]): A list of pis for Bernoulli trials for each trait.
        """
        sites_num = len(candidates)
        cand_array = np.array(candidates)

        # Sample sites for each trait
        trait_sites = []
        for pi in pis:
            # Bernoulli trials: 1 if selected, 0 if not
            selected = scipy.stats.bernoulli.rvs(pi, size=sites_num).astype(bool)
            trait_sites.append(cand_array[selected])
        
        # Get unique sites across all traits
        all_sites = np.concatenate(trait_sites) if trait_sites else np.array([])
        unique_sites = np.unique(all_sites)
        
        if len(unique_sites) == 0:
            print("WARNING: No causal sites were drawn from the candidate sites. "
                "It's recommended to consider increasing the Bernoulli trial parameter Pis "
                "and rerun the program if this is not a demanded behavior", flush=True)
        
        # Build DataFrame
        df_out = pd.DataFrame({'Sites': unique_sites})
        
        # Mark which traits each site affects
        for i, trait_site_list in enumerate(trait_sites):
            df_out[f'trait_{i}'] = df_out['Sites'].isin(trait_site_list).astype(int)
        
        df_out.sort_values(by='Sites', inplace=True)
        return df_out  

    def _apply_redo(self, df):
        """
        Apply redo policy: none/effsize/sites. 
        Redo handling in run_effsize_generation (csv path).

        Parameters:
            df (pd.DataFrame):
        Returns:
            pd.DataFrame containing new genetic effect sizes.
        """
        redo = self.cfg.params.get("redo")
        if not redo:
            return df
        
        if redo == "effsize":
            for i in range(self.cfg.trait_num["transmissibility"] + self.cfg.trait_num["drug_resistance"]):
                df[f'trait_{i}'] = df[f'trait_{i}'].astype(float)
                df.loc[df[f'trait_{i}'] != 0, f'trait_{i}'] = 1
        elif redo == "sites":
            df = self._select_sites(list(df["Sites"]), self.cfg.pis)

        return df

    # ---------- Sampling (uni or MV) ----------
    def _sample(self, df_id):
        if self.cfg.pleiotropy:
            return self._sample_pleiotropy(df_id)
        return self._sample_univariate(df_id)

    def _sample_univariate(self, df_id: pd.DataFrame) -> pd.DataFrame:
        """
        Univariate sampling per func = 'n'|'l'|'st' using corrected branches.
        Port: draw_eff_size non-pleiotropy paths (pointnormal/laplace/studentst).

        Parameters:
            df_id (pd.DataFrame): The pandas data frame where entries reprenent causal
        Returns:
            Returns drawn effect sizes for all the traits
            should be a pandas data frame where rows are sites
            and columns are traits
        """
        func = self.cfg.func # default is n if nothing is given
        # type the df_id columns
        # df_id[f'trait_{i}'] = df_id[f'trait_{i}'].astype(float)
        if func == "n":
            for i in range(sum(self.cfg.trait_num.values())):
                df_id[f'trait_{i}'] = df_id[f'trait_{i}'].astype(float)
                df_id.loc[df_id[f'trait_{i}'] > 0, f'trait_{i}'] =list(
                    self._pointnormal(n = np.sum(df_id[f'trait_{i}']).astype(int), 
                    tau = self.cfg.params.get("taus")[i]))
        elif func == "l":
            for i in range(sum(self.cfg.trait_num.values())):
                df_id[f'trait_{i}'] = df_id[f'trait_{i}'].astype(float)
                df_id.loc[df_id[f'trait_{i}'] > 0, f'trait_{i}'] =list(
                    self._laplace(n = np.sum(df_id[f'trait_{i}']).astype(int), 
                    b = self.cfg.params.get("bs")[i]))
        elif func == "st":
            for i in range(sum(self.cfg.trait_num.values())):
                df_id[f'trait_{i}'] = df_id[f'trait_{i}'].astype(float)
                df_id.loc[df_id[f'trait_{i}'] > 0, f'trait_{i}'] = list(
                    self._studentst(n = np.sum(df_id[f'trait_{i}']).astype(int), 
                    scale = self.cfg.params.get("s")[i], 
                    nv = self.cfg.params.get("nv")))
        return df_id


    def _sample_pleiotropy(self, df_id: pd.DataFrame) -> pd.DataFrame:
        """
        Row-wise MVN/MVL/MVT sampling with stabilized conditioning and squeezes.
        Port: multivar_normal/multivar_laplace/multivar_st and pleiotropy branch.

        Parameters:
            df_id (pd.DataFrame): The pandas data frame where entries reprenent causal
        Returns:
            Returns drawn effect sizes for all the traits
            should be a pandas data frame where rows are sites
            and columns are traits
        """
        func = self.cfg.func # default is n if nothing is given
        cov = self.cfg.params.get("cov")
        if cov is None:
            raise CustomizedError("Missssing required argument: -cov in the pleitropy mode")
        if cov.shape != (2, 2):   # Might not be nesessary after the main function is completed
                raise CustomizedError(f"The given dimensions of the covariance matrix (-cov) {cov.shape}"
                        f" do not match the number of traits to be drawn {(self.cfg.trait_num, self.cfg.trait_num)} ")

        trait_cols = df_id.columns[1:1+sum(self.cfg.trait_num.values())]
        df_id[trait_cols] = df_id[trait_cols].astype(float)

        row_ids = df_id.loc[df_id.iloc[:, 1:].sum(axis=1) > 1]
        if len(row_ids) == 0:
            print("WARNING: There is no pleiotropy site", flush = True)
        
        if func == "n":
            for j in df_id.index:
                df_id.loc[j, df_id.columns[1:]] = self._multivar_normal(df_id.loc[j, df_id.columns[1:]], cov)
        elif func == "l":
            for j in df_id.index:
                df_id.loc[j, df_id.columns[1:]] = self._multivar_laplace(df_id.loc[j, df_id.columns[1:]], cov)
        elif func == "st":
            for j in df_id.index:
                df_id.loc[j, df_id.columns[1:]] = self._multivar_st(df_id.loc[j, df_id.columns[1:]], 
                    scale = self.cfg.params.get("s_cov"), nv = self.cfg.params("nv"))
        return df_id



    def _pointnormal(self, n, tau):
        """
        Spike and slab to draw effect size. Return effect sizes for the selected
        sites following a point normal distribution for ONE trait.

        Parameters:
            n (int): Number of effect sizes to draw using this hyperparameter tau
            tau (float): Standard deviation of the effect sizes for each trait ~ N(0, tau^2)
        """
        return np.random.normal(0, tau, size=n)

    def _laplace(self, n, b):
        """
        Draw n effect sizes for one trait from a Laplace distribution.

        Parameters:
            n (int): Number of effect sizes to draw using this hyperparameter tau
            b (float): Scale of the Laplace distribution
        """
        return np.random.laplace(0, b, size=n)

    def _studentst(self, n, scale=1, nv=3):
        """
        Draw n effect sizes for one trait from a Student's t distribution.

        Parameters:
            n (int): Number of effect sizes to draw using this hyperparameter tau
            scale (float): Scale of the student's t's distribution
            nv (float): Degrees of freedom of the Student's t distribution
        """
        return scale * np.random.standard_t(nv, size=int(n))

    def _multivar_normal(self, indicator, cov):
        """
        Sample active traits conditional on multivariante normal distribution.

        Parameters:
            indicator (pd.Series or np.array): 0/1 vector of active traits
            cov (2D array): Full covariance matrix (d x d)

        Returns:
            np.array: Length-d effect size vector
        """
        indicator = np.array(indicator).astype(int)
        d = len(indicator)

        active_idx = np.where(indicator == 1)[0]
        # inactive_idx = np.where(indicator == 0)[0]

        if len(active_idx) == 0:
            return np.zeros(d)

        cov_sub = cov[np.ix_(active_idx, active_idx)]

        # stablizating for numerical computation
        cov_sub = (cov_sub + cov_sub.T) / 2.0
        min_eig = np.min(np.linalg.eigvalsh(cov_sub))
        if min_eig < 1e-12: 
            cov_sub += (1e-12 - min_eig) * np.eye(cov_sub.shape[0])
        sample = np.random.multivariate_normal(np.zeros(len(active_idx)), cov_sub)
        # Place into full-length vector
        effect = np.zeros(d)
        effect[active_idx] = sample.squeeze()
        return effect

    def _multivar_laplace(self, indicator, cov):
        """
        Sample active traits under the multivariate laplace.

        Parameters:
            indicator (pd.Series or np.array): 0/1 vector of active traits
            cov (2D array): Full covariance matrix (d x d)

        Returns:
            np.array: Length-d effect size vector
        """
        indicator = np.array(indicator).astype(int)
        d = len(indicator)

        active_idx = np.where(indicator == 1)[0]
        # inactive_idx = np.where(indicator == 0)[0]

        if len(active_idx) == 0:
            return np.zeros(d)

        new_cov = cov[np.ix_(active_idx, active_idx)]

        # stablizing for numerical computation
        new_cov = (new_cov + new_cov.T) / 2.0
        min_eig = np.min(np.linalg.eigvalsh(new_cov))
        if min_eig < 1e-12:
            new_cov = new_cov + (1e-12 - min_eig) * np.eye(new_cov.shape[0])

        sampler = MvLaplaceSampler(np.zeros(len(active_idx)), new_cov)
        sample = sampler.sample(1).squeeze() # to prevent shape mismatching, getting rid of extra 1 dim
        effect = np.zeros(d)
        effect[active_idx] = sample

        return effect

    def _multivar_st(self, indicator, scale, nv=3):
        """
        Sample active traits under the multivariate student's t distribution.

        Parameters:
            indicator (pd.Series or np.array): 0/1 vector of active traits
            scale (2D array): Scale matrix (d x d)
            nv: Degree of freedom

        Returns:
            np.array: Length-d effect size vector
        """
        indicator = np.array(indicator).astype(int)
        cov = np.array(scale)
        d = len(indicator)

        active_idx = np.where(indicator == 1)[0]
        # inactive_idx = np.where(indicator == 0)[0]

        if len(active_idx) == 0:
            return np.zeros(d)

        # stablizing for numerical computation
        new_cov = cov[np.ix_(active_idx, active_idx)]
        new_cov = (new_cov + new_cov.T) / 2.0
        min_eig = np.min(np.linalg.eigvalsh(new_cov))
        if min_eig < 1e-12:
            new_cov = new_cov + (1e-12 - min_eig) * np.eye(new_cov.shape[0])
        
        mvt_dist = multivariate_t(loc=np.zeros(len(active_idx)), shape=new_cov, df=nv)
        sample = mvt_dist.sample(1).squeeze()
        effect = np.zeros(d)
        effect[active_idx] = sample

        return effect

        # ---------- Seeds traits & calibration ----------
    def _compute_seed_traits(self, df_eff: pd.DataFrame) -> pd.DataFrame:
        """
        Sum effects per seed using VCF files in wkdir/originalvcfs.
        Port: seeds_trait_calc (with warnings, empty fallback when no VCFs).
        
        Parameters:
            df_eff (pd.DataFrame): Effect size dataframe
        """
        seeds_vcf_dir = os.path.join(self.cfg.wk_dir, "originalvcfs/")
        trait_cols = list(df_eff.columns)[1:]

        seed_vals = []

        # raise exception if we do not have access to VCF of individual seeds
        if not os.path.exists(seeds_vcf_dir):
            print("WARNING: seed_generator.py hasn't been run. "
                    "If you want to use seed sequence different than reference genome, "
                    "you must run seed_generator first", flush = True)
            if self.cfg.n_seed == 0:
                raise CustomizedError("The number of seeds cannot be 0 when seeding with one reference genome.")
            empty_data = {"Seed_ID": [f"seed_{i}" for i in range(self.cfg.n_seed)],
            **{trait: [0] * self.cfg.n_seed for trait in trait_cols}}

            return pd.DataFrame(empty_data)

        else:
            seeds = sorted([f for f in os.listdir(seeds_vcf_dir) if f.endswith(".vcf")])
            if len(seeds) > self.cfg.n_seed:
                print(f"WARNING: More seeding sequences ({len(seeds)}) than the specified number ({self.cfg.n_seed}) "
                    f"are detected. Only the first {self.cfg.n_seed} files will be used", flush = True)
            all_effpos = df_eff["Sites"].tolist()

            for _ , seed_file in enumerate(seeds[:self.cfg.n_seed]):
                sum_trait = np.zeros(len(trait_cols))

                with open(os.path.join(seeds_vcf_dir, seed_file), "r") as seed_vcf:
                    for line in seed_vcf:
                        if not line.startswith("#"):
                            fields = line.rstrip("\n").split("\t")
                            mut_pos = int(fields[1])
                            if mut_pos in all_effpos:
                                effect_row = df_eff.loc[df_eff["Sites"] == mut_pos, trait_cols].values.squeeze()
                                sum_trait += effect_row

                seed_vals.append(sum_trait)

            # Convert list of arrays to DataFrame
            df_out = pd.DataFrame(seed_vals, columns=trait_cols)
            df_out["Seed_ID"] = list(range(self.cfg.n_seed))
            df_out = df_out[["Seed_ID"] + trait_cols]
            return df_out

    def _calibrate(self, df_eff: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
        """
        Optional calibration placeholder.
        Port: calibration (currently no-op).

        Parameters:
            df_eff (dataframe): Uncalibrated effect size data frame
            seeds (dataframe): Trait values of the seeds
        """
        return df_eff
        

    # ---------- Rename helpers ----------
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to transmissibility_i and drug_resistance_i based on trait_n.
        Port: rename_df.

        Parameters:
        df: Pandas dataframe, first column can be seed_ID or sites, other columns are traits.
        """
        df.columns = sum([[df.columns[0]], 
                         [f"transmissibility_{i}" for i in range(self.cfg.trait_num["transmissibility"])],
                         [f"drug_resistance_{i}" for i in range(self.cfg.trait_num["drug_resistance"])]], [])
        return df

def effsize_generation_byconfig(all_config):
    # """
    # Generates effect size file and compute seeds' trait values based on a provided config file.

    # Parameters:
    #     all_config (dict): A dictionary of the configuration (read with read_params()).
    # """

    # genetic_config = all_config["GenomeElement"]
    # wk_dir = all_config["BasicRunConfiguration"]["cwdir"]
    # effsize_method = genetic_config["effect_size"]["method"]
    # random_seed = all_config["BasicRunConfiguration"].get("random_number_seed", None)
    # num_seed = all_config["SeedsConfiguration"]["seed_size"]
    # subst_model_param = all_config["EvolutionModel"]["subst_model_parameterization"]
    # if subst_model_param=="mut_rate":
    #     use_subst_matrix=False
    # elif subst_model_param=="mut_rate_matrix":
    #     use_subst_matrix=True
    # else:
    #     raise CustomizedError(f"The given subst_model_parameterization is NOT valid -- please input 'mut_rate' or 'mut_rate_matrix'.")
    # mu_matrix_ori = all_config["EvolutionModel"]["burn_in_mutrate_matrix"]
    # mu_matrix = {"A": mu_matrix_ori[0], "C": mu_matrix_ori[1], "G": mu_matrix_ori[2], "T": mu_matrix_ori[3]}

    # eff_params_config = genetic_config["effect_size"]["randomly_generate"]
    # error = run_effsize_generation(method=effsize_method, wk_dir=wk_dir, trait_n=genetic_config["traits_num"], 
    #                      effsize_path=genetic_config["effect_size"]["user_input"]["path_effsize_table"],
    #                      causal_sizes=eff_params_config["genes_num"], es_lows=eff_params_config["effsize_min"], 
    #                      es_highs=eff_params_config["effsize_max"], gff_in=eff_params_config["gff"], 
    #                      n_gen=all_config["EvolutionModel"]["n_generation"], mut_rate=all_config["EvolutionModel"]["mut_rate"], 
    #                      norm_or_not=eff_params_config["normalize"], rand_seed = random_seed, num_seed=num_seed,
    #                      use_subst_matrix=use_subst_matrix, mu_matrix=mu_matrix, ref=all_config["GenomeElement"]["ref_path"], 
    #                      final_T = eff_params_config["final_trait"])
    # return error
    raise NotImplementedError

def main():
    parser = argparse.ArgumentParser(description='Generate or modify seeds.')
    parser.add_argument('-method', action='store',dest='method', type=str, required=True, help="Method of the genetic element file generation, using csv or gff")
    parser.add_argument('-wkdir', action='store',dest='wkdir', type=str, required=True, help="Working directory")
    parser.add_argument('-csv', action='store',dest='csv', type=str, required=False, help="Path to the user-provided effect size genetic element csv file", default="")
    parser.add_argument('-gff', action='store',dest='gff', type=str, required=False, help='Path to the gff file', default="")
    parser.add_argument('-trait_n', action='store', dest='trait_n', type=ast.literal_eval, required=True, 
        help="Number of traits that user want to generate a genetic architecture for transmissibility and drug resistance, format: '{\"transmissibility\": x, \"drug-resistance\": y}'", default="")
    parser.add_argument('-redo', action='store',dest='redo', type=str, required=False, help="Which steps to redo in the effect size generating process (sites/effsize/none)")
    parser.add_argument('-func', action='store',dest='func', type=str, required=True, help="Function to generate the effect sizes given causal sites. (n/l/st)")
    parser.add_argument('-pleiotropy', action='store',dest='pleiotropy', type=str2bool, required=False, help="Whether to do pleiotropy", default=False)
    parser.add_argument('-pis','--pis', nargs='+', help='The probability of the Bernoulli trials for each candidate sites for each trait. Should be a float list with the same length of the number of traits in total', required=True, type=float, default=[])
    parser.add_argument('-taus','--taus', nargs='+', help='Standard deviation of the effect sizes for each trait under the point normal model. Required when func=n', required=False, type=float, default=[])
    parser.add_argument('-bs','--bs', nargs='+', help='Scales of the laplace distribution of the effect sizes for each trait under the Laplace model. Required when func=l', required=False, type=float, default=[])
    parser.add_argument('-nv','--nv', action='store', help='Degree of freedom of the Student\'s t\'s distribution of the effect sizes for each trait under the student\'s t model. Optional when func=st', required=False, type=float, default=3)
    parser.add_argument('-cov','--cov', help='Covariance matrix in the format of nested list, e.g., [[1,2],[3,4]]. Required when pleiotropy=True and func=n/l. For pleiotropy with func=st, use -s_cov.', required=False, type=ast.literal_eval, default="[]")
    parser.add_argument('-s','--s', nargs='+', help='Scales of the Student\'s t distribution of the effect sizes for each trait under the Student\'s t model. Required when func=st', required=False, type=float, default=[])
    parser.add_argument('-s_cov','--s_cov', help='Scale matrix in the format of nested list, e.g., [[1,2],[3,4]]. Required when pleiotropy=True and func=st', required=False, type=ast.literal_eval, default="[]")
    parser.add_argument('-random_seed', action = 'store', dest = 'random_seed', required = False, type = int, default = None)
    parser.add_argument('-n_seed', action='store', dest = 'n_seed', required = True, type = int, default = 1)
    parser.add_argument('-calibration', action='store',dest='calibration', type=str2bool, required=False, help="Whether to do pleiotropy", default=False)

    args = parser.parse_args()
    config = GeneticEffectConfig(
        method = args.method,
        wk_dir = args.wkdir,
        n_seed = args.n_seed,
        func = args.func,
        calibration = args.calibration,
        random_seed = args.random_seed,
        csv = args.csv,
        gff = args.gff,
        trait_num = args.trait_n,
        pleiotropy = args.pleiotropy,
        pis = args.pis,
        taus = args.taus,
        bs = args.bs,
        nv = args.nv,
        s = args.s,
        s_cov = np.array(args.s_cov),
        cov = np.array(args.cov),
        redo = args.redo,
    )

    generator = EffectGenerator(config)
    generator.run()

if __name__ == "__main__":
    main()

    

        
    

        






