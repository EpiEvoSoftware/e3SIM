from e3SIM_codes.base_func import *
from e3SIM_codes.error_handling import CustomizedError
import numpy as np
import ast
import pandas as pd
import scipy
import argparse, statistics, os, json
from Bio import SeqIO

START_IDX = 0
END_IDX = 1
E_SIZE = 2

GFF_START = 3
GFF_END = 4
GFF_INFO = 8

DEFAULT_R = 1.0  # Default odds ratio / hazard ratio per SD
DEFAULT_VTGT = 1.0 # Default target variance for calibrating effect sizes


# ------------- Genetic Effect Configuration ----------------------- #
class GeneticEffectConfig:
    def __init__(self, method, wk_dir, n_seed, func, calibration, trait_num, random_seed, pis, **kwargs):
        self.method = method # 'gff' | 'csv'
        self.wk_dir = wk_dir 
        self.n_seed = n_seed # number of seeds to generate
        self.calibration = calibration # whether to do calibration
        self.trait_num = trait_num # number of traits
        self.func = func # n | st | l
        self.random_seed = random_seed
        self.pis = pis
        self.params = kwargs # gff, csv, nv, bs, taus, s, site_method, Rs, var_target, caliberation link

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
            raise CustomizedError(f"{self.func} isn't a valid method for sampling effect sizes. Please choose a permitted method."
                                "(n/l/st)")
        if self.method=="gff":
            if self.params.get("site_method") not in ("p", "n"):
                raise CustomizedError(f"{self.params.get("site_method")} isn't a valid method for resampling causal sites. Please choose a permitted method."
                                "(n/p)")
            if self.params.get("site_method") == "p":
                if len(self.pis) != sum(self.trait_num.values()):
                    raise CustomizedError("If you wish to sample causal sites from the candidate region using Bernoulli trials, "
                        f"Please provide the success probability list (-pis) with the same length as your trait quantities({sum(self.trait_num.values())})")
                if any(x < 0 or x > 1 for x in self.pis):
                    raise CustomizedError("The success probability for causal site sampling has to be within [0, 1].")
            elif self.params.get("site_method")=="n":
                if len(self.params.get("Ks")) != sum(self.trait_num.values()):
                    raise CustomizedError("If you wish to sample causal sites from the candidate region using specified numbers by uniform sampling, "
                        f"Please provide the causal site number list (-Ks) with the same length as your trait quantities({sum(self.trait_num.values())})")
                if any(type(x)!=int for x in self.params["Ks"]):
                    raise CustomizedError("The success probability for causal site sampling has to be an integer.")

        
            if self.func == "n" and len(self.params.get("taus")) != sum(self.trait_num.values()):
                raise CustomizedError(f"The given length of the variance (-taus) {self.params.get("taus")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the point normal mode.")
            if self.func == "l" and len(self.params.get("bs")) != sum(self.trait_num.values()):
                raise CustomizedError(f"The given length of the scales (-bs) {self.params.get("bs")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the laplace mode.")
            if self.func == "st" and len(self.params.get("s")) != sum(self.trait_num.values()):
                raise CustomizedError(f"The given length of the scales (-s) {self.params.get("s")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the laplace mode.")

        if self.params.get("calibration_link"):
            if self.params.get("link") not in ["logit", "cloglog"]:
                raise CustomizedError(f"If you would like to calibrate the link-scale slope,"
                            f" -link value needs to be either 'logit' or 'cloglog'.")
            if self.params.get("Rs") == []:
                self.params["Rs"] = np.full(sum(self.trait_num.values()), DEFAULT_R)
            elif len(self.params.get("Rs")) != sum(self.trait_num.values()):
                raise CustomizedError("If you wish to provide a odds ratio / hazard ratio per SD for calibration"
                    f"Please provide a list with the same length as your trait quantities({sum(self.trait_num.values())})")
            elif any(x <= 0 for x in self.params["Rs"]):
                raise CustomizedError("If you wish to provide a odds ratio / hazard ratio per SD for calibration"
                    f"The odds ratio or the hazard ratio (-Rs) per SD has to be a list of possible numbers.")


# ------------- Genetic Effect Generation ----------------------- #
class EffectGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        np.random.seed(cfg.random_seed)

    def run(self):
        try:
            self.cfg.validate()
            df_eff = self._build_effect_df()
            seeds, seeds_state = self._compute_seed_traits(df_eff)
            df_eff = self._rename_columns(df_eff)
            if self.cfg.calibration:
                if self.cfg.params.get("var_target") == []:
                    self.cfg.params["var_target"] = np.full(sum(self.cfg.trait_num.values()), DEFAULT_VTGT)
                elif len(self.cfg.params.get("var_target")) != sum(self.cfg.trait_num.values()):
                    raise CustomizedError("If you wish to provide a target variance for calibration"
                            f"Please provide a list with the same length as your trait quantity {sum(self.cfg.trait_num.values())}")
                df_eff, em_var = self._calibrate(df_eff, seeds_state)
                if self.cfg.params.get("calibration_link"):
                    calibrated_alphas = self._calibrate_linkslope(self.cfg.params.get("Rs"), self.cfg.params.get("link"), em_var, self.cfg.trait_num)

            seeds_named = self._rename_columns(seeds)
            self._write_outputs(df_eff, seeds_named)
            print("********************************************************************\n"
                "                  GENETIC ARCHITECTURES GENERATED\n"
                "********************************************************************",
                flush=True)
            return None
        except Exception as e:
            print(f"Genetic effects generation - An error occured: {e}.", flush = True)
            error_message = e
            return error_message

        
    # ---------- Build effect df ----------
    def _build_effect_df(self) -> pd.DataFrame:
        if self.cfg.method == "gff":
            candidates = self._read_gff_sites()
            df_sites = self._select_sites(candidates, self.cfg.params["site_method"], self.cfg.pis, self.cfg.params["Ks"])
            return self._sample(df_sites)
        elif self.cfg.method == "csv":
            df = self._read_effsize_csv()
            return(df)
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
        if self.cfg.trait_num["transmissibility"] + self.cfg.trait_num["drug_resistance"] > just_read.shape[1] - 1:
            raise CustomizedError(f"The number of traits in the provided csv '{csv_path}'"
                        f" does not match the number of traits required '{self.cfg.trait_num}'.")
        else:
            just_read = just_read.iloc[:, :self.cfg.trait_num["transmissibility"] + self.cfg.trait_num["drug_resistance"] + 1]
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
    def _select_sites(self, candidates, method, pis, Ks):
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
        if method=="p":
            for pi in pis:
                # Bernoulli trials: 1 if selected, 0 if not
                selected = scipy.stats.bernoulli.rvs(pi, size=sites_num).astype(bool)
                trait_sites.append(cand_array[selected])
        elif method=="n":
            for k in Ks:
                if k > sites_num:
                    raise CustomizedError(f"You required a causal site quantity {k} that is larger than the candidate site list {sites_num}. Please consider using a smaller k.")
                # Sample Ki sites for each trait i
                selected = np.random.choice(sites_num, k, replace=False)
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

    # ---------- Sampling (uni or MV) ----------
    def _sample(self, df_id):
        return self._sample_univariate(df_id)

    def _sample_univariate(self, df_id: pd.DataFrame) -> pd.DataFrame:
        """
        Univariate sampling per func = 'n'|'l'|'st' using corrected branches.
        Port: draw_eff_size non-pleiotropy paths (pointnormal/laplace/studentst).

        Parameters:
            df_id (pd.DataFrame): The pandas data frame where entries reprenent causal (0 == no causal; 1 == causal)
        Returns:
            Returns drawn effect sizes for all the traits
            should be a pandas data frame where rows are sites
            and columns are traits
        """
        func = self.cfg.func # default is n if nothing is given

        if func == "n":
            for i in range(sum(self.cfg.trait_num.values())):
                col = f"trait_{i}"
                mask = df_id[col] > 0
                df_id[col] = df_id[col].astype(float)
                df_id.loc[mask, col] =list(
                    self._pointnormal(n = np.sum(df_id[col]).astype(int), 
                    tau = self.cfg.params.get("taus")[i]))
        elif func == "l":
            for i in range(sum(self.cfg.trait_num.values())):
                col = f"trait_{i}"
                mask = df_id[col] > 0
                df_id[col] = df_id[col].astype(float)
                df_id.loc[mask, col] =list(
                    self._laplace(n = np.sum(df_id[col]).astype(int), 
                    b = self.cfg.params.get("bs")[i]))
        elif func == "st":
            for i in range(sum(self.cfg.trait_num.values())):
                col = f"trait_{i}"
                mask = df_id[col] > 0
                df_id[col] = df_id[col].astype(float)
                df_id.loc[mask, col] = list(
                    self._studentst(n = np.sum(df_id[col]).astype(int), 
                    scale = self.cfg.params.get("s")[i], 
                    nv = self.cfg.params.get("nv")))
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
                    "If you want to use seed sequence different from the reference genome, "
                    "you must run seed_generator first", flush = True)
            empty_data = {"Seed_ID": [f"seed_{i}" for i in range(self.cfg.n_seed)],
            **{trait: [0] * self.cfg.n_seed for trait in trait_cols}}

            return pd.DataFrame(empty_data)

        else:
            seeds = sorted([f for f in os.listdir(seeds_vcf_dir) if f.endswith(".vcf")])
            df_AF = df_eff.iloc[:, 0].to_frame(name="Sites")
            for i in range(self.cfg.n_seed):
                df_AF[f"seed_{i}"] = 0
            if len(seeds) > self.cfg.n_seed:
                print(f"WARNING: More seeding sequences ({len(seeds)}) than the specified number ({self.cfg.n_seed}) "
                    f"are detected. Only the first {self.cfg.n_seed} files will be used", flush = True)
            all_effpos = df_eff["Sites"].tolist()

            for seed_idx , seed_file in enumerate(seeds[:self.cfg.n_seed]):
                with open(os.path.join(seeds_vcf_dir, seed_file), "r") as seed_vcf:
                    sum_trait = np.zeros(len(trait_cols)) # number of traits
                    for line in seed_vcf:
                        if not line.startswith("#"):
                            fields = line.rstrip("\n").split("\t")
                            mut_pos = int(fields[1])
                            if mut_pos in all_effpos:
                                effect_row = df_eff.loc[df_eff["Sites"] == mut_pos, trait_cols].values.squeeze()
                                sum_trait += effect_row
                                df_AF.loc[df_AF["Sites"] == mut_pos, f"seed_{seed_idx}"] += 1

                seed_vals.append(sum_trait)

            # Convert list of arrays to DataFrame
            df_out = pd.DataFrame(seed_vals, columns=trait_cols)
            df_out["Seed_ID"] = list(range(self.cfg.n_seed))
            df_out = df_out[["Seed_ID"] + trait_cols]
            return df_out, df_AF


    def _calibrate(self, df_eff: pd.DataFrame, seeds_state: pd.DataFrame) -> pd.DataFrame:
        """
        Optional calibration placeholder.

        Parameters:
            df_eff (dataframe): Uncalibrated effect size data frame
            seeds_state (dataframe): df_AF, mutation state of the seeds

        Returns
            df_eff_calibrated : pd.DataFrame
            Calibrated effect sizes with the same structure as df_eff.
            var_empirical : pd.Series
            Empirical variance of each trait before calibration.
        """
        # Extract numeric parts (exclude site identifiers)
        geno = seeds_state.iloc[:, 1:].to_numpy(dtype=float)  # (n_sites, n_seeds)
        eff = df_eff.iloc[:, 1:].to_numpy(dtype=float)        # (n_sites, n_traits)

        # Compute allele frequency and center genotype
        AF_all = geno.mean(axis=1, keepdims=True)             # (n_sites, 1)
        center_geno = geno - AF_all                           # (n_sites, n_seeds)

        # Compute centered trait matrix: (n_seeds, n_traits)
        center_trait = center_geno.T @ eff                    # (n_seeds, n_traits)

        # Empirical variance of each trait
        var_empirical = (center_trait ** 2).sum(axis=0) / (geno.shape[1] - 2)
        var_empirical = pd.Series(var_empirical, index=df_eff.columns[1:])

        # Target variances
        var_target = np.array(self.cfg.params["var_target"], dtype=float)

        # Compute scaling coefficients safely
        with np.errstate(divide='ignore', invalid='ignore'):
            c_i = np.sqrt(var_target / var_empirical.replace(0, np.nan))
        c_i = c_i.fillna(1.0).to_numpy()

        # Print warnings only where needed
        zero_var_traits = var_empirical.index[var_empirical == 0]
        for name in zero_var_traits:
            print(f"WARNING: No variance in trait {name}. Calibration not applicable. Original effect sizes preserved.")

        # Apply calibration
        eff_calibrated = eff * c_i   # broadcast across sites -> ci = [1, n_traits]

        # Construct calibrated DataFrame
        df_eff_calibrated = pd.concat(
            [df_eff.iloc[:, [0]].reset_index(drop=True), 
            pd.DataFrame(eff_calibrated, columns=df_eff.columns[1:])],
            axis=1
        )

        return df_eff_calibrated, var_empirical
        
        

    # ---------- Rename helpers ----------
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename columns to transmissibility_i and drug_resistance_i based on trait_n.
        Port: rename_df.

        Parameters:
        df: Pandas dataframe, first column can be seed_ID or sites, other columns are traits.
        """
        df.columns = sum([[df.columns[0]], 
                         [f"transmissibility_{i+1}" for i in range(self.cfg.trait_num["transmissibility"])],
                         [f"drug_resistance_{i+1}" for i in range(self.cfg.trait_num["drug_resistance"])]], [])
        return df

    def _calibrate_linkslope(self, Rs: np.array, link_type: str, var_em: np.array, trait_num: dict) -> np.array:
        """
        Calibrate the link-scale slope by specifying the effect per SD of the trait values.

        Parameters:
        Rs: Numpy array, the odds ratio or hazard ratio per SD, defauly are 1s
        link_type: string, cloglog or logit
        var_em: empirical variance for each trait in the seed population calculated from _calibrate()
        trait_num: Dictionary of the number of traits

        """
        SD_em = np.sqrt(var_em)
        SD_safe = np.where(SD_em > 0, SD_em, np.nan)

        trans_num = trait_num["transmissibility"]
        drug_num = trait_num["drug_resistance"]

            # Base slope computation
        if link_type == "logit":
            alphas = np.log(Rs) / SD_safe
        elif link_type == "cloglog":
            alphas = np.concatenate([
                np.log(Rs[:trans_num]) / SD_safe[:trans_num] if trans_num else np.array([]),
                -np.log(Rs[trans_num:]) / SD_safe[trans_num:] if drug_num else np.array([])
            ])
        else:
            raise ValueError(f"Unknown link_type: {link_type}")

        # Split by trait type
        alpha_trans = alphas[:trans_num] if trans_num else np.array([])
        alpha_drug  = alphas[trans_num:] if drug_num else np.array([])

         # ---- Printing section ----
        print(f"The calibrated link-scale slopes under the {link_type} link are as follows.")

        for i, val in enumerate(alpha_trans):
            label = f"transmissibility_{i+1}"
            print(f"  {label}: {'NA' if np.isnan(val) else f'{val:.4f}'}")

        for i, val in enumerate(alpha_drug):
            label = f"drug_resistance_{i+1}"
            print(f"  {label}: {'NA' if np.isnan(val) else f'{val:.4f}'}")

        
        if np.all(np.isfinite(alpha_trans)) and np.all(np.isfinite(alpha_drug)):
            print('Please write the following part to your configuration file under the "trait_prob_link" key:')   
            alpha_trans_list = np.round(alpha_trans, 4).tolist()
            alpha_drug_list  = np.round(alpha_drug, 4).tolist()

            config = {
                "link": link_type,
                link_type: {
                    "alpha_trans": alpha_trans_list,
                    "alpha_drug": alpha_drug_list,
                },
            }

            config_json = json.dumps(config, indent=2)
            print(config_json)
            return config_json

        else:
            print('TIPS: slopes shown as "NA" will have to be specified manually since slope calibration failed for those traits.\
                Please write non-NA values into your configuration file under the "trait_prob_link" key')
            return({})




def effsize_generation_byconfig(all_config):
    """
    Generates effect size file and compute seeds' trait values based on a provided config file.

    Parameters:
        all_config (dict): A dictionary of the configuration (read with read_params()).
    """

    genetic_config = all_config["GenomeElement"]
    wk_dir = all_config["BasicRunConfiguration"]["cwdir"]
    random_seed = all_config["BasicRunConfiguration"].get("random_number_seed", None)
    num_seed = all_config["SeedsConfiguration"]["seed_size"]
    
    try:
        config = GeneticEffectConfig(
            method = genetic_config["effect_size"]["method"],
            wk_dir = wk_dir,
            n_seed = num_seed,
            func = genetic_config["effect_size"]["effsize_params"]["effsize_function"],
            calibration = genetic_config["effect_size"]["calibration"]["do_calibration"],
            random_seed = random_seed,
            csv = genetic_config["effect_size"]["filepath"]["csv_path"],
            gff = genetic_config["effect_size"]["filepath"]["gff_path"],
            trait_num = genetic_config["traits_num"],
            pis = genetic_config["effect_size"]["causalsites_params"]["pis"],
            Ks = genetic_config["effect_size"]["causalsites_params"]["Ks"],
            taus = genetic_config["effect_size"]["effsize_params"]["normal"]["taus"],
            bs = genetic_config["effect_size"]["effsize_params"]["laplace"]["bs"],
            nv = genetic_config["effect_size"]["effsize_params"]["studentst"]["nv"],
            s = genetic_config["effect_size"]["effsize_params"]["studentst"]["s"],
            var_target = genetic_config["effect_size"]["calibration"]["V_target"],
            calibration_link = genetic_config["trait_prob_link"]["calibration"],
            Rs = genetic_config["trait_prob_link"]["Rs"],
            link = genetic_config["trait_prob_link"]["link"],
            site_method = genetic_config["effect_size"]["causalsites_params"]["method"]
        )
    except Exception as e:
        return e

    generator = EffectGenerator(config) # no validation going on so leave it out of the try catch clause
    error = generator.run()
    

    return error


def main():
    parser = argparse.ArgumentParser(description='Generate or modify seeds.')
    parser.add_argument('-method', action='store',dest='method', type=str, required=True, help="Method of the genetic element file generation, using csv or gff")
    parser.add_argument('-wkdir', action='store',dest='wkdir', type=str, required=True, help="Working directory")
    parser.add_argument('-csv', action='store',dest='csv', type=str, required=False, help="Path to the user-provided effect size genetic element csv file", default="")
    parser.add_argument('-gff', action='store',dest='gff', type=str, required=False, help='Path to the gff file', default="")
    parser.add_argument('-trait_n', action='store', dest='trait_n', type=ast.literal_eval, required=True, 
        help="Number of traits that user want to generate a genetic architecture for transmissibility and drug resistance, format: '{\"transmissibility\": x, \"drug-resistance\": y}'", default="")
    # parser.add_argument('-redo', action='store',dest='redo', type=str, required=False, default="sites", help="Which steps to redo in the effect size generating process (sites/effsize/none)")
    parser.add_argument('-func', action='store',dest='func', type=str, required=True, help="Function to generate the effect sizes given causal sites. (n/l/st)")
    # parser.add_argument('-pleiotropy', action='store',dest='pleiotropy', type=str2bool, required=False, help="Whether to do pleiotropy", default=False)
    parser.add_argument('-site_method', action='store',dest='site_method', type=str, required=False, help="Method to sample causal site, by probability (p) or by number of sites (n)", default="p")
    parser.add_argument('-pis','--pis', nargs='+', help='The probability of the Bernoulli trials for each candidate sites for each trait. Should be a float list with the same length of the number of traits in total. Required when site_method=p.', required=False, type=float, default=[])
    parser.add_argument('-Ks','--Ks', nargs='+', help='The number of causal sites for each trait. Should be a integer list with the same length of the number of traits in total. Required when site_method=n.', required=False, type=int, default=[])
    parser.add_argument('-taus','--taus', nargs='+', help='Standard deviation of the effect sizes for each trait under the point normal model. Required when func=n', required=False, type=float, default=[])
    parser.add_argument('-bs','--bs', nargs='+', help='Scales of the laplace distribution of the effect sizes for each trait under the Laplace model. Required when func=l', required=False, type=float, default=[])
    parser.add_argument('-nv','--nv', action='store', help='Degree of freedom of the Student\'s t\'s distribution of the effect sizes for each trait under the student\'s t model. Optional when func=st', required=False, type=float, default=3)
    parser.add_argument('-s','--s', nargs='+', help='Scales of the Student\'s t distribution of the effect sizes for each trait under the Student\'s t model. Required when func=st', required=False, type=float, default=[])
    parser.add_argument('-random_seed', action = 'store', dest = 'random_seed', required = False, type = int, default = None)
    parser.add_argument('-n_seed', action='store', dest = 'n_seed', required = True, type = int, default = 1)
    parser.add_argument('-calibration', action='store',dest='calibration', type=str2bool, required=False, help="Whether to calibrate the effect size values", default=False)
    parser.add_argument('-var_target', '--var_target', nargs='+', help='The target variance of the seeds\' genetic values', required=False, type=float, default=[])
    parser.add_argument('-calibration_link', action='store',dest='calibration_link', type=str2bool, required=False, help="Whether to calibrate the link-scale slope", default=False)
    parser.add_argument('-Rs', '--Rs', nargs='+', help='The odds ratio for the transmission/survival per SD of trait values under logit, or the hazard ratio per SD under cloglog', required=False, type=float, default=[])
    parser.add_argument('-link', action='store',dest='link', type=str, required=False, help="Link type: logit or cloglog", default="logit")

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
        site_method = args.site_method,
        pis = args.pis,
        taus = args.taus,
        Ks = args.Ks,
        bs = args.bs,
        nv = args.nv,
        s = args.s,
        var_target = args.var_target,
        calibration_link = args.calibration_link,
        Rs = args.Rs,
        link = args.link
    )

    generator = EffectGenerator(config)
    generator.run()

if __name__ == "__main__":
    main()

    

        
    

        






