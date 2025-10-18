from numpy.random.mtrand import randint
from base_func import *
from error_handling import CustomizedError
import numpy as np
import ast
import pandas as pd
import scipy
import argparse, statistics, os, json
from Bio import SeqIO

START_IDX = 0
END_IDX = 1
E_SIZE = 2

DEFAULT_R_OHR = 1.5  # Default odds ratio / transmisison hazard ratio per SD
DEFAULT_R_CLR = 0.667  # Default clearance ratio per SD
DEFAULT_VTGT = 1.0 # Default target variance for calibrating effect sizes
DEFAULT_MU = 0.003
DEFAULT_TAUS = 1
DEFAULT_BS = 1
DEFAULT_NV = 3
DEFAULT_S = 1
EXP_BETAPRIOR = 1.0 / 6.0 # Expected allele frequency


# ------------- Genetic Effect Configuration ----------------------- #
class GeneticEffectConfig:
    def __init__(self, method, wk_dir, num_init_seq, calibration, trait_num, random_seed, **kwargs):
        self.method = method # 'user_input' | 'randomly_generate'
        self.wk_dir = wk_dir 
        self.num_init_seq = num_init_seq # number of seeds to generate
        self.calibration = calibration # whether to do calibration
        self.trait_num = trait_num # number of traits (dict[trait_category] = number of traits of that category)
        self.random_seed = random_seed
        self.params = kwargs # csv, nv, bs, taus, s, Rs, var_target, caliberation_link, 
        # func (n | st | l), link

    def validate(self):
        if not os.path.exists(self.wk_dir):
            raise CustomizedError(f"Working directory {self.wk_dir} does not exist.")
        if self.method not in ("user_input", "randomly_generate"):
            raise CustomizedError(f"{self.method} isn't a valid method. Please provide a permitted method. "
                            "(user_input/randomly_generate)")
        if self.num_init_seq <= 0:
            raise CustomizedError("Seed size must be positive.")
        if len(self.trait_num.keys()) != 2:
            raise CustomizedError("Please specify exactly 2 kinds of traits' quantities in a list (-trait_n for transmissibility and drug resistance)")
        if sum(self.trait_num.values()) < 1:
            raise CustomizedError("Please provide a list of trait quantities (-trait_n) that sums up to at least 1")

        if self.method=="randomly_generate":
            if self.params.get("site_frac") == []:
                self.params["site_frac"] = [DEFAULT_MU for _ in range(sum(self.trait_num.values()))]
            if len(self.params.get("site_frac")) != sum(self.trait_num.values()):
                raise CustomizedError("If you wish to sample causal sites from the candidate regions, "
                        f"Please provide the expected fraction of causal sites for each trait (-site_frac) with the same length as your trait quantities({sum(self.trait_num.values())})")
            if any(x <= 0 or x >= 1 for x in self.params.get("site_frac")):
                raise CustomizedError("The expected fraction for causal site sampling has to be within (0, 1).")
            if self.params.get("site_disp") <= 0:
                raise CustomizedError("The dispersion of causal site fraction (-site_disp) has to be positive")

            if self.params.get("func") not in ("n", "l", "st"):
                raise CustomizedError(f"{self.params.get("func")} isn't a valid method for sampling effect sizes. Please choose a permitted method."
                                 "(n/l/st) for -func")

            if self.params.get("func") == "n":
                if len(self.params.get("taus")) != sum(self.trait_num.values()):
                    if self.params.get("taus") == []:
                        self.params["taus"] = [DEFAULT_TAUS for _ in range(sum(self.trait_num.values()))]
                    else:
                        raise CustomizedError(f"The given length of the variance (-taus) {self.params.get("taus")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the point normal mode.")
                if any(x < 0 for x in self.params["taus"]):
                    raise CustomizedError("If you wish to provide a variance for the normal distribution (-taus)"
                        f" All the entries have to be not negative.")
            if self.params.get("func") == "l":
                if len(self.params.get("bs")) != sum(self.trait_num.values()):
                    if self.params.get("bs") == []:
                        self.params["bs"] = [DEFAULT_BS for _ in range(sum(self.trait_num.values()))]
                    else:
                        raise CustomizedError(f"The given length of the scales (-bs) {self.params.get("bs")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the laplace mode.")
                if any(x < 0 for x in self.params["bs"]):
                    raise CustomizedError("If you wish to provide a scale for the laplace distribution (-bs)"
                        f" All the entries have to be not negative.")
            if self.params.get("func") == "st":
                if len(self.params.get("s")) != sum(self.trait_num.values()):
                    if self.params.get("s") == []:
                        self.params["s"] = [DEFAULT_S for _ in range(sum(self.trait_num.values()))]
                    else:
                        raise CustomizedError(f"The given length of the scales (-s) {self.params.get("s")}"
                            f" do not match the number of traits to be drawn {self.trait_num} in the student's t mode.")
                if any(x < 0 for x in self.params["bs"]):
                    raise CustomizedError("If you wish to provide a scale for the student's t distribution (-s)"
                        f" All the entries have to be not negative.")
                if self.params.get("nv") <=0 :
                    raise CustomizedError(f"The degrees of freedom for the student's t distribution (-nv) {self.params.get("nv")}"
                            f" has to be positive in the student's t mode.")


        if self.params.get("calibration_link"):
            if self.params.get("link") not in ["logit", "cloglog"]:
                raise CustomizedError(f"If you would like to calibrate the link-scale slope,"
                            f" -link value needs to be either 'logit' or 'cloglog'.")
            if self.params.get("Rs") == []:
                if self.params.get("link") == "logit":
                    self.params["Rs"] = np.full(sum(self.trait_num.values()), DEFAULT_R_OHR)
                if self.params.get("link") == "cloglog":
                    self.params["Rs"] = np.concatenate([np.full(self.trait_num["transmissibility"], DEFAULT_R_OHR), 
                        np.full(self.trait_num["drug_resistance"], DEFAULT_R_CLR)])
            elif len(self.params.get("Rs")) != sum(self.trait_num.values()):
                raise CustomizedError("If you wish to provide a odds ratio / hazard ratio per SD for calibration"
                    f"Please provide a list with the same length as your trait quantities({sum(self.trait_num.values())})")
            elif any(x <= 0 for x in self.params["Rs"]):
                raise CustomizedError("If you wish to provide a odds ratio / hazard ratio per SD for calibration"
                    f"The odds ratio or the hazard ratio (-Rs) per SD has to be a list of positive numbers.")


# ------------- Genetic Effect Generation ----------------------- #
class EffectGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        np.random.seed(cfg.random_seed)

    def run(self):
        try:
            self.cfg.validate()
            df_eff = self._build_effect_df()
            print("Effect sizes sampled!")
            seeds, seeds_state = self._compute_seed_traits(df_eff)
            print("Seeding sequences' traits calculated!")
            df_eff = self._rename_columns(df_eff)
            if self.cfg.calibration:
                if self.cfg.params.get("var_target") == []:
                    self.cfg.params["var_target"] = np.full(sum(self.cfg.trait_num.values()), DEFAULT_VTGT)
                elif len(self.cfg.params.get("var_target")) != sum(self.cfg.trait_num.values()):
                    raise CustomizedError("If you wish to provide a target variance for calibration"
                            f"Please provide a list with the same length as your trait quantity {sum(self.cfg.trait_num.values())}")
                df_eff, em_var = self._calibrate(df_eff, seeds_state)
                seeds, seeds_state = self._compute_seed_traits(df_eff)
                print("Effect sizes calibrated and seeding sequences' traits recalculated!")
            if self.cfg.params.get("calibration_link"):
                if not self.cfg.calibration:
                    em_var = self._variance_calc(df_eff, seeds_state)
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
        if self.cfg.method == "randomly_generate":
            candidates = self._read_candregion_csv()
            df_sites = self._select_sites(candidates, self.cfg.params.get("site_frac"), self.cfg.params.get("site_disp"))
            print("Causal sites selected!")
            return self._sample(df_sites)
        else: # elif self.cfg.method == "user_input":
            df = self._read_effsize_csv()
            return(df)

    # ---------- IO ----------
    def _read_candregion_csv(self):
        """
        Returns causal sites provided by the csv file
        """
        csv_path = self.cfg.params.get("csv")
        if not os.path.exists(csv_path):
            raise CustomizedError(
                f"The provided candidate genomic region file '{csv_path}' does not exist."
            )

        df = pd.read_csv(csv_path)
        num_all_traits = (
            self.cfg.trait_num["transmissibility"] + self.cfg.trait_num["drug_resistance"]
        )

        if df.shape[0] == 0:
            print(
                "WARNING: No causal sites were detected from the imported CSV file. "
                "Please check your CSV file if this is not desired, or use the non-genetic option in OutbreakSimulator.",
                flush=True,
            )

        # Need: at least 2 columns for [start, end]
        if num_all_traits > df.shape[1] - 2:
            raise CustomizedError(
                f"The number of traits in the provided candidate regions CSV '{csv_path}' "
                f"isn't sufficient for the required number of traits '{self.cfg.trait_num}'."
            )

        # Ensure numeric for start and end positions of each gene
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="raise")
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="raise")

        cand_causal_sites = {i: [] for i in range(num_all_traits)}

        # For each row (region), expand [start, end] to the list of sites
        for i in range(df.shape[0]):
            start = int(df.iat[i, 0])
            end = int(df.iat[i, 1])
            if end < start:
                # swap or skip; here we swap and warn
                start, end = end, start
                print(f"WARNING: swapped start/end in row {i+1} (end < start).", flush=True)
            this_range = list(range(start, end + 1))  # inclusive end

            # For each trait column j (0-based), check indicator in column j+2 (+2 to skip [start, end] columns)
            for j in range(num_all_traits):
                val = df.iat[i, j + 2]
                try:
                    flag = float(val) > 0
                except Exception:
                    flag = False
                if flag:
                    cand_causal_sites[j].extend(this_range)

        # Deduplicate & sort for stability
        for j in cand_causal_sites:
            cand_causal_sites[j] = sorted(set(cand_causal_sites[j]))

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
        if just_read.shape[0] == 0:
            print("WARNING: No causal sites were detected from the imported CSV file."
                "Please check your CSV file if this is not a desired situation or use the non-genetic option in OutbreakSimulator.")
        return just_read

        

    def _write_outputs(self, df_eff, seeds):
        """
        Write causal_gene_info.csv and seeds_trait_values.csv into wkdir.
        """
        df_eff.to_csv(os.path.join(self.cfg.wk_dir, "causal_gene_info.csv"), index = False)
        seeds.to_csv(os.path.join(self.cfg.wk_dir, "seeds_trait_values.csv"), index = False)
    
    # ---------- Selection & redo ----------
    def _select_sites(self, candidates, frac=0.003, dispersion=100):
        """
        Returns causal sites chosen from the regions

        Parameters:
            candidates (dict[int: list[int]]): Candidate causal sites.
            frac (list[float]): Expected fraction of causal sites within the candidate regions (mu_i)
            dispersion (float): Prior dispersion of site fraction
        """
        # First do permutation:
        trait_ids = list(candidates.keys())
        # Normalize frac to a dict per trait
        if isinstance(frac, (float, int)):
            mu = {tid: float(frac) for tid in trait_ids}
        elif isinstance(frac, list):
            if len(frac) != len(trait_ids):
                raise CustomizedError("Length of 'frac' must match number of traits.")
            mu = {tid: float(frac[k]) for k, tid in enumerate(trait_ids)}
        elif isinstance(frac, dict):
            mu = {tid: float(frac[tid]) for tid in trait_ids}
        else:
            raise CustomizedError("Unsupported type for 'frac'.")

        permuted = list(np.random.permutation(trait_ids))

        already_selected = set()
        trait_sites = {tid: [] for tid in trait_ids}

        for tid in permuted:
            # Available candidates after excluding sites already used by previous traits
            available = sorted(set(candidates[tid]) - already_selected)
            sites_num = len(available)

            if sites_num == 0:
                print(
                    f"WARNING: No available candidate sites remain for trait {tid}; skipping.",
                    flush=True,
                )
                continue

            # Beta–Binomial draw for K_i
            a_i = mu[tid] * dispersion
            b_i = (1.0 - mu[tid]) * dispersion
            # Guard against invalid a/b if mu is 0 or 1
            a_i = max(a_i, 1e-12)
            b_i = max(b_i, 1e-12)
            pi_i = float(np.random.beta(a_i, b_i))

            print("start while")
            # Draw until K_i >= 1
            K_i = 0
            while K_i == 0:
                K_i = int(np.random.binomial(sites_num, pi_i))
            
            print("end repeat")

            # Sample indices without replacement
            if K_i >= sites_num:
            # Use all available, warn if capped
                if K_i > sites_num:
                    print(
                        f"WARNING: Drawn K_i={K_i} exceeds available={sites_num} for trait {tid}; using all available.",
                        flush=True,
                    )
                selected_idx = list(range(sites_num))
            else:
                selected_idx = np.random.choice(sites_num, K_i, replace=False).tolist()

            chosen = [available[k] for k in selected_idx]
            trait_sites[tid] = chosen
            already_selected.update(chosen)

        # Build tidy DataFrame of unique sites and trait indicators
        all_sites = sorted(already_selected)
        df_out = pd.DataFrame({"Sites": all_sites})

        for tid, site_list in trait_sites.items():
            df_out[f"trait_{tid}"] = df_out["Sites"].isin(site_list).astype(int)

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
        func = self.cfg.params.get("func") # default is n if nothing is given

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
            n (int): Number of effect sizes to draw using this hyperparameter b
            b (float): Scale of the Laplace distribution
        """
        return np.random.laplace(0, b, size=n)

    def _studentst(self, n, scale=1, nv=3):
        """
        Draw n effect sizes for one trait from a Student's t distribution.

        Parameters:
            n (int): Number of effect sizes to draw using this hyperparameter scale and nv
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

        df_AF = df_eff.iloc[:, 0].to_frame(name = "Sites")
        for i in range(self.cfg.num_init_seq):
            df_AF[f"seed_{i}"] = 0

        # raise exception if we do not have access to VCF of individual seeds
        if not os.path.exists(seeds_vcf_dir):
            print("WARNING: seed_generator.py hasn't been run. "
                    "If you want to use seed sequence different from the reference genome, "
                    "you must run seed_generator first - NOW you are regarding reference genome as seeding sequences", flush = True)
            empty_data = {"Seed_ID": [f"seed_{i}" for i in range(self.cfg.num_init_seq)],
            **{trait: [0] * self.cfg.num_init_seq for trait in trait_cols}} # seed X trait

            return pd.DataFrame(empty_data), df_AF

        else:
            seeds = sorted([f for f in os.listdir(seeds_vcf_dir) if f.endswith(".vcf")])
            if len(seeds) > self.cfg.num_init_seq:
                print(f"WARNING: More seeding sequences ({len(seeds)}) than the specified number ({self.cfg.num_init_seq}) "
                    f"are detected. Only the first {self.cfg.num_init_seq} files will be used", flush = True)
            all_effpos = df_eff["Sites"].tolist()

            for seed_idx , seed_file in enumerate(seeds[:self.cfg.num_init_seq]):
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
            df_out["Seed_ID"] = list(range(self.cfg.num_init_seq))
            df_out = df_out[["Seed_ID"] + trait_cols]
            return df_out, df_AF # seed X trait, site X seed


    def _variance_calc(self, df_eff: pd.DataFrame, seeds_state: pd.DataFrame) -> pd.Series:
        """
        Calculate the empirical seeding population trait variances.
        0 variance will be fixed using uniform beta prior

        Parameters:
            df_eff (dataframe): Uncalibrated effect size data frame
            seeds_state (dataframe): df_AF, mutation state of the seeds

        Returns
            var_empirical : pd.Series
        """
        geno = seeds_state.iloc[:, 1:].to_numpy(dtype=float)  # (n_sites, num_init_seqs)
        eff = df_eff.iloc[:, 1:].to_numpy(dtype=float)        # (n_sites, n_traits)

        # Compute allele frequency and center genotype
        AF_all = geno.mean(axis=1, keepdims=True)             # (n_sites, 1)
        center_geno = geno - AF_all                           # (n_sites, num_init_seqs)

        # Compute centered trait matrix: (num_init_seqs, n_traits)
        center_trait = center_geno.T @ eff                    # (num_init_seqs, n_traits)

        # Empirical variance of each trait
        var_empirical = (center_trait ** 2).sum(axis=0) / (geno.shape[1] - 2)
        var_empirical = pd.Series(var_empirical, index=df_eff.columns[1:])


        for col in var_empirical.index:
            if np.isclose(var_empirical.loc[col], 0.0):
                var_empirical.loc[col] = df_eff[col].pow(2).sum() * EXP_BETAPRIOR

        return var_empirical



    def _calibrate(self, df_eff: pd.DataFrame, seeds_state: pd.DataFrame) -> pd.DataFrame:
        """
        Calibration of effect sizes

        Parameters:
            df_eff (dataframe): Uncalibrated effect size data frame
            seeds_state (dataframe): df_AF, mutation state of the seeds

        Returns
            df_eff_calibrated : pd.DataFrame
            Calibrated effect sizes with the same structure as df_eff.
            var_empirical : pd.Series
            Empirical variance of each trait before calibration.
        """
        var_empirical = self._variance_calc(df_eff, seeds_state)

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

        df_eff_calibrated = df_eff.copy()
        df_eff_calibrated.iloc[:, 1:] = df_eff.iloc[:, 1:] * c_i

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
            raise CustomizedError(f"Unknown link_type: {link_type}")

        # Split by trait type
        alpha_trans = alphas[:trans_num] if trans_num else np.array([])
        alpha_drug  = alphas[trans_num:] if drug_num else np.array([])

         # ---- Printing section ----
        print(f"The calibrated link-scale slopes under the {link_type} link are as follows.")

        for i, val in enumerate(alpha_trans):
            label = f"transmissibility_{i+1}"
            print(f"  {label}: {0 if np.isnan(val) else f'{val:.4f}'}")

        for i, val in enumerate(alpha_drug):
            label = f"drug_resistance_{i+1}"
            print(f"  {label}: {0 if np.isnan(val) else f'{val:.4f}'}")

        
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


def effsize_generation_byconfig(all_config):
    """
    Generates effect size file and compute seeds' trait values based on a provided config file.

    Parameters:
        all_config (dict): A dictionary of the configuration (read with read_params()).
    """

    genetic_config = all_config["GenomeElement"]
    wk_dir = all_config["BasicRunConfiguration"]["cwdir"]
    rand_seed = all_config["BasicRunConfiguration"].get("random_number_seed", None)
    num_seed = all_config["SeedsConfiguration"]["seed_size"]
    
    try:
        config = GeneticEffectConfig(
            method = genetic_config["effect_size"]["method"],
            wk_dir = wk_dir,
            num_init_seq = num_seed,
            func = genetic_config["effect_size"]["effsize_params"]["effsize_function"],
            calibration = genetic_config["effect_size"]["calibration"]["do_calibration"],
            random_seed = rand_seed,
            csv = genetic_config["effect_size"]["csv_path"],
            trait_num = genetic_config["traits_num"],
            site_frac = genetic_config["effect_size"]["causalsites_params"]["exp_fraction"],
            site_disp = genetic_config["effect_size"]["causalsites_params"]["fraction_dispersion"],
            taus = genetic_config["effect_size"]["effsize_params"]["normal"]["taus"],
            bs = genetic_config["effect_size"]["effsize_params"]["laplace"]["bs"],
            nv = genetic_config["effect_size"]["effsize_params"]["studentst"]["nv"],
            s = genetic_config["effect_size"]["effsize_params"]["studentst"]["s"],
            var_target = genetic_config["effect_size"]["calibration"]["V_target"],
            calibration_link = genetic_config["trait_prob_link"]["calibration"],
            Rs = genetic_config["trait_prob_link"]["Rs"],
            link = genetic_config["trait_prob_link"]["link"]
        )
    except CustomizedError as e:
        return e

    generator = EffectGenerator(config) # no validation going on so leave it out of the try catch clause
    error = generator.run()
    

    return error


def main():
    parser = argparse.ArgumentParser(description='Generate or modify seeds.')
    parser.add_argument('-method', action='store',dest='method', type=str, choices=['user_input', 'randomly_generate'], required=True, help="Method of the genetic element file generation, using csv or gff")
    parser.add_argument('-wkdir', action='store',dest='wkdir', type=str, required=True, help="Working directory")
    parser.add_argument('-csv', action='store',dest='csv', type=str, required=True, help="Path to the user-provided effect size genetic element csv file", default="")
    parser.add_argument('-trait_n', action='store', dest='trait_n', type=ast.literal_eval, required=True, 
        help="Number of traits that user want to generate a genetic architecture for transmissibility and drug resistance, format: '{\"transmissibility\": x, \"drug-resistance\": y}'", default="")
    parser.add_argument('-func', action='store',dest='func', type=str, required=False, choices=['n', 'l', 'st'], help="Function to generate the effect sizes given causal sites. (n/l/st)")
    parser.add_argument('-site_frac','--site_frac', nargs='+', help='The expected fraction of candidate sites being causal for each trait.', required=False, type=float, default=[])
    parser.add_argument('-site_disp','--site_disp', action='store', help='The dispersion of fraction of candidate sites being causal for each trait.', required=False, type=float, default=100)
    parser.add_argument('-taus','--taus', nargs='+', help='Standard deviation of the effect sizes for each trait under the point normal model. Required when func=n', required=False, type=float, default=[])
    parser.add_argument('-bs','--bs', nargs='+', help='Scales of the laplace distribution of the effect sizes for each trait under the Laplace model. Required when func=l', required=False, type=float, default=[])
    parser.add_argument('-nv','--nv', action='store', help='Degree of freedom of the Student\'s t\'s distribution of the effect sizes for each trait under the student\'s t model. Optional when func=st', required=False, type=float, default=3)
    parser.add_argument('-s','--s', nargs='+', help='Scales of the Student\'s t distribution of the effect sizes for each trait under the Student\'s t model. Required when func=st', required=False, type=float, default=[])
    parser.add_argument('-random_seed', action = 'store', dest = 'random_seed', required = False, type = int, default = None)
    parser.add_argument('-num_init_seq', action='store', dest = 'num_init_seq', required = True, type = int, default = 1)
    parser.add_argument('-calibration', action='store',dest='calibration', type=str2bool, help="Whether to calibrate the effect size values", default=None)
    parser.add_argument('-var_target', '--var_target', nargs='+', help='The target variance of the seeds\' genetic values', required=False, type=float, default=[])
    parser.add_argument('-calibration_link', action='store',dest='calibration_link', type=str2bool, required=False, help="Whether to calibrate the link-scale slope", default=False)
    parser.add_argument('-Rs', '--Rs', nargs='+', help='The odds ratio for the transmission/survival per SD of trait values under logit, or the hazard ratio per SD under cloglog', required=False, type=float, default=[])
    parser.add_argument('-link', action='store',dest='link', type=str, required=False, choices=['logit', 'cloglog'], help="Link type: logit or cloglog", default="logit")

    args = parser.parse_args()

    if args.calibration is None:
        args.calibration = True if args.method == 'randomly_generate' else False

    config = GeneticEffectConfig(
        method = args.method,
        wk_dir = args.wkdir,
        num_init_seq = args.num_init_seq,
        calibration = args.calibration,
        random_seed = args.random_seed,
        func = args.func,
        csv = args.csv,
        trait_num = args.trait_n,
        site_frac = args.site_frac,
        site_disp = args.site_disp,
        taus = args.taus,
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