suppressPackageStartupMessages({
  library("Biostrings")
  library("parallel")
  require("data.table")
})


args <- commandArgs(trailingOnly = TRUE)
wk_dir <- args[1]

sample_csv = fread(file.path(wk_dir, "sample.csv.gz"), header=FALSE, sep=" ", col.names = c("generation", "host_id", "seed_id"))

sample_csv[, name := paste0(sample_csv[,generation], ".", sample_csv[,host_id])]
namesneeded = sample_csv$name

vcf_path = file.path(wk_dir, "sampled_pathogen_sequences.vcf")

vcf <- fread(vcf_path)
new_table = vcf[, ..namesneeded]


selected_cols = c(c("POS", "REF", "ALT"), namesneeded)
new_table = vcf[, ..selected_cols]

row_condition <- apply(new_table[, ..namesneeded], 1, function(row) all(row == row[1]))
selected_rows <- new_table[row_condition==FALSE, ]


vcf2fasta <- function(vcf_ind, vcf_info)
{
  real_alt = c()
  for (i in 1:nrow(vcf_info))
  {
    alts = strsplit(vcf_info[i,]$ALT, ",")[[1]]
    if (vcf_ind[i]==0)
    {
      real_alt = c(real_alt, vcf_info[i,]$REF)
    }
    else
    {
      real_alt = c(real_alt, alts[vcf_ind[i]])
    }
  }
  return(DNAString(paste(real_alt, collapse="")))
}

vcf_lists = as.list(selected_rows[, ..namesneeded])

vcf_info_ = selected_rows[, c("POS", "REF", "ALT")]
total_cores <- detectCores()
sampled_genomes <- DNAStringSet(mclapply(vcf_lists, vcf2fasta, vcf_info=vcf_info_, mc.cores = total_cores))
all_names_current = names(sampled_genomes)

write.csv(data.frame(id = 1:nrow(vcf_info_), position = vcf_info_[,"POS"]),
        file.path(wk_dir, "final_samples_snp_pos.csv"), quote=F, row.names=F)

fas_out_path = file.path(wk_dir, "sample.SNPs_only.fasta")

writeXStringSet(sampled_genomes, filepath=fas_out_path, format="fasta", append=TRUE)

