suppressPackageStartupMessages({
  library("ape")
  library("phylobase")
  library("ggtree")
  library("data.table")
  library("ggplot2")
})

# Define global variables
GREY <<- "grey"
NUM_META_COLS_A <<- 6
NUM_META_COLS_B <<- 4

# Function to find ancestors of all nodes in a tree
find_ancestors <- function(g1){
  nodes_sum <- length(nodeId(g1))
  ancestor_ids <- c()
  for (i in 1:nodes_sum) {
    ancestor_id <- ancestor(g1, i)
    ancestor_ids <- c(ancestor_ids, ancestor_id)
  }
  return(ancestor_ids)
}

# Function to assign ID - the original trees' tips were labeled by sample_gen.host_id
assign_id <- function(meta_df, g1, id_value, new_layer, ancestor_ids, nTips_g1, labels_g1) {
  new_new_layer <- c() # Initialize new layer
  for (i in new_layer) {
    ancestor_id <- ancestor(g1, i) # Get the ancestor ID of the current node
    new_new_layer  <- c(new_new_layer, ancestor_id)
    if (i > nTips_g1) { # If the node is an internal node
      child_ids <- which(ancestor_ids == i)
      id_value[i,]$ori_id <- meta_df[which(meta_df$node_id %in% id_value[child_ids,]$ori_id),]$parent_id[1] # Use child's parents original ID
    }
    else { # For tips
      name <- labels_g1[i]
      id_value[i,]$ori_id <- meta_df[which(meta_df$name == name),]$node_id
    }
  }
  
  new_new_layer <- unique(new_new_layer) # Remove duplicates from new layer
  new_new_layer <- new_new_layer[!is.na(new_new_layer)] # Remove NA values we got from nodes whose ancestral nodes are the root

  if (any(is.na(id_value$ori_id))) {   # Check for NA values in ori_id
    if (is.null(new_new_layer)) {  # If new layer is empty that means we reached the root. assign -1 to remaining NA values of root
      id_value[which(is.na(id_value$ori_id)),]$ori_id <- -1
    }
    return(assign_id(meta_df, g1, id_value, new_new_layer, ancestor_ids, nTips_g1, labels_g1)) # Recursively call the function with new layer
  }
  else { # If all ori_id values are assigned, return id_value
    return(id_value)
  }
}

# Function to assign color
assign_color <- function(meta_df, id_value, g1) {
  # Get the size of the tree
  nodes_num <- length(nodeId(g1))
  for (i in 1:nodes_num) {
    ori_id <- id_value[i,]$ori_id
    if (ori_id == -1) { # color the root grey
      id_value[i,]$color <- GREY
    }
    else { # color the internal nodes exluding root and tips by assigned color
      id_value[i,]$color <- meta_df[which(meta_df$node_id == ori_id),]$color_trait
      id_value[i,]$label_nm <- meta_df[which(meta_df$node_id == ori_id),]$name 
    }
  }
  return(id_value)
}

# Function to assign drug resistance
assign_dr <- function(meta_df, id_value, dr_id, n_trans, g1) {
  col_id <- dr_id + NUM_META_COLS_A + n_trans # Get the position of the column of wanted drug resistance
  nodes_num <- length(nodeId(g1)) # Get the size of the tree
  # Construct a table of #rows = nodes_num, |col| = NA
  dr_df <- data.frame(dr = rep(NA, nodes_num))
  colnames(dr_df) <- paste0("dr", dr_id)

  for (i in 1:nodes_num) {
    ori_id <- id_value[i,]$ori_id
    matching_row <- which(meta_df$node_id == ori_id)
    if (length(matching_row) > 0) {
      dr_df[i,1] <- meta_df[matching_row,col_id]
    }
    else {
      dr_df[i,1] <- 0
    }
  }

  # Combine drug resistance data frame with existing data
  id_value = cbind(id_value, dr_df)
  return(id_value)
}

assign_trans <- function(meta_df, id_value, trans_id, g1) {
  col_id <- trans_id + NUM_META_COLS_A  # Get the position of the column of wanted transmissibility
  nodes_num <- length(nodeId(g1)) # Get the size of the tree
  # Construct a table of #rows = nodes_num, |col| = NA
  trans_df <- data.frame(trans = rep(NA, nodes_num))
  colnames(trans_df) <- paste0("trans", trans_id)

  for (i in 1:nodes_num) {
    ori_id <- id_value[i,]$ori_id
    matching_row <- which(meta_df$node_id == ori_id)
    if (length(matching_row) > 0) {
      trans_df[i,1] <- meta_df[matching_row,col_id]
    }
    else {
      trans_df[i,1] <- 0
    }
  }

  # Combine transmissibility data frame with existing data
  id_value = cbind(id_value, trans_df)
  return(id_value)
}

# Function to prepare data for the drug resistance heatmap
prepare_drug_resist_heatmap <- function(color_value_df, g1, n_dr, n_trans) {
  if (length(NUM_META_COLS_B:ncol(color_value_df))==1)
  {
    df_heatmap <- data.frame(dr1 = color_value_df[1:nTips(g1), NUM_META_COLS_B:ncol(color_value_df)])
  }
  else
  {
    df_heatmap <- color_value_df[1:nTips(g1), NUM_META_COLS_B:ncol(color_value_df)]
  }
  rownames(df_heatmap) <- color_value_df[1:nTips(g1),]$label_nm
  return(df_heatmap)
}

# Function to prepare data for the transmissibility heatmap
prepare_trans_heatmap <- function(color_value_df, g1, n_trans) {
  # trans_heatmap <- color_value_df[1:nTips(g1), NUM_META_COLS_B:ncol(color_value_df)]
  if (length(NUM_META_COLS_B:ncol(color_value_df))==1)
  {
    trans_heatmap <- data.frame(trans1 = color_value_df[1:nTips(g1), NUM_META_COLS_B:ncol(color_value_df)])
  }
  else
  {
    trans_heatmap <- color_value_df[1:nTips(g1), NUM_META_COLS_B:ncol(color_value_df)]
  }
  rownames(trans_heatmap) <- color_value_df[1:nTips(g1),]$label_nm
  return(trans_heatmap)
}

# Function to add drug resistance heatmap
add_drug_resistance_heatmap <- function(p2, df_heatmap) {
  p3 <- gheatmap(p2, df_heatmap, offset = 5, width = 0.1, legend_title = "Drug-resistance",
                 colnames_offset_y = -5) + 
      theme(plot.margin = margin(0,0,0.5,1, "cm"))
  p3$scales$scales <- list()
  p3 <- p3 + scale_fill_gradient(low = "white", high = "orange")
  return(p3)
}

# Function to save transmission tree plot
save_transmission_tree_plot <- function(p3, wk_dir, seed_id) {
  if (missing(seed_id)) {
    suppressWarnings(ggsave(file.path(wk_dir, "whole_transmission_tree.pdf"), plot = p3, width = 5, height = 10, dpi = 300))
  } else {
  suppressWarnings(ggsave(file.path(wk_dir, "transmission_tree", paste0("tree.", seed_id - 1, ".pdf")), 
                          plot = p3, width = 5, height = 10, dpi = 300))
  }
}

# Function to plot the transmission tree if it has > 1 tip
plot_transmission_tree_helper <- function(tree, meta_df, n_dr, n_trans, wk_dir, heatmap_trait, seed_id) {
  g1 <- as(tree, 'phylo4')
  nodes_num <- length(nodeId(g1)) # Get the number of nodes
  # Create data frame to store color values
  color_value <- data.frame(color = rep(GREY, nodes_num), ori_id = rep(NA, nodes_num), 
                            label_nm = rep(NA, nodes_num),
                            row.names = nodeId(g1))
  
  # Assign colors to nodes
  ancestor_ids <- find_ancestors(g1)
  nTips_g1 <- nTips(g1)
  labels_g1 <- labels(g1)

  color_value_df = assign_color(meta_df, assign_id(meta_df, g1, color_value, 1:nTips_g1, ancestor_ids, nTips_g1, labels_g1), g1)
  # Compute the starting ID for internal nodes
  start_id <- nTips(g1) + 1
  # Prepare nodes and tip dataframes for ggtree plotting
  rNodeData <- data.frame(color = color_value_df[start_id:nodes_num,]$color,
                          row.names = nodeId(g1, "internal"))
  rTipData <- data.frame(color = color_value_df[1:nTips(g1),]$color)
  
  # If drug resistance is enabled
  
  plot_heatmap=FALSE
  
  # Prepare heatmap dataframe if we want to plot the heatmap
  if (heatmap_trait=="none") {
    cat(paste0("No heatmap on the tree plot.\n"))
  }
  else if (heatmap_trait=="drug_resistance") {
    cat(paste0("Plotting drug resistance traits as heatmap on the tree plot.\n"))
    if (n_dr > 0) {
      for (i in 1:n_dr) { # Assign drug resistance values
        color_value_df <- assign_dr(meta_df, color_value_df, i, n_trans, g1)
        dr_df <- data.frame(dr = color_value_df[start_id:nodes_num, ncol(color_value_df)])
        colnames(dr_df) <- paste0("dr", i)
        rNodeData <- cbind(rNodeData, dr_df)
        rTipData <- cbind(rTipData, color_value_df[1:nTips(g1), ncol(color_value_df)])
      }
      df_heatmap <- prepare_drug_resist_heatmap(color_value_df, g1, n_dr, n_trans)
      plot_heatmap = TRUE
    }
    else {
      cat(paste0("There's no drug_resistance values to plot.\n"))
    }
  }
  else if (heatmap_trait=="transmissibility") {
    cat(paste0("Plotting transmissibility traits as heatmap on the tree plot.\n"))
    if (n_trans > 0) {
      for (i in 1:n_trans) { # Assign transmissibility values
        color_value_df <- assign_trans(meta_df, color_value_df, i, g1)
        trans_df <- data.frame(trans = color_value_df[start_id:nodes_num, ncol(color_value_df)])
        colnames(trans_df) <- paste0("trans", i)
        rNodeData <- cbind(rNodeData, trans_df)
        rTipData <- cbind(rTipData, color_value_df[1:nTips(g1), ncol(color_value_df)])
      }
      df_heatmap <- prepare_trans_heatmap(color_value_df, g1, n_trans)
      plot_heatmap = TRUE
    }
    else {
      cat(paste0("There's no drug_resistance values to plot.\n"))
    }
  }



  # Convert to phylo4d object      
  g2 <- phylo4d(g1)
  # Assign node and tip colors
  nodeData(g2) <- rNodeData
  tipData(g2) <- rTipData

  # Generate the tree plot with ggtree
  p2 <- ggtree(g2, aes(color=I(color)), ladderize = TRUE)

  # Add drug resistance heatmap if applicable
  if (plot_heatmap==TRUE) {
    p3 <- add_drug_resistance_heatmap(p2, df_heatmap)
  }
  else {
    p3 <- p2
  }
  if (missing(seed_id)) {
    save_transmission_tree_plot(p3, wk_dir)
  } else {
    save_transmission_tree_plot(p3, wk_dir, seed_id)
  } 
}

# Plot tranmission tree of individual seed
plot_transmission_tree <- function(seed_id, wk_dir, meta_df, n_trans, n_dr, whole_phylo_output, seed_phylo, heatmap_trait){
# Print message indicating the transmission tree being plotted
  cat(paste0("Plotting seed ", seed_id - 1, "'s transmission tree...\n"))
  # File path to the transmission tree
  trans_tree_file <- file.path(wk_dir, "transmission_tree", paste0(seed_id - 1, ".nwk"))
  # Read transmission tree file
  tree_str <- paste(readLines(trans_tree_file), collapse = "\n")

  # Check if the tree starts with "(" indicating it is NOT empty
  if (startsWith(tree_str, "(")) {
    # Read tree string into tree object if exists
    tree <- read.tree(trans_tree_file)

    # Check if the tree has only on tip, not suitable for visualization
    if (length(tree$tip.label) == 1) {
      cat("Sorry, we do NOT support visualizing single branch tree.\n")
    } else { # Convert the tree into phylo4 object
      plot_transmission_tree_helper(tree, meta_df, n_dr, n_trans, wk_dir, heatmap_trait, seed_id)
    }
    return(tree)
  } else { # If the seed has no progeny sampled
    cat("No samples for this seed's progeny.\n")
    return(NULL)
  }
}

# Main function
main <- function(){
  # All R warnings are gnored
  options(warn = -1)

  # Input tree metadata file
  args <- commandArgs(trailingOnly = TRUE)
  wk_dir <- args[1]
  seed_size <- as.integer(args[2])
  configs_file <- args[3]
  n_trans <- as.integer(args[4])
  n_dr <- as.integer(args[5])
  seed_tree <- args[6]
  heatmap_trait <- args[7]

  # Read metadata
  meta_df <- data.frame(fread(file.path(wk_dir, "transmission_tree_metadata.csv"), sep = ","))
  colnames(meta_df)[6] <- "color_trait"
  meta_df$name <- as.character(meta_df$name)

  # Read configuration file
  slim_config <- read.delim(configs_file, header = FALSE, sep = ":")
  n_gen <- as.integer(slim_config[slim_config$V1 == "n_generation",]$V2)

  # Initialize seed_phylo_output_flag
  whole_phylo_output <- FALSE

  # Read seed tree if exists
  if (file.exists(seed_tree)) {
	seed_phylo <- read.tree(seed_tree)
	  if (length(seed_phylo$tip.label) > 1) {
      whole_phylo_output <- TRUE
    }
  }

  # Iterate over seed size
  for (seed_id in 1:seed_size) {
    # Plot transmission tree
    tree <- plot_transmission_tree(seed_id, wk_dir, meta_df, n_trans, n_dr, whole_phylo_output, seed_phylo, heatmap_trait)
    if(whole_phylo_output) {
      if (is.null(tree)) {
        seed_phylo <- drop.tip(seed_phylo, as.character(seed_id - 1))
      }
      else{
        # Manipulate the seed phylogeny so it is easy to add later on
        edge_dt <- as.data.frame(seed_phylo$edge) # Extract edges
        setDT(edge_dt) # Transform it into data.table
        edge_dt[, edge_idx := as.integer(rownames(edge_dt))] # Add edge_idx column
        tip_dt <- data.table(tip_label = seed_phylo$tip, 
                            edge_idx = sapply(seed_phylo$tip.label, function(tip_label) which.edge(seed_phylo, tip_label)))
        tip_dt <- tip_dt[edge_dt, on = .(edge_idx)] # edge, edge_idx, tip_label
        seed_phylo <- bind.tree(seed_phylo, tree, where = tip_dt[tip_dt$tip_label == seed_id - 1]$V2, 0)
      }
    }
  }

  # Plot whole transmission tree if exists
  if (whole_phylo_output){
    write.tree(seed_phylo, file = file.path(wk_dir, "whole_transmission_tree.nwk"))
    cat("Plotting whole transmission tree including descendants of all seeds.\n")
    plot_transmission_tree_helper(seed_phylo, meta_df, n_dr, n_trans, wk_dir, heatmap_trait)
  }
}


# Execute the plotting
main()