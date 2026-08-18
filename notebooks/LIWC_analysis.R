#--------------------------------------------------------------------------------------------------------------------#
####---loading packages---####
#--------------------------------------------------------------------------------------------------------------------#

library(dplyr)
library(lmtest)
library(sandwich)

#--------------------------------------------------------------------------------------------------------------------#
####---data pre-processing---####
#--------------------------------------------------------------------------------------------------------------------#

setwd("C:/Users/Giuliano.DESKTOP-NPATJ24/Desktop/PTSD_online_survey/data")

dataset<- read.csv2("dataset_preprocessed.csv", sep=";", header=TRUE, encoding='utf-8')

dataset$condition_pcl_numeric <- ifelse(dataset$condition_pcl == "clin_pcl", 1,
                                        +ifelse(dataset$condition_pcl == "subclinical", 0, NA))
dataset_negnt  <- read.csv2("responsenegnt_LIWC.csv", sep=",", header=TRUE, encoding='utf-8')
dataset_trauma <- read.csv2("responsetrauma_LIWC.csv", sep=",", header=TRUE, encoding='utf-8')
dataset_neutr  <- read.csv2("responseneutr_LIWC.csv", sep=",", header=TRUE, encoding='utf-8')

key <- "ResponseId"

merge_with_suffix <- function(main_df, add_df, suffix, key) {
  main_df[[key]] <- as.character(main_df[[key]])
  add_df[[key]]  <- as.character(add_df[[key]])
  shared <- intersect(names(main_df), names(add_df))
  shared <- setdiff(shared, key)
  add_df <- add_df[, !(names(add_df) %in% shared), drop=FALSE]
  cols_to_rename <- setdiff(names(add_df), key)
  names(add_df)[names(add_df) %in% cols_to_rename] <- paste0(cols_to_rename, suffix)
  merge(main_df, add_df, by=key, all.x=TRUE)
}

dataset_final <- merge_with_suffix(dataset, dataset_negnt, "_negnt", key)
dataset_final <- merge_with_suffix(dataset_final, dataset_trauma, "_trauma", key)
dataset_final <- merge_with_suffix(dataset_final, dataset_neutr, "_neutr", key)

#--------------------------------------------------------------------------------------------------------------------#
####---Identify trauma columns and ensure numeric---####
#--------------------------------------------------------------------------------------------------------------------#

trauma_cols <- grep("_trauma$", names(dataset_final), value=TRUE)
negnt_cols <- grep("_negnt$", names(dataset_final), value = TRUE)
dataset_final[trauma_cols] <- lapply(dataset_final[trauma_cols], as.numeric)
dataset_final[negnt_cols] <- lapply(dataset_final[negnt_cols], as.numeric)
cat_names <- sub("_trauma$", "", trauma_cols)

group0 <- dataset_final[dataset_final$condition_pcl_numeric == 0, c(trauma_cols, negnt_cols), drop=FALSE]
group1 <- dataset_final[dataset_final$condition_pcl_numeric == 1, c(trauma_cols, negnt_cols), drop=FALSE]

#--------------------------------------------------------------------------------------------------------------------#
####---Compare between high and low ptsd---####
#--------------------------------------------------------------------------------------------------------------------#
#Basierend auf dem Review von Quillivic und Kollegen analysieren wir folgende LIWC-Kategorien: 
#use of first person pronouns, depression / anxiety, death, cognitive processing words
t.test(group1$i_trauma, group0$i_trauma)
t.test(group1$anx_trauma, group0$anx_trauma)
t.test(group1$sad_trauma, group0$sad_trauma)
t.test(group1$death_trauma, group0$death_trauma)
t.test(group1$cogproc_trauma, group0$cogproc_trauma)
# Run tests and store p-values
pvals <- c(
  t.test(group1$i_trauma, group0$i_trauma)$p.value,
  t.test(group1$anx_trauma, group0$anx_trauma)$p.value,
  t.test(group1$sad_trauma, group0$sad_trauma)$p.value,
  t.test(group1$death_trauma, group0$death_trauma)$p.value,
  t.test(group1$cogproc_trauma, group0$cogproc_trauma)$p.value
)

# Adjust p-values
p.adjust(pvals, method = "BH")           # Benjamini-Hochberg (FDR)

#vergleich negnt
t.test(group1$i_negnt, group0$i_negnt)
t.test(group1$anx_negnt, group0$anx_negnt)
t.test(group1$sad_negnt, group0$sad_negnt)
t.test(group1$death_negnt, group0$death_negnt)
t.test(group1$cogproc_negnt, group0$cogproc_negnt)
# Run tests and store p-values
pvals <- c(
  t.test(group1$i_negnt, group0$i_negnt)$p.value,
  t.test(group1$anx_negnt, group0$anx_negnt)$p.value,
  t.test(group1$sad_negnt, group0$sad_negnt)$p.value,
  t.test(group1$death_negnt, group0$death_negnt)$p.value,
  t.test(group1$cogproc_negnt, group0$cogproc_negnt)$p.value
)

# Adjust p-values
p.adjust(pvals, method = "BH")           # Benjamini-Hochberg (FDR)



#--------------------------------------------------------------------------------------------------------------------#
####---Compute mean differences and standardized differences---####
#--------------------------------------------------------------------------------------------------------------------#

mean_diff   <- numeric(length(trauma_cols))
std_diff    <- numeric(length(trauma_cols))
PTSD0_mean  <- numeric(length(trauma_cols))
PTSD1_mean  <- numeric(length(trauma_cols))

for (i in seq_along(trauma_cols)) {
  x0 <- group0[[trauma_cols[i]]]
  x1 <- group1[[trauma_cols[i]]]
  
  x0 <- x0[!is.na(x0)]
  x1 <- x1[!is.na(x1)]
  
  if(length(x0) < 2 || length(x1) < 2 || var(x0) == 0 || var(x1) == 0) {
    mean_diff[i] <- NA
    std_diff[i]  <- NA
    PTSD0_mean[i] <- mean(x0, na.rm=TRUE)
    PTSD1_mean[i] <- mean(x1, na.rm=TRUE)
    next
  }
  
  mean_diff[i] <- mean(x1, na.rm=TRUE) - mean(x0, na.rm=TRUE)
  pooled_sd <- sqrt(((length(x0)-1)*var(x0, na.rm=TRUE) + (length(x1)-1)*var(x1, na.rm=TRUE)) / (length(x0)+length(x1)-2))
  std_diff[i]  <- mean_diff[i] / pooled_sd
  PTSD0_mean[i] <- mean(x0, na.rm=TRUE)
  PTSD1_mean[i] <- mean(x1, na.rm=TRUE)
}

ptsd_table <- data.frame(
  Category   = cat_names,
  PTSD0_Mean = PTSD0_mean,
  PTSD1_Mean = PTSD1_mean,
  MeanDiff   = mean_diff,
  StdDiff    = std_diff
)

#--------------------------------------------------------------------------------------------------------------------#
####---Select top 10 by absolute standardized difference---####
#--------------------------------------------------------------------------------------------------------------------#

top10_vars <- head(ptsd_table$Category[order(-abs(ptsd_table$StdDiff))], 10)
top10_cols <- paste0(top10_vars, "_trauma")

#--------------------------------------------------------------------------------------------------------------------#
####---Run robust regression (HC3) and FDR for top 10---####
#--------------------------------------------------------------------------------------------------------------------#

robust_beta <- numeric(length(top10_cols))
robust_p    <- numeric(length(top10_cols))
ci_lower    <- numeric(length(top10_cols))
ci_upper    <- numeric(length(top10_cols))

for(i in seq_along(top10_cols)){
  varname <- top10_cols[i]
  formula_str <- as.formula(paste(varname, "~ PTSD.Risk"))
  model <- lm(formula_str, data=dataset_final)
  robust_test <- coeftest(model, vcov=vcovHC(model, type="HC3"))
  
  robust_beta[i] <- robust_test["PTSD.Risk","Estimate"]
  robust_p[i]    <- robust_test["PTSD.Risk","Pr(>|t|)"]
  ci <- confint(model)["PTSD.Risk",]
  ci_lower[i] <- ci[1]
  ci_upper[i] <- ci[2]
}

# FDR correction
robust_p_FDR <- p.adjust(robust_p, method="fdr")

top10_results <- data.frame(
  Category     = top10_vars,
  Robust_Beta  = robust_beta,
  Robust_p     = robust_p,
  Robust_p_FDR = robust_p_FDR,
  CI_Lower     = ci_lower,
  CI_Upper     = ci_upper
)

# Add flag for significance
top10_results$Significant_FDR <- top10_results$Robust_p_FDR < 0.05

#--------------------------------------------------------------------------------------------------------------------#
####---Merge top 10 robust results back into main table for full view---####
#--------------------------------------------------------------------------------------------------------------------#

ptsd_table_full <- merge(ptsd_table, top10_results, by="Category", all.x=TRUE)

# Fill NAs in Robust columns for non-top10 variables
ptsd_table_full$Significant_FDR[is.na(ptsd_table_full$Significant_FDR)] <- FALSE

# Order by StdDiff for readability
ptsd_table_full <- ptsd_table_full[order(-abs(ptsd_table_full$StdDiff)), ]

#--------------------------------------------------------------------------------------------------------------------#
####---Outputs---####
#--------------------------------------------------------------------------------------------------------------------#

# Full table (all trauma variables, top 10 robust results included)
ptsd_table_full

# Optional: just the significant top-10 variables
ptsd_significant <- ptsd_table_full[ptsd_table_full$Significant_FDR == TRUE, ]
ptsd_significant
