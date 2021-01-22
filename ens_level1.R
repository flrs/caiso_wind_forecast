library(tidymodels)
options(tidymodels.dark = TRUE)

library(modeltime)
library(modeltime.ensemble)
library(rules)

library(tidyverse)
library(lubridate)
library(timetk)

library(future)
library(doFuture)

library(vip)
library(plotly)
library(lhs)

# Set Run Config ----

enable_parallel <- T
algo_desc <- "ens_level1"
dataset <- "comb_data_rev8.RDS"

# Model Settings ----

# Loadings see meta_weight_study.xlsx
loadings <- c(6.74908,4.14138,3.38822,8.65193,0.25778,1.23898,2.38443,1.08196)

# Validation settings ----

validation_ratio <- 0.8
test_ratio <- 0.8
cv_n_folds <- 4
lag <- 300

# Load data ----

comb_data <- readRDS(dataset)

# Arrange data for ML ----

n_samples_total <- nrow(comb_data)
n_samples_train_test <- round(nrow(comb_data)*validation_ratio,0)
n_samples_train <- round(n_samples_train_test*test_ratio,0)
n_samples_test <- n_samples_train_test-n_samples_train

splits <- time_series_split(
  comb_data %>% head(n_samples_train+n_samples_test),
  initial = n_samples_train,
  assess = n_samples_test,
  cumulative = TRUE)

# Define recipe and model ----

model_names <- c(
  "cubist_level0_b2caff6_1.rds",
  "cubist_level0_b2caff6_3.rds",
  "rf_level0_cc50409_1.rds",
  "rf_level0_cc50409_2.rds",
  "rf_level0_cc50409_3.rds",
  "xgboost_level0_9dc6cbe_1.rds",
  "xgboost_level0_9dc6cbe_2.rds",
  "xgboost_level0_9dc6cbe_3.rds"
)

models <- list()
for(model_name in model_names){
  model <- read_rds(paste("models/", model_name, sep=""))
  models <- c(models, list(model))
}

models_tbl <- do.call(combine_modeltime_tables, models)

# Run model ----

tags <- list()
tags['source'] <- system("git rev-parse HEAD", intern=TRUE)
model_name <- paste0("models/", algo_desc, "_", substr(tags['source'], 0, 7), "_", x)

if(enable_parallel){
  registerDoFuture()
  n_cores <- 2
  plan(
    strategy = cluster,
    workers = parallel::makeCluster(n_cores)
  )
}

models_tbl_predicted <- models_tbl %>%
    modeltime_calibrate(testing(splits))

weighted_model_table <- modeltime_table()

weighted_model_table %>%
  add_modeltime_model(
    models_tbl_predicted %>%
      ensemble_weighted(loadings = as.numeric(loadings))
  )

weighted_model_table %>%
  write_rds(paste0(model_name, ".rds"))

if(enable_parallel){
  plan(strategy = sequential)
}