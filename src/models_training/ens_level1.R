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

library(mlflow)

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

# Set up MLFlow ----

tracker <- mlflow_client("http://localhost:5000")

tryCatch(mlflow_create_experiment(
  name = algo_desc,
  client=tracker
), error=function(e){})

mlflow_exp_info <- mlflow_get_experiment(
  name = algo_desc,
  client=tracker
)

mlflow_exp_id <- mlflow_exp_info %>% slice(1) %>% pull(experiment_id)

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

mlflow_run_info <- mlflow_start_run(experiment_id = mlflow_exp_id,
                                    client = tracker,
                                    tags = tags)

mlflow_run_id <- mlflow_run_info %>% slice(1) %>% pull(run_id)

model_name <- paste0("models/", algo_desc, "_", substr(tags['source'], 0, 7))

mlflow_log_param("model_name", model_name, run_id = mlflow_run_id, client = tracker)

mlflow_log_param("loadings", paste(unlist(loadings), collapse=","), run_id = mlflow_run_id, client = tracker)

mlflow_log_param("model_names", paste(unlist(model_names), collapse=","), run_id = mlflow_run_id, client = tracker)
mlflow_log_param("n_samples_total", n_samples_total, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("n_samples_train_test", n_samples_train_test, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("n_samples_train", n_samples_train, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("n_samples_test", n_samples_train, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("validation_ratio", validation_ratio, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("test_ratio", test_ratio, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("algo_desc", algo_desc, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("dataset", dataset, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("lag", lag, run_id = mlflow_run_id, client = tracker)


if(enable_parallel){
  registerDoFuture()
  n_cores <- 2
  plan(
    strategy = cluster,
    workers = parallel::makeCluster(n_cores)
  )
}

ensemble_fit <- models_tbl %>%
      ensemble_weighted(loadings = as.numeric(loadings))

ensemble_fit %>%
  write_rds(paste0(model_name, ".rds"),compress='bz')

splits_validation <- time_series_split(
  comb_data,
  initial = n_samples_train_test,
  assess = n_samples_total-n_samples_train_test,
  cumulative = TRUE)

ensemble_fit_predicted_oos <- ensemble_fit %>%
  modeltime_calibrate(testing(splits_validation))

score_oos <- ensemble_fit_predicted_oos %>%
  modeltime_accuracy()

mlflow_log_metric("rmse", score_oos %>% slice(1) %>% pull(rmse),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("rsq", score_oos %>% slice(1) %>% pull(rsq),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("mae", score_oos %>% slice(1) %>% pull(mae),
                  run_id = mlflow_run_id,
                  client = tracker)

ensemble_fit_predicted_is <- ensemble_fit %>%
  modeltime_calibrate(training(splits_validation))

score_is <- ensemble_fit_predicted_is %>%
  modeltime_accuracy()

mlflow_log_metric("rmse_insample", score_is %>% slice(1) %>% pull(rmse),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("rsq_insample", score_is %>% slice(1) %>% pull(rsq),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("mae_insample", score_is %>% slice(1) %>% pull(mae),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_end_run(run_id = mlflow_run_id, client = tracker)

if(enable_parallel){
  plan(strategy = sequential)
}