library(tidymodels)
library(modeltime)
library(modeltime.ensemble)

library(tidyverse)
library(lubridate)
library(timetk)

library(future)
library(doFuture)

library(vip)
library(plotly)

library(mlflow)

# Set Run Config ----

enable_parallel <- T
algo_desc <- "knn_feat"
dataset <- "comb_data_rev3.RDS"

# Validation settings ----

n_samples_single <- 10000
n_samples_cv <- 10000
cv_n_folds <- 4
train_test_split_ratio <- 0.2
cv_skip_ratio <- 2

# Model settings ----

neighbors <- 10
dist_power <- 10
weight_func <- "optimal"

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

splits <- time_series_split(comb_data %>% head(n_samples_single),
                            assess = n_samples_single*train_test_split_ratio,
                            cumulative = TRUE)

cv_splits <- comb_data %>%
  time_series_cv(
    cumulative = FALSE,
    slice_limit = cv_n_folds,
    initial = n_samples_cv,
    assess = n_samples_cv*train_test_split_ratio,
    skip=n_samples_cv*cv_skip_ratio
  )


# Define recipe and model ----

recipe_spec <- recipe(formula = Wind ~ .,
                      data = training(splits)) %>%
    step_timeseries_signature(Time) %>%
    step_rm(ends_with(".num")) %>%
    step_rm(contains("year")) %>%
    step_rm(contains("day")) %>%
    step_rm(contains("minute")) %>%
    step_rm(contains("hour12")) %>%
    step_rm(contains("second")) %>%
    step_rm(contains("am.pm")) %>%
    step_rm(contains(".iso")) %>%
    step_rm(contains("mweek")) %>%
    step_rm(contains("week2")) %>%
    step_rm(contains("week3")) %>%
    step_rm(contains("week4")) %>%
    step_rm(contains("wday")) %>%
    step_rm(ends_with("xts")) %>%
    step_rm(ends_with("lbl")) %>%
    step_rm("Time")

features <- names(recipe_spec %>% prep() %>% juice())

model_spec <- nearest_neighbor(
  mode = "regression",
  neighbors = neighbors,
  dist_power = dist_power,
  weight_func = weight_func
) %>%
  set_engine("kknn")

# Run model ----

tags <- list()
tags['source'] <- system("git rev-parse HEAD", intern=TRUE)

mlflow_run_info <- mlflow_start_run(experiment_id = mlflow_exp_id,
                                    client = tracker,
                                    tags = tags)

mlflow_run_id <- mlflow_run_info %>% slice(1) %>% pull(run_id)

mlflow_log_param("neighbors", neighbors, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("dist_power", dist_power, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("weight_func", weight_func, run_id = mlflow_run_id, client = tracker)

mlflow_log_param("features", paste(unlist(features), collapse=","), run_id = mlflow_run_id, client = tracker)
mlflow_log_param("n_samples_single", n_samples_single, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("n_samples_cv", n_samples_cv, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("cv_n_folds", cv_n_folds, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("train_test_split_ratio", train_test_split_ratio, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("algo_desc", algo_desc, run_id = mlflow_run_id, client = tracker)
mlflow_log_param("dataset", dataset, run_id = mlflow_run_id, client = tracker)

set.seed(42)
workflow_fit <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec) %>%
  fit(training(splits))

model_tbl <- modeltime_table(
  workflow_fit
)

analysis_metric_set <- metric_set(rmse, rsq, mae)

if(enable_parallel){
  registerDoFuture()
  n_cores <- parallel::detectCores() -1
  plan(
    strategy = cluster,
    workers = parallel::makeCluster(n_cores)
  )
}

set.seed(42)
model_tbl_cv <- model_tbl %>%
  modeltime_fit_resamples(
    cv_splits,
    control = control_resamples(verbose = TRUE, allow_par = enable_parallel)
  )

if(enable_parallel){
  plan(strategy = sequential)
}

model_tbl_predicted <- model_tbl %>%
  modeltime_calibrate(testing(splits))

# Evaluate model ----

# Out-of-Sample

score_oos <- model_tbl_predicted %>%
  modeltime_accuracy(
    metric_set = analysis_metric_set
  )

score_oos

mlflow_log_metric("rmse", score_oos %>% slice(1) %>% pull(rmse),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("rsq", score_oos %>% slice(1) %>% pull(rsq),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("mae", score_oos %>% slice(1) %>% pull(mae),
                  run_id = mlflow_run_id,
                  client = tracker)

# In-Sample

score_is <- model_tbl_predicted %>%
  modeltime_accuracy(
    new_data = training(splits),
    metric_set = analysis_metric_set
  )

score_is

mlflow_log_metric("rmse_insample", score_is %>% slice(1) %>% pull(rmse),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("rsq_insample", score_is %>% slice(1) %>% pull(rsq),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("mae_insample", score_is %>% slice(1) %>% pull(mae),
                  run_id = mlflow_run_id,
                  client = tracker)


# Cross-Validation Out-of-Sample

score_cvs_oos <- model_tbl_cv %>%
  modeltime_resample_accuracy(
    metric_set = analysis_metric_set,
    summary_fns = list(mean=mean, sd=sd)
  )

score_cvs_oos

mlflow_log_metric("rmse_mean_cv", score_cvs_oos %>% slice(1) %>% pull(rmse_mean),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("rsq_mean_cv", score_cvs_oos %>% slice(1) %>% pull(rsq_mean),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("mae_mean_cv", score_cvs_oos %>% slice(1) %>% pull(mae_mean),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("rmse_sd_cv", score_cvs_oos %>% slice(1) %>% pull(rmse_sd),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("rsq_sd_cv", score_cvs_oos %>% slice(1) %>% pull(rsq_sd),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_log_metric("mae_sd_cv", score_cvs_oos %>% slice(1) %>% pull(mae_sd),
                  run_id = mlflow_run_id,
                  client = tracker)

mlflow_end_run(run_id = mlflow_run_id, client = tracker)
