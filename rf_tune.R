library(tidymodels)
options(tidymodels.dark = TRUE)

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
algo_desc <- "rf_tune"
dataset <- "comb_data_rev7.RDS"

# Validation settings ----

n_samples_single <- 10000
n_samples_cv <- 40000
cv_n_folds <- 4
train_test_split_ratio <- 0.2
cv_skip_ratio <- 2
lag <- 300

# Model settings ----

mtry <- c(7, 58)
trees <- c(182, 153)
min_n <- c(9, 39)
max_depth <- c(25, 47)

# Tuning settings ----

grid_size <- 20

# Load data ----

comb_data <- readRDS(dataset)

# Arrange data for ML ----

splits <- time_series_split(
  comb_data %>% head(n_samples_single),
  initial = n_samples_single*(1-train_test_split_ratio),
  assess = n_samples_single*train_test_split_ratio,
  cumulative = TRUE)

cv_splits <- comb_data %>%
  time_series_cv(
    cumulative = FALSE,
    slice_limit = cv_n_folds,
    initial = n_samples_cv,
    assess = n_samples_cv*train_test_split_ratio,
    skip=n_samples_cv*cv_skip_ratio,
    lag = lag
  )


# Define recipe and model ----

recipe_spec <- recipe(formula = Wind ~ .,
                      data = training(splits)) %>%
    step_rm("Time")

features <- names(recipe_spec %>% prep() %>% juice())

model_spec_tune <- rand_forest(
  mode = "regression",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_engine("ranger",
             max.depth = tune(),
             importance="impurity")

grid_spec <- grid_latin_hypercube(
  parameters(model_spec_tune) %>%
    update(
      min_n = min_n(range = min_n),
      trees = trees(range = trees),
      mtry = mtry(range = mtry),
      max.depth = num_terms(range = max_depth)
    ),
  size = grid_size
)

# Run model ----

tags <- list()
tags['source'] <- system("git rev-parse HEAD", intern=TRUE)

workflow_tune <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec_tune)

analysis_metric_set <- metric_set(rmse, rsq, mae)

if(enable_parallel){
  registerDoFuture()
  n_cores <- parallel::detectCores()
  plan(
    strategy = cluster,
    workers = parallel::makeCluster(n_cores)
  )
}

set.seed(42)
tune_results <- workflow_tune %>%
  tune_grid(
    resamples = cv_splits,
    grid = grid_spec,
    metrics = analysis_metric_set,
    control = control_grid(verbose = TRUE,
                           save_pred = FALSE,
                           allow_par = enable_parallel,
                           parallel_over = "everything")
  )

if(enable_parallel){
  plan(strategy = sequential)
}

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

upload_results <- function (content, name) {
  mlflow_run_info <- mlflow_start_run(experiment_id = mlflow_exp_id,
                                    client = tracker,
                                    tags = tags)

  mlflow_run_id <- mlflow_run_info %>% slice(1) %>% pull(run_id)

  mlflow_log_param("mtry", content %>% slice(1) %>% pull("mtry"), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("trees", content %>% slice(1) %>% pull("trees"), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("min_n", content %>% slice(1) %>% pull("min_n"), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("max_depth", content %>% slice(1) %>% pull("max.depth"), run_id = mlflow_run_id, client = tracker)

  mlflow_log_param("features", paste(unlist(features), collapse=","), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("n_samples_single", n_samples_single, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("n_samples_cv", n_samples_cv, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("cv_n_folds", cv_n_folds, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("train_test_split_ratio", train_test_split_ratio, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("algo_desc", algo_desc, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("dataset", dataset, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("lag", lag, run_id = mlflow_run_id, client = tracker)

  content %>%
    rowwise() %>%
    apply(1, function(x){
      mlflow_log_metric(x['.metric'], as.double(x['mean']),
                  run_id = mlflow_run_id,
                  client = tracker)
    })
  mlflow_end_run(run_id = mlflow_run_id, client = tracker)
}

tune_results_groups <- tune_results %>% collect_metrics() %>% group_by(.config)

tune_results_groups %>% group_walk(.f = upload_results)

