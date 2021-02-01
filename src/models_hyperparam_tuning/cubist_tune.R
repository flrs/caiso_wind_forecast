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

library(mlflow)

# Set Run Config ----

enable_parallel <- T
algo_desc <- "cubist_tune"
dataset <- "comb_data_rev8.RDS"
allow_git <- T

# Validation settings ----

n_samples_cv <- 20000
cv_n_folds <- 4
train_test_split_ratio <- 0.2
cv_skip_ratio <- 1
lag <- 300

# Model settings ----

committees <- c(70, 100)
neighbors <- c(0, 9)
max_rules <- c(1, 800)

# Tuning settings ----

grid_size <- 10

# Load data ----

comb_data <- readRDS(dataset)

# Arrange data for ML ----

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
                      data = comb_data) %>%
    step_rm("Time")

features <- names(recipe_spec %>% prep() %>% juice())

model_spec_tune <- cubist_rules(
  mode = "regression",
  committees = tune(),
  neighbors = tune(),
  max_rules = tune()
) %>%
  set_engine("Cubist")

grid_spec <- grid_latin_hypercube(
  parameters(model_spec_tune) %>%
    update(
      committees = committees(range = committees),
      neighbors = neighbors(range = neighbors),
      max_rules = max_rules(range = max_rules)
    ),
  size = grid_size
)

# Run model ----

tags <- list()
if(allow_git){
  tags['source'] <- system("git rev-parse HEAD", intern=TRUE)
}

workflow_tune <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec_tune)

analysis_metric_set <- metric_set(rmse, rsq, mae)

if(enable_parallel){
  registerDoFuture()
  n_cores <- 2
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

  mlflow_log_param("committees", content %>% slice(1) %>% pull("committees"), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("neighbors", content %>% slice(1) %>% pull("neighbors"), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("max_rules", content %>% slice(1) %>% pull("max_rules"), run_id = mlflow_run_id, client = tracker)

  mlflow_log_param("features", paste(unlist(features), collapse=","), run_id = mlflow_run_id, client = tracker)
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

