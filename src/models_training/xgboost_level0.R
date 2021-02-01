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
algo_desc <- "xgboost_level0"
dataset <- "comb_data_rev8.RDS"

# Validation settings ----

# Validation data is not considered at all for model training in this file.
validation_ratio <- 0.8
test_ratio <- 0.8
cv_n_folds <- 4
lag <- 300

# Model settings ----

# parameters according to:
#  http://localhost:5001/#/experiments/10/runs/28f7f95605bd460bb62c0ede645a31a4
#  http://localhost:5001/#/experiments/10/runs/148611b6f22548aa992ab157538a5e8d
#  http://localhost:5001/#/experiments/10/runs/8a44965ed00a4ea3a4d5a09f4e0aeca3

learn_rate <- c(0.0960214107678631, 0.104404198674058, 0.1)
mtry <- c(88, 180, 115)
sample_size <- c(0.67659, 0.667839314496766, 0.60283372461802)
tree_depth <- c(3, 4, 3)
stop_iter <- c(10, 10, 10)

model_parameters <- tibble(
  learn_rate = learn_rate,
  mtry = mtry,
  sample_size = sample_size,
  tree_depth = tree_depth,
  stop_iter = stop_iter
)

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

cv_splits <- comb_data %>%
  head(n_samples_train+n_samples_test) %>%
  time_series_cv(
    cumulative = TRUE,
    slice_limit = cv_n_folds,
    initial = n_samples_train,
    assess = round(n_samples_test/cv_n_folds,0),
    skip=round(n_samples_test/cv_n_folds,0),
    lag = lag
  )

# Define recipe and model ----

recipe_spec <- recipe(formula = Wind ~ .,
                      data = training(splits)) %>%
    step_rm("Time")

features <- names(recipe_spec %>% prep() %>% juice())

for(x in 1:nrow(model_parameters)){
  model_spec <- boost_tree(
      mode = "regression",
      learn_rate = !!(model_parameters %>% slice(x) %>% pull(learn_rate)),
      mtry = !!(model_parameters %>% slice(x) %>% pull(mtry)),
      sample_size = !!(model_parameters %>% slice(x) %>% pull(sample_size)),
      tree_depth = !!(model_parameters %>% slice(x) %>% pull(tree_depth))
    ) %>%
    set_engine("xgboost",
               validation = !!(1-test_ratio),
               stop_iter = !!(model_parameters %>% slice(x) %>% pull(stop_iter))
    )

  # Run model ----

  tags <- list()
  tags['source'] <- system("git rev-parse HEAD", intern=TRUE)

  mlflow_run_info <- mlflow_start_run(experiment_id = mlflow_exp_id,
                                      client = tracker,
                                      tags = tags)

  mlflow_run_id <- mlflow_run_info %>% slice(1) %>% pull(run_id)

  model_name <- paste0("models/", algo_desc, "_", substr(tags['source'], 0, 7), "_", x)

  mlflow_log_param("model_name", model_name, run_id = mlflow_run_id, client = tracker)

  mlflow_log_param("learn_rate", model_parameters %>% slice(x) %>% pull(learn_rate), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("mtry", model_parameters %>% slice(x) %>% pull(mtry), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("sample_size", model_parameters %>% slice(x) %>% pull(sample_size), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("tree_depth", model_parameters %>% slice(x) %>% pull(tree_depth), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("stop_iter", model_parameters %>% slice(x) %>% pull(stop_iter), run_id = mlflow_run_id, client = tracker)

  mlflow_log_param("features", paste(unlist(features), collapse=","), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("n_samples_total", n_samples_total, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("n_samples_train_test", n_samples_train_test, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("n_samples_train", n_samples_train, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("n_samples_test", n_samples_train, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("validation_ratio", validation_ratio, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("test_ratio", test_ratio, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("cv_n_folds", cv_n_folds, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("algo_desc", algo_desc, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("dataset", dataset, run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("lag", lag, run_id = mlflow_run_id, client = tracker)

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
    n_cores <- parallel::detectCores()
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

  model_tbl %>%
    write_rds(paste0(model_name, ".rds"))

  mlflow_end_run(run_id = mlflow_run_id, client = tracker)

}