library(tidymodels)
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

enable_parallel <- F
algo_desc <- "cubist_level0"
dataset <- "comb_data_rev8.RDS"

# Validation settings ----

# Validation data is not considered at all for model training in this file.
validation_ratio <- 0.8
test_ratio <- 0.8
cv_n_folds <- 4
lag <- 300

# Model settings ----

# parameters according to:
#  http://localhost:5001/#/experiments/11/runs/ff81d95cee504d4c881698694292dd90
#  http://localhost:5001/#/experiments/11/runs/cf49252a512e43ab8a1c3f4f7ae21f7e
#  http://localhost:5001/#/experiments/11/runs/3afd748874954061938d4c41e2a27151

committees <- c(79,98,86)
neighbors <- c(6,6,8)
max_rules <- c(280,530,169)

model_parameters <- tibble(
  committees = committees,
  neighbors = neighbors,
  max_rules = max_rules
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
  if(x==2) next

  model_spec <- cubist_rules(
    mode = "regression",
    committees = !!(model_parameters %>% slice(x) %>% pull(committees)),
    neighbors = !!(model_parameters %>% slice(x) %>% pull(neighbors)),
    max_rules = !!(model_parameters %>% slice(x) %>% pull(max_rules))
    ) %>%
  set_engine("Cubist")

  # Run model ----

  tags <- list()
  tags['source'] <- system("git rev-parse HEAD", intern=TRUE)

  mlflow_run_info <- mlflow_start_run(experiment_id = mlflow_exp_id,
                                      client = tracker,
                                      tags = tags)

  mlflow_run_id <- mlflow_run_info %>% slice(1) %>% pull(run_id)

  model_name <- paste0("models/", algo_desc, "_", substr(tags['source'], 0, 7), "_", x)

  mlflow_log_param("model_name", model_name, run_id = mlflow_run_id, client = tracker)

  mlflow_log_param("committees", model_parameters %>% slice(x) %>% pull(committees), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("neighbors", model_parameters %>% slice(x) %>% pull(neighbors), run_id = mlflow_run_id, client = tracker)
  mlflow_log_param("max_rules", model_parameters %>% slice(x) %>% pull(max_rules), run_id = mlflow_run_id, client = tracker)

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
    # n_cores <- parallel::detectCores()
    n_cores <- 2
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