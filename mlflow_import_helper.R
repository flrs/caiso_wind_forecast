library(tidymodels)
options(tidymodels.dark = TRUE)

library(mlflow)

experiment_name <- "rf_tune_rapids"
file_name <- "/Users/florian/Downloads/run6_5_cv.csv"
parameters <- c(
"min_rows_per_node",
"rows_sample",
"max_features"
)
metrics <- c(
  "mae",
  "rmse",
  "r2"
)

allow_git <- T

data <- read.csv(file_name)

tracker <- mlflow_client("http://localhost:5000")

tryCatch(mlflow_create_experiment(
  name = experiment_name,
  client=tracker
), error=function(e){})

mlflow_exp_info <- mlflow_get_experiment(
  name = experiment_name,
  client=tracker
)

tags <- list()
if(allow_git){
  tags['source'] <- system("git rev-parse HEAD", intern=TRUE)
}
tags['file_name'] <- file_name

mlflow_exp_id <- mlflow_exp_info %>% slice(1) %>% pull(experiment_id)

upload_results <- function (content, name) {
  mlflow_run_info <- mlflow_start_run(experiment_id = mlflow_exp_id,
                                      client = tracker,
                                      tags = tags)
  mlflow_run_id <- mlflow_run_info %>% slice(1) %>% pull(run_id)

  for(parameter in parameters){
    mlflow_log_param(parameter, content[parameter] %>% slice(1) %>% pull(parameter), run_id = mlflow_run_id, client = tracker)
  }

  for(metric in metrics){
    mlflow_log_metric(metric, content[metric] %>% slice(1) %>% pull(metric), run_id = mlflow_run_id, client = tracker)
  }

  mlflow_end_run(run_id = mlflow_run_id, client = tracker)
}

for(x in 1:nrow(data)){
  data %>%
    slice(x) %>%
    upload_results()
}