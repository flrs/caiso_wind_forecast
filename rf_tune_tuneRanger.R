# See https://arxiv.org/pdf/1804.03515.pdf

library(tidymodels)
options(tidymodels.dark = TRUE)

library(tuneRanger)
library(mlr)

library(future)
library(doFuture)

# Set Run Config ----

enable_parallel <- T
algo_desc <- "rf_tune_tuneRanger"
dataset <- "comb_data_rev7.RDS"

# Validation settings ----

n_samples_single <- 180000

# Load data ----

comb_data <- readRDS(dataset)

# Arrange data for ML ----

training <- comb_data %>%
  head(n_samples_single) %>%
  select(-Time)

colnames(training) <- paste0("c", colnames(training))
training <- as.data.frame(training)

# Define recipe and model ----

regr.task <- makeRegrTask(data = training, target = "cWind")

# Run model ----

if(enable_parallel){
  registerDoFuture()
  n_cores <- parallel::detectCores()
  plan(
    strategy = cluster,
    workers = parallel::makeCluster(n_cores)
    )
}

res = tuneRanger(regr.task, measure = list(mae), num.trees = 1000, num.threads = n_cores)

if(enable_parallel){
  plan(strategy = sequential)
}

res

res$model
