library(tidyverse)
library(timetk)

# Load data ----

prod_data <- read_csv('../../data/db_pull_production_data_raw_20201208.csv')
weather_data <- read_csv('../../data/db_pull_weather_data_raw_20201208.csv',
                         col_types = "Tnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn",
                         skip_empty_rows = TRUE)
feat_gross_prod_data <- read_csv('../../data/db_pull_feature_gross_production_20201209.csv')

# Clean data ----

prod_data <- prod_data %>%
  drop_na() %>% 
  arrange('Time')

prod_data <- prod_data %>%
  mutate(
    across(.cols = contains("Wind"),
           .fns = function(x) ifelse(x<0, NA, x))
    )

weather_data <- weather_data %>%
  drop_na() %>% 
  arrange('Time')

feat_gross_prod_data <- feat_gross_prod_data %>%
  drop_na() %>%
  arrange('Time')

prod_data <- prod_data %>% 
  filter_by_time(.start_date = (weather_data %>% head(1) %>% select(Time) %>% pull()))

# Prepare data ----

weather_data_padded <- weather_data %>%
  pad_by_time(.date_var = Time,
              .by="5 min") %>%
  mutate_at(vars(`0_wind_speed_ms`:`9_temp_c`), .funs= ts_impute_vec, period=1)

feat_gross_prod_data_padded <- feat_gross_prod_data %>%
  pad_by_time(.date_var = Time,
              .by="5 min") %>%
  mutate_at(vars(`0_wind`:`4_wind`), .funs= ts_impute_vec, period=1)

comb_data <- prod_data %>%
  select(matches("(Time)|(Wind)")) %>%
  left_join(weather_data_padded, by = c("Time" = "Time")) %>%
  left_join(feat_gross_prod_data_padded, by = c("Time" = "Time")) %>%
  select(matches("(Time)|(Wind)|wind|temp"))

comb_data <- comb_data %>%
  drop_na() %>%
  arrange('Time')

comb_data <- comb_data %>%
  mutate(across(matches("_ms"), function(x) x**3))

comb_data %>%
  plot_acf_diagnostics(Time,
                       Wind,
                       .ccf_vars = `1_wind`)

comb_data_padded <- comb_data %>%
  pad_by_time(.date_var = Time,
              .by="5 min") %>%
  mutate_at(vars(`Wind`:`4_wind`), .funs= ts_impute_vec, period=1)

comb_data_padded <- comb_data_padded %>%
  tk_augment_lags(
    .value = matches("_wind|temp"),
    .lags = c(1, 9, 276)) %>%
  drop_na()

saveRDS(comb_data_padded, "comb_data_rev4.rds")
