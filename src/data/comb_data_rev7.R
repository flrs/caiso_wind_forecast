library(tidyverse)
library(timetk)

# Load data ----

prod_data <- read_csv('../../../../data/db_pull_production_data_raw_20201208.csv')
weather_data <- read_csv('../../../../data/db_pull_weather_data_raw_20201208.csv',
                         col_types = "Tnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn",
                         skip_empty_rows = TRUE)
feat_gross_prod_data <- read_csv('../../../../data/db_pull_feature_gross_production_20201209.csv')

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

comb_data %>%
  plot_acf_diagnostics(Time,
                       Wind,
                       .ccf_vars = `Wind`)

comb_data_padded <- comb_data %>%
  pad_by_time(.date_var = Time,
              .by="5 min") %>%
  mutate_at(vars(`Wind`:`4_wind`), .funs= ts_impute_vec, period=1)

comb_data_padded <- comb_data_padded %>%
  tk_augment_lags(
    .value = matches("_wind|temp"),
    .lags = c(1, 2, 3, 4, 5, 9, 276)) %>%
  tk_augment_lags(
    matches("Wind"),
    .lags = 300
  ) %>%
  tk_augment_holiday_signature(
     .date_var = Time,
     .locale_set      = "US",
     .holiday_pattern = "none",
     .exchange_set    = "none"
  ) %>%
  tk_augment_timeseries_signature(Time) %>%
  drop_na()

comb_data_padded <- comb_data_padded %>%
  select(matches("(Time)|(Wind)|(1_wind_speed_ms_lag5)|(1_wind_speed_ms_lag1)|(1_wind_speed_ms_lag3)|(1_wind_lag1)|(1_wind_speed_ms_lag4)|(1_wind)|(1_wind_lag4)|(1_wind_speed_ms_lag2)|(1_wind_lag2)|(1_wind_speed_ms)|(1_wind_lag3)|(1_wind_speed_ms_lag9)|(1_wind_lag9)|(1_wind_lag5)|(3_wind_lag5)|(3_wind_speed_ms_lag2)|(3_wind_speed_ms_lag5)|(3_wind_speed_ms_lag1)|(3_wind_speed_ms_lag4)|(3_wind_lag4)|(3_wind_lag9)|(3_wind_lag2)|(3_wind_lag3)|(3_wind_speed_ms)|(2_wind_lag9)|(3_wind_speed_ms_lag3)|(2_wind_lag2)|(2_wind_lag1)|(3_wind_lag1)|(3_wind)|(2_wind_speed_ms_lag2)|(2_wind_speed_ms_lag5)|(2_wind_lag5)|(3_wind_speed_ms_lag9)|(2_wind_lag3)|(2_wind_speed_ms)|(2_wind_speed_ms_lag3)|(2_wind_lag4)|(Wind_lag300)|(2_wind)|(2_wind_speed_ms_lag9)|(2_wind_speed_ms_lag4)|(4_wind_speed_ms_lag3)|(2_wind_speed_ms_lag1)|(4_wind_speed_ms_lag5)|(1_wind_speed_ms_lag276)|(1_wind_speed_ms_lag300)|(Time_week)|(1_wind_lag4_lag300)|(4_wind_lag9)|(4_wind_speed_ms)|(4_wind_speed_ms_lag2)|(2_wind_lag276)|(1_wind_speed_ms_lag1_lag300)|(4_wind_lag4)|(1_wind_speed_ms_lag4_lag300)|(1_wind_lag9_lag300)|(3_wind_speed_ms_lag5_lag300)|(1_wind_lag3_lag300)|(4_wind_lag2)|(2_wind_speed_ms_lag276)|(4_wind_lag1)|(0_wind_speed_ms_lag5)|(3_wind_lag2_lag300)|(4_wind)|(1_wind_speed_ms_lag5_lag300)|(4_wind_lag3)|(1_wind_speed_ms_lag2_lag300)|(4_wind_lag5)|(4_wind_speed_ms_lag4)|(4_wind_speed_ms_lag9)|(1_wind_lag2_lag300)|(1_wind_lag1_lag300)|(3_wind_speed_ms_lag276)|(2_wind_speed_ms_lag2_lag300)|(4_wind_speed_ms_lag1)|(1_wind_speed_ms_lag3_lag300)|(1_wind_speed_ms_lag9_lag300)|(Time_hour)|(1_wind_lag300)|(2_wind_lag300)|(Time_month)|(3_wind_speed_ms_lag300)|(2_wind_speed_ms_lag5_lag300)|(2_wind_lag1_lag300)|(0_wind_speed_ms_lag2)|(1_wind_lag5_lag300)|(1_wind_lag276)|(3_wind_lag4_lag300)|(3_wind_lag5_lag300)|(3_wind_lag3_lag300)|(3_wind_speed_ms_lag3_lag300)|(0_wind_speed_ms_lag1)|(3_wind_lag1_lag300)|(2_wind_lag9_lag300)|(3_wind_speed_ms_lag1_lag300)|(2_wind_speed_ms_lag1_lag300)|(0_temp_c_lag1)|(0_temp_c_lag9)|(2_wind_lag2_lag300)|(0_temp_c_lag3)|(0_temp_c)|(0_temp_c_lag5)|(2_wind_speed_ms_lag3_lag300)|(2_wind_lag3_lag300)|(Time_day)|(3_wind_speed_ms_lag2_lag300)|(8_temp_c_lag3)|(2_wind_speed_ms_lag4_lag300)|(8_temp_c)|(3_wind_lag276)|(8_temp_c_lag276)|(0_temp_c_lag2)|(1_temp_c)|(3_wind_speed_ms_lag4_lag300)|(0_temp_c_lag4)|(1_temp_c_lag5)|(4_wind_lag276_lag300)|(4_wind_speed_ms_lag276_lag300)|(0_wind_speed_ms_lag9)|(1_temp_c_lag4)|(8_temp_c_lag2)|(3_temp_c_lag276)|(8_temp_c_lag1)"
))
saveRDS(comb_data_padded, "../../data/processed/comb_data_rev7.rds")
