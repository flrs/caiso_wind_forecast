library(tidyverse)
library(timetk)

# Load data ----

prod_data <- read_csv('../../data/db_pull_production_data_raw_20201208.csv')
weather_data <- read_csv('../../data/db_pull_weather_data_raw_20201208.csv',
                         col_types = "Tnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn",
                         skip_empty_rows = TRUE)

# Clean data ----

prod_data <- prod_data %>%
  drop_na() %>% 
  arrange('Time')

weather_data <- weather_data %>%
  drop_na() %>% 
  arrange('Time')

prod_data <- prod_data %>% 
  filter_by_time(.start_date = (weather_data %>% head(1) %>% select(Time) %>% pull()))

# Prepare data ----

weather_data_padded <- weather_data %>%
  pad_by_time(.date_var = Time,
              .by="5 min") %>%
  mutate_at(vars(`0_wind_speed_ms`:`9_temp_c`), .funs= ts_impute_vec, period=1)

comb_data <- prod_data %>%
  select(matches("(Time)|(Wind)")) %>%
  left_join(weather_data_padded, by = c("Time" = "Time")) %>%
  select(matches("(Time)|(Wind)|wind"))

comb_data <- comb_data %>%
  drop_na() %>%
  arrange('Time')

comb_data <- comb_data %>%
  mutate(across(`0_wind_speed_ms`:`4_wind_speed_ms`, function(x) x**3))

comb_data_padded <- comb_data %>%
  pad_by_time(.date_var = Time,
              .by="5 min") %>%
  mutate_at(vars(`Wind`:`4_wind_speed_ms`), .funs= ts_impute_vec, period=1)

saveRDS(comb_data_padded, "comb_data_rev1.rds")
