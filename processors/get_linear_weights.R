library(tidyverse)

library(tidyverse)

calculate_college_linear_weights <- function(pbp_data, re24_matrix) {
  n_rows <- nrow(pbp_data)
  event_types <- c("walk", "hit_by_pitch", "single", "double", "triple", "home_run", "out", "other")
  event_counts <- numeric(length(event_types))
  names(event_counts) <- event_types
  event_re24_sums <- numeric(length(event_types))
  names(event_re24_sums) <- event_types
  
  event_lookup <- c(
    "2" = "out", "3" = "out", "6" = "out",
    "14" = "walk", "16" = "hit_by_pitch",
    "20" = "single", "21" = "double",
    "22" = "triple", "23" = "home_run"
  )
  
  get_re <- function(base_cd, outs) {
    base_cd <- as.numeric(as.character(base_cd))
    outs <- as.numeric(as.character(outs))
    
    base_state_map <- c(
      "0" = "_ _ _",           # No runners
      "1" = "1B _ _",          # Runner on first
      "2" = "_ 2B _",          # Runner on second
      "3" = "_ _ 3B",          # Runner on third
      "4" = "1B 2B _",         # Runners on first and second
      "5" = "1B _ 3B",         # Runners on first and third
      "6" = "_ 2B 3B",         # Runners on second and third
      "7" = "1B 2B 3B"         # Bases loaded
    )
    
    base_states <- base_state_map[as.character(base_cd)]
    base_states[is.na(base_states)] <- "___"  # Default to empty bases if NA
    
    base_idx <- match(base_states, re24_matrix$Bases)
    
    out_cols <- c("X0", "X1", "X2")
    outs_idx <- out_cols[pmin(pmax(outs + 1, 1), 3)]
    
    # Get values from matrix using base index and out column
    values <- as.numeric(sapply(seq_along(base_idx), function(i) {
      if (is.na(base_idx[i]) || is.na(outs_idx[i])) return(0)
      re24_matrix[base_idx[i], outs_idx[i]]
    }))
    
    return(values)
  }
  
  # Calculate all RE values at once with proper NA handling
  re_start <- get_re(pbp_data$base_cd_before, pbp_data$outs_before)
  re_end <- c(get_re(pbp_data$base_cd_before[-1], pbp_data$outs_before[-1]), 0)
  
  # Zero out RE end values for inning endings
  re_end[pbp_data$inn_end == 1] <- 0
  
  # Ensure runs_on_play is numeric
  runs_on_play <- as.numeric(as.character(pbp_data$runs_on_play))
  runs_on_play[is.na(runs_on_play)] <- 0
  
  # Calculate RE24 for all plays at once
  re24 <- re_end - re_start + runs_on_play
  
  # Convert event_cd to character once
  events <- event_lookup[as.character(pbp_data$event_cd)]
  events[is.na(events)] <- "other"
  
  # Use table() for counting events
  event_table <- table(factor(events, levels = event_types))
  event_counts[] <- as.vector(event_table)
  
  # Calculate RE24 sums by event type using tapply
  event_sums <- tapply(re24, factor(events, levels = event_types), sum, default = 0)
  event_re24_sums[] <- event_sums
  
  # Create final tibble
  linear_weights <- tibble(
    events = event_types,
    count = event_counts,
    linear_weights_above_average = event_re24_sums / event_counts
  ) |>
    filter(events != "other") |>
    mutate(
      linear_weights_above_average = round(linear_weights_above_average, 3),
      linear_weights_above_outs = linear_weights_above_average - 
        linear_weights_above_average[events == "out"]
    ) |>
    arrange(desc(linear_weights_above_average))
  
  return(linear_weights)
}

calculate_normalized_linear_weights <- function(linear_weights, stats) {
  required_columns <- c("events", "linear_weights_above_outs", "count")
  if (!all(required_columns %in% names(linear_weights))) {
    stop("linear_weights must contain columns: events, linear_weights_above_outs, and count")
  }
  
  woba_scale_exists <- "wOBA scale" %in% linear_weights$events
  
  if (woba_scale_exists) {
    woba_scale_row <- linear_weights[linear_weights$events == "wOBA scale", ]
    linear_weights <- linear_weights[linear_weights$events != "wOBA scale", ]
  }
  
  total_value <- sum(linear_weights$linear_weights_above_outs * linear_weights$count)
  
  total_pa <- sum(linear_weights$count)
  
  denominator <- total_value / total_pa
  
  league_obp <- (sum(stats$H) + sum(stats$BB) + sum(stats$HBP)) / 
    (sum(stats$AB) + sum(stats$BB) + sum(stats$HBP) + sum(stats$SF) + sum(stats$SH))
  
  woba_scale <- league_obp / denominator
  
  normalized_weights <- linear_weights |>
    mutate(
      normalized_weight = linear_weights_above_outs * woba_scale,
      normalized_weight = round(normalized_weight, 3)
    )
  
  woba_scale_row <- data.frame(
    events = "wOBA scale",
    linear_weights_above_outs = NA,
    count = NA,
    normalized_weight = round(woba_scale, 3)
  )
  
  result <- bind_rows(normalized_weights, woba_scale_row)
  
  return(result)
}

calculate_normalized_linear_weights <- function(linear_weights, stats) {
  required_columns <- c("events", "linear_weights_above_outs", "count")
  if (!all(required_columns %in% names(linear_weights))) {
    stop("linear_weights must contain columns: events, linear_weights_above_outs, and count")
  }
  
  woba_scale_exists <- "wOBA scale" %in% linear_weights$events
  
  if (woba_scale_exists) {
    woba_scale_row <- linear_weights[linear_weights$events == "wOBA scale", ]
    linear_weights <- linear_weights[linear_weights$events != "wOBA scale", ]
  }
  
  total_value <- sum(linear_weights$linear_weights_above_outs * linear_weights$count)
  
  total_pa <- sum(linear_weights$count)
  
  denominator <- total_value / total_pa
  
  league_obp <- (sum(stats$H) + sum(stats$BB) + sum(stats$HBP)) / 
    (sum(stats$AB) + sum(stats$BB) + sum(stats$HBP) + sum(stats$SF) + sum(stats$SH))
  
  woba_scale <- league_obp / denominator
  
  normalized_weights <- linear_weights |>
    mutate(
      normalized_weight = linear_weights_above_outs * woba_scale,
      normalized_weight = round(normalized_weight, 3)
    )

    woba_scale_row <- data.frame(
    events = "wOBA scale",
    linear_weights_above_outs = NA,
    count = NA,
    normalized_weight = round(woba_scale, 3)
  )
  
  result <- bind_rows(normalized_weights, woba_scale_row)
  
  return(result)
}

main <- function(data_dir, year) {
  for (division in 1:3) {
    cli::cli_alert_info(paste("Processing division:", division))
    div_name <- switch(division,
                       "1" = "d1",
                       "2" = "d2",
                       "3" = "d3")
    
    pbp_path <- file.path(data_dir, paste0("/play_by_play/", div_name, "_parsed_pbp_", year, ".csv"))
    stats_path <- file.path(data_dir, paste0("/stats/", div_name, "_batting_", year, ".csv"))
    re_path <- file.path(data_dir, paste0("/miscellaneous/", div_name, "_expected_runs_", year, ".csv"))
    
    if (!file.exists(pbp_path)) {
      cli::cli_alert_warning(sprintf("PBP file not found: %s - skipping division", input_path))
      next
    }
    
    pbp_data <- read.csv(pbp_path)
    re24_matrix <- read.csv(re_path)
    stats <- read.csv(stats_path)
    
    cli::cli_alert_info(sprintf("Calculating linear weights for Division %d", division))
    lw <- calculate_college_linear_weights(pbp_data, re24_matrix)
    lw <- calculate_normalized_linear_weights(lw, stats)
    
    output_path <- file.path(data_dir, paste0("miscellaneous/", div_name, "_linear_weights_", year, ".csv"))
    write_csv(lw, output_path)
    cli::cli_alert_success(sprintf("Saved linear weights data to: %s", output_path))
  }
  
  cli::cli_alert_success("Linear weights calculated successfully!")
}

