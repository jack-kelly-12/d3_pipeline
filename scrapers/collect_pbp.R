library(baseballr)
library(collegebaseball)
library(tidyverse)
library(httr)
library(rvest)
library(DBI)
library(RSQLite)

get_pbp_data <- function(schedules) {
  all_games_pbp <- data.frame()
  
  if (is.null(schedules) || nrow(schedules) == 0) {
    message("No schedules provided to process")
    return(data.frame())
  }
  
  if (!"contest_id" %in% names(schedules)) {
    message("Error: contest_id column not found in schedules data")
    return(data.frame())
  }
  
  for (i in 1:nrow(schedules)) {
    game_id <- schedules$contest_id[i]
    
    if (is.na(game_id) || is.null(game_id)) {
      message(sprintf("Skipping row %d: Invalid game ID", i))
      next
    }
    
    tryCatch({
      pbp_data <- base_pbp(game_id = game_id)
      message(sprintf('Processing pbp data for game: %s', game_id))
      
      if (is.data.frame(pbp_data) && nrow(pbp_data) > 0) {
        if (all(c("away_text", "home_text") %in% names(pbp_data))) {
          pbp_data <- pbp_data %>% 
            filter(!(grepl("^\\d+-\\d+$", away_text) & grepl("^\\d+-\\d+$", home_text)))
          
          if (nrow(pbp_data) > 0) {
            all_games_pbp <- rbind(all_games_pbp, pbp_data)
          } else {
            message(sprintf("Game ID %s has no valid play-by-play data after filtering", game_id))
          }
        } else {
          message(sprintf("Game ID %s data does not contain expected 'away_text' and 'home_text' columns", game_id))
        }
      } else {
        message(sprintf("Game ID %s returned no data or invalid data structure", game_id))
      }
    }, error = function(e) {
      message(sprintf("Error processing game ID %s: %s", game_id, e$message))
    })
    
    Sys.sleep(0.5)
  }
  
  if (nrow(all_games_pbp) == 0) {
    message("Warning: No play-by-play data was collected for any games")
  }
  
  return(all_games_pbp)
}

base_pbp <- function(game_id = NA_real_, game_pbp_url = NA_character_, ...) {
  if (is.na(game_pbp_url) && is.na(game_id)) {
    stop("No game_info_url or game_id provided")
  }
  
  if (!is.na(game_id) & is.na(game_pbp_url)) {
    url <- paste0("https://stats.ncaa.org/contests/", game_id, "/play_by_play")
  } else {
    url <- game_pbp_url
  }
  
  tryCatch({
    pbp_payload <- xml2::read_html(url)
  }, error = function(e) {
    stop(sprintf("Failed to read URL for game ID %s: %s", game_id, e$message))
  })
  
  table_list <- tryCatch({
    (pbp_payload %>%
       rvest::html_elements("table"))[-c(1,2,3)] %>%
      rvest::html_table()
  }, error = function(e) {
    stop(sprintf("Failed to parse tables for game ID %s: %s", game_id, e$message))
  })
  
  if (length(table_list) == 0) {
    message(sprintf("No play-by-play tables found for game ID %s", game_id))
    return(data.frame())
  }
  
  date_slug <- pbp_payload %>%
    rvest::html_elements("tr:nth-child(4) .grey_text") %>%
    rvest::html_text(trim=T)
  loc_slug <- pbp_payload %>%
    rvest::html_elements("tr:nth-child(5) .grey_text") %>%
    rvest::html_text(trim=T)
  att <- pbp_payload %>%
    rvest::html_elements("tr:nth-child(6) .grey_text") %>%
    rvest::html_text(trim=T)
  
  add_inning_column <- function(df, inning) {
    df$inning <- inning
    return(df)
  }
  
  mapped_table <- lapply(seq_along(table_list), function(i) add_inning_column(table_list[[i]], i))
  mapped_table <- dplyr::bind_rows(mapped_table)
  
  if (nrow(mapped_table) == 0) {
    message(sprintf("No play-by-play data found for game ID %s", game_id))
    return(data.frame())
  }
  
  col_names <- names(mapped_table)
  away_team <- col_names[1]
  home_team <- col_names[3]
  
  mapped_table <- mapped_table %>%
    dplyr::rename(away_des = 1, home_des = 3) %>%
    dplyr::mutate(
      away_team = away_team,
      home_team = home_team,
      game_id = as.numeric(gsub("\\D", "", url)),
      date = substr(date_slug, start = 1, stop = 10),
      year = as.integer(format(as.Date(date, "%m/%d/%Y"), "%Y")),
      away_text = ifelse(away_des != "", away_des, ""),
      home_text = ifelse(home_des != "", home_des, ""),
      away_score = gsub("-.*", "", Score),
      home_score = gsub(".*-", "", Score)
    ) %>%
    dplyr::filter(!grepl("LOB:", away_des) & !grepl("LOB:", home_des))
  
  mapped_table <- mapped_table %>%
    dplyr::select(
      year,
      date,
      game_id,
      inning,
      away_team,
      home_team,
      away_score,
      home_score,
      away_text,
      home_text
    )
  return(mapped_table)
}

get_schedules <- function(output_dir, division, year) {
  file_path <- file.path(output_dir, sprintf("d%d_schedules_%d.csv", division, year))
  
  if (file.exists(file_path)) {
    schedules <- read.csv(file_path)
    
    if ("contest_id" %in% colnames(schedules)) {
      schedules <- schedules[!duplicated(schedules$contest_id), ]
    } else {
      message("contest_id column not found in the schedule file.")
    }
    
    return(schedules)
  } else {
    message(sprintf("Schedule file not found for Division %d, Year %d", division, year))
    return(data.frame())
  }
}

main <- function(working_dir, output_dir) {
  year <- 2025
  setwd(working_dir)
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  pbp_data_list <- list()
  
  for (division in 1:3) {
    cli::cli_alert_info(paste("Processing division:", division))
    div_name <- switch(division,
                      "1" = "d1",
                      "2" = "d2",
                      "3" = "d3")
    
    schedule_path <- file.path(output_dir, paste0(div_name, "_schedules_", year, ".csv"))
    
    if (!file.exists(schedule_path)) {
      cli::cli_alert_warning(sprintf("Schedule file not found: %s - skipping division", schedule_path))
      next
    }
    
    schedules <- read.csv(schedule_path)
    
    pbp_data <- get_pbp_data(schedules)
    
    if (nrow(pbp_data) > 0) {
      pbp_data_list[[division]] <- pbp_data
      
      output_path <- file.path(output_dir, paste0(div_name, "_pbp_", year, ".csv"))
      write.csv(pbp_data, output_path, row.names = FALSE)
      cli::cli_alert_success(paste("Saved play-by-play CSV:", output_path))
    } else {
      cli::cli_alert_warning(sprintf("No play-by-play data collected for division %d", division))
    }
    
    Sys.sleep(2)
  }

  cli::cli_alert_success("Process completed successfully!")
}