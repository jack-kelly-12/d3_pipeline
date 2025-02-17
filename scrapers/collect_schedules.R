library(collegebaseball)
library(dplyr)
library(baseballr)

get_schedules <- function(year = 2025, division = 1) {
  teams_df <- baseballr:::rds_from_url(
    "https://raw.githubusercontent.com/robert-frey/college-baseball/main/ncaa_team_lookup.rds"
  ) %>%
    dplyr::filter(year == !!year,
                  division == !!division) %>%
    distinct(team_id, .keep_all = TRUE)
  
  team_names <- teams_df$team_name
  team_ids <- data.frame(team_name = character(), team_id = numeric(), stringsAsFactors = FALSE)
  all_schedules <- data.frame()
  
  message(sprintf("Processing %d teams for year %d, division %d", length(team_names), year, division))
  
  for (team in team_names) {
    tryCatch({
      id_result <- ncaa_school_id_lookup(team_name = team, season = year)
      if (nrow(id_result) > 0) {
        team_ids <- rbind(team_ids, data.frame(team_name = team, team_id = id_result$team_id[1]))
        message(sprintf("Found ID for %s: %s", team, id_result$team_id[1]))
      }
    }, error = function(e) {
      message(sprintf("Error looking up ID for team %s: %s", team, e$message))
    })
  }
  
  for (id in team_ids$team_id) {
    tryCatch({
      schedule <- ncaa_schedule(team_id = id, year = year)
      if (!is.null(schedule) && nrow(schedule) > 0) {
        schedule$year <- year
        schedule$division <- division
        all_schedules <- rbind(all_schedules, schedule)
        message(sprintf("Got schedule for team ID %s with %d games", id, nrow(schedule)))
      }
    }, error = function(e) {
      message(sprintf("Error getting schedule for team ID %s: %s", id, e$message))
    })
  }
  
  return(all_schedules)
}

safe_dbWriteTable <- function(conn, table_name, data, append = FALSE) {
  data <- data %>%
    mutate(across(where(is.character), as.character),
           across(where(is.numeric), as.numeric),
           across(where(is.integer), as.integer))
  
  if (append) {
    if (!dbExistsTable(conn, table_name)) {
      dbWriteTable(conn, table_name, data)
    } else {
      dbWriteTable(conn, table_name, data, append = TRUE)
    }
  } else {
    dbWriteTable(conn, table_name, data, overwrite = TRUE)
  }
}

main <- function(working_dir, output_dir) {
  year <- 2025
  setwd(working_dir)
  for (division in c(1, 2, 3)) {
    division_schedules <- data.frame()
    dir.create(output_dir, showWarnings = FALSE)
    # Fix: Add div_name definition
    div_name <- paste0("d", division)  # This creates "d1", "d2", "d3"
    file_path <- file.path(output_dir, paste0(div_name, "_schedules_", year, ".csv"))
  
    message(sprintf("Processing Division %d, Year %d", division, year))
    year_schedules <- get_schedules(year = year, division = division) 
      
    if (nrow(year_schedules) > 0) {
      if ("contest_id" %in% names(year_schedules)) {
        year_schedules <- year_schedules %>% distinct(contest_id, .keep_all = TRUE)
      }
      
      write.csv(
        year_schedules,
        file_path,
        row.names = FALSE
      )
      
      division_schedules <- rbind(division_schedules, year_schedules)
      message(sprintf("Added %d games for Division %d, Year %d", 
                      nrow(year_schedules), division, year))
    } else {
      message(sprintf("No schedules found for Division %d, Year %d", division, year))
    }
  }
}