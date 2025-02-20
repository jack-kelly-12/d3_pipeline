library(dplyr)
library(purrr)
library(collegebaseball)
library(DBI)
library(baseballr)

ncaa_stats_bulk <- function(year, 
                          type = 'batting', 
                          divisions = 3, 
                          situation = "all") {
  if (year < 2013) {
    stop('Year must be greater than or equal to 2013')
  }
  
  if (!type %in% c("batting", "pitching", "fielding")) {
    stop('Type must be "batting", "pitching", or "fielding"')
  }
  
  teams_lookup <- baseballr:::rds_from_url(
    "https://raw.githubusercontent.com/robert-frey/college-baseball/main/ncaa_team_lookup.rds"
  ) %>%
    dplyr::filter(year == !!year,
                  division %in% !!divisions) %>%
    distinct(team_id, .keep_all = TRUE)
  
  total_teams <- nrow(teams_lookup)
  cli::cli_alert_info(paste("Retrieving", type, "stats for", total_teams, "teams"))
  
  safe_ncaa_stats <- purrr::safely(ncaa_stats)
  
  results <- purrr::map(
    seq_len(nrow(teams_lookup)),
    function(i) {
      team <- teams_lookup[i,]
      
      if (i %% 10 == 0) {
        cli::cli_alert_info(paste("Processing team", i, "of", total_teams))
      }
      
      result <- safe_ncaa_stats(
        team_id = team$team_id,
        year = year,
        type = type,
        situation = situation
      )
      
      if (!is.null(result$error)) {
        cli::cli_alert_warning(paste("Error processing team_id:", team$team_id))
        return(NULL)
      }
      
      if (!is.null(result$result)) {
        result$result <- result$result %>%
          mutate(across(where(is.logical), as.character))
      }
      
      return(result$result)
    }
  )
  
  combined_stats <- results %>%
    purrr::compact() %>%
    dplyr::bind_rows()
  
  cli::cli_alert_success(paste("Retrieved stats for", 
                             nrow(combined_stats), 
                             "players across",
                             length(unique(combined_stats$team_id)),
                             "teams"))
  
  return(combined_stats)
}

main <- function(working_dir, output_dir, year) {
  setwd(working_dir)
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  batting_stats <- list()
  pitching_stats <- list()
  
  for (division in 1:3) {
    cli::cli_alert_info(paste("Processing division:", division))
    div_name <- switch(division,
                      "1" = "d1",
                      "2" = "d2",
                      "3" = "d3")
    
    batting <- ncaa_stats_bulk(year = year, type = "batting", divisions = division)
    batting_stats[[division]] <- batting
    
    output_path <- file.path(output_dir, paste0(div_name, "_batting_", year, ".csv"))
    write.csv(batting, output_path, row.names = FALSE)
    cli::cli_alert_success(paste("Saved batting CSV:", output_path))
    
    pitching <- ncaa_stats_bulk(year = year, type = "pitching", divisions = division)
    pitching_stats[[division]] <- pitching
    
    output_path <- file.path(output_dir, paste0(div_name, "_pitching_", year, ".csv"))
    write.csv(pitching, output_path, row.names = FALSE)
    cli::cli_alert_success(paste("Saved pitching CSV:", output_path))
  }
  
  cli::cli_alert_success("Process completed successfully!")
}