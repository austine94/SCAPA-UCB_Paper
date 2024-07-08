ucb_scapa_contextual_test <- function(reward_vec, beta, beta_tilde, transform = tierney){
  
  #function for testing reward residuals using scapa
  #transform is the function used by scapa on the inputted data
  #default is Tierney - this scales sequentially by sequentially estimated mean and sd
  
  #return(list(collective = FALSE, point = FALSE, anomaly_time = NA ))
  #anomaly_time is for collective only - we sort the point out in the SCAPA-UCB main func
  
  scapa_run <- scapa.uv(reward_vec, beta, beta_tilde, transform = transform, min_seg_len = 5,
                        max_seg_len = 500)
  plot(scapa_run)
  if( sum(scapa_run@anomaly_types) == 0){  #no anomalies
    collective <- point <- FALSE
    anomaly_time <- NA
  }else if( any(scapa_run@anomaly_types == 2)){  #collective anomalies
    collective <- TRUE
    point <- FALSE
    #store start of collective anomaly
    anomaly_time <- min(which(scapa_run@anomaly_types == 2)) #indices with anomaly
  }else if( scapa_run@anomaly_types[length(reward_vec)] == 1){  #only point anomaly
    collective <- FALSE 
    point <- TRUE
    anomaly_time <- NA 
  }else{  #only interested in most recent point being a point anomaly
    collective <- point <- FALSE
    anomaly_time <- NA
  }
  
  return(list(collective = collective, point = point, anomaly_time = anomaly_time ))
}
