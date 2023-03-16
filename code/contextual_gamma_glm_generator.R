#contextual gamma glm scenario generator

contextual_gamma_glm_generator <- function(time_horizon, K, m, n_cont, n_discrete,
                                                 n_binary, delta_feature = 5, delta_coeff = 5,
                                                 shape = 10, overlap = FALSE, training = 100){
  #function to generate random bandit scenarios with gamma glms
  #does not generate overlapping changes automatically
  
  #time_horizon is the length of the time frame
  #K is the number of arms
  #m is the number of changes
  #n_cont is the number of continuous features
  #n_discrete, and n_binary are the number of discrete and binary features
  #delta_feature is the max value of the features
  #delta_coeff is the max value of the coefficients
  #shape is the shape parameter for the gamma dist
  #training generates a specific training set for the arms - if NA, no set is returned,
  #otherwise training specifies the size
  
  #we return the change locations (if any) for each arm
  #the feature matrix -- the first column is the intercept
  #and the rewards for each arm
  
  require(tidyverse)
  
  if(!is.numeric(time_horizon) | time_horizon < 1 | round(time_horizon) != time_horizon){
    stop("time_horizon must be a positive integer")
  }
  
  if(!is.numeric(K) | K < 1 | round(K) != K){
    stop("K must be a positive integer")
  }
  
  if(!is.numeric(m) | m < 0 | round(m) != m){
    stop("m must be a non-negative integer")
  }
  
  if( !is.numeric(n_cont) | n_cont < 0 | round(n_cont) != n_cont){
    stop("n_cont must be a non-negative integer")
  }
  
  if( !is.numeric(n_discrete) | n_discrete < 0 | round(n_discrete) != n_discrete){
    stop("n_discrete must be a non-negative integer")
  }
  
  if( !is.numeric(n_binary) | n_binary < 0 | round(n_binary) != n_binary){
    stop("n_binary must be a non-negative integer")
  }
  
  if(!is.numeric(delta_feature) | delta_feature <= 0 ){
    stop("delta_feature must be a positive numeric")
  }
  
  if(!is.numeric(delta_coeff) | delta_coeff <= 0 ){
    stop("delta_coeff must be a positive numeric")
  }
  
  if( !is.logical(overlap)){
    stop("overlap must be either true or false")
  }
  
  if(!is.numeric(shape) | shape <= 0){
    stop("shape must be a positive numeric")
  }
  
  if(!is.na(training)){
    if(!is.numeric(training) | training < 1 | round(training) != training){
      stop("training must be either NA or the size of the training dataset")
    }
  }else{
    training_features <- training_rewards <- NA
  }
  
  total_features <- n_cont + n_discrete + n_binary + 1 #add 1 for intercept
  
  #create feature mat and base coeff mat (no changes)
  #we use smaller values for the coeffs to give greater prob that change segments are "better"
  
  feature_mat <- r_features(time_horizon, n_cont, n_binary, n_discrete, delta_feature)
  #this coeff mat will be overwritten if there are changes
  coeff_mat <- runif((K * total_features), 0, delta_coeff) %>%
    matrix(nrow = total_features, ncol = K)
  
  reward_linear <- feature_mat %*% coeff_mat
  
  reward_mat <- apply(exp(reward_linear), 2, function(x) 
                rgamma(time_horizon, shape = shape, rate =  shape / x))
  
  #create training set
  
  training_features <- r_features(training, n_cont, n_binary, n_discrete, delta_feature)
  training_linear <- training_features %*% coeff_mat  #use same coefficients as pre-change arms
  
  training_rewards <- apply(exp(training_linear), 2, function(x) 
    rgamma(training, shape = shape, rate =  shape / x))
  
  if(m == 0){  #if no changes to insert
    
    return(list(change_locations = NA, feature_mat = feature_mat, reward_mat = reward_mat,
                training_features = training_features, training_rewards = training_rewards))
  }
  
  if(overlap){
    
    #generate change locations for each arm, then replace each segment with new rewards
    
    n_changes_per_arm <- rmultinom(1, m, rep(1/K, K)) %>% as.vector() #randomly assign m changes
    change_locations <- vector("list", K)
    for(k in 1:K){
      
      if(n_changes_per_arm[k] > 0){
        change_locations[[k]] <- sample(1:(time_horizon-1), n_changes_per_arm[k]) %>% sort()
        change_times <- c(0, change_locations[[k]],
                          time_horizon) 
        segment_lengths <- diff(change_times)
        
        for(i in 1:(n_changes_per_arm[k] + 1)){
          new_coeffs <- runif(total_features, 0, delta_coeff) %>% 
            matrix(nrow = total_features, ncol = 1)
          
          new_linear <- feature_mat[ ((change_times[i] + 1):change_times[i+1]), ] %*% new_coeffs
          
          reward_mat[ ((change_times[i] + 1):change_times[i+1]), k ] <-
                apply(exp(new_linear), 2, function(x)
                rgamma((change_times[i+1] - change_times[i]), shape = shape, rate =  shape / x))
        }
      }
      
    }
    
    
  }else{
    change_points <- sample(1:(time_horizon-1), m)  %>% sort() #times of changes
    change_times <- c(0, change_points, time_horizon)
    segment_lengths <- diff(change_times)
    arms_to_change <- sample(1:K, (m+1), replace = TRUE)
    
    for(i in 1:(m+1)){
      new_coeffs <- runif(total_features, 0, delta_coeff) %>% 
        matrix(nrow = total_features, ncol = 1)
      
      new_linear <- feature_mat[ ((change_times[i] + 1):change_times[i+1]), ] %*% new_coeffs
      
      reward_mat[ ((change_times[i] + 1):change_times[i+1]), arms_to_change[i] ] <-
        apply(exp(new_linear), 2, function(x)
          rgamma((change_times[i+1] - change_times[i]), shape = shape, rate =  shape / x))

    }
    
    change_locations <- vector("list", K) #store changes by arm
    for(i in 1:m){
      change_locations[[arms_to_change[i]]] <- c(change_locations[[arms_to_change[i]]], change_points[i])
    }
  }
  
  return(list(change_locations = change_locations, feature_mat = feature_mat,
              reward_mat = reward_mat, training_features = training_features,
              training_rewards = training_rewards))
  
}
