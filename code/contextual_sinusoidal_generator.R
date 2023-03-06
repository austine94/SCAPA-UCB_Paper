#contextual sinusoidal scenario generator

contextual_sinusoidal_generator <- function(time_horizon, K, m, t_min, t_max,
                                            delta_coeff = 5, frequency = 1,
                                            overlap = FALSE, training = 100){
  #function to generate random bandit scenarios with polynomial regression models
  #does not generate overlapping changes automatically
  #uses gaussian noise
  
  #time_horizon is the length of the time frame
  #K is the number of arms
  #m is the number of changes
  #t_min and t_max denote the domain for the polynomial
  #delta_coeff is the max value of the coefficients
  #frequency is the frequency of the sinusoid
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
  
  if(!is.numeric(t_min)){
    stop("t_min is a numeric for the lowest value for the domain of the polynomial")
  }
  
  if(!is.numeric(t_max)){
    stop("t_max is a numeric for the greatest value for the domain of the polynomial")
  }
  
  if(t_min >= t_max){
    stop("t_min and t_max do not form an ordered interval")
  }
  
  if(!is.numeric(delta_coeff) | delta_coeff <= 0 ){
    stop("delta_coeff must be a positive numeric")
  }
  
  if(!is.numeric(frequency) | round(frequency) != frequency | frequency < 1){
    stop("frequency must be a positive integer")
  }
  
  if( !is.logical(overlap)){
    stop("overlap must be either true or false")
  }
  
  if(!is.na(training)){
    if(!is.numeric(training) | training < 1 | round(training) != training){
      stop("training must be either NA or the size of the training dataset")
    }
  }else{
    training_features <- training_rewards <- NA
  }
  
  time <- seq(t_min, t_max, length = 1000)
  
  input_mat <- runif(time_horizon, t_min, t_max)
  training_input <- runif(training, t_min, t_max)
  
  sinusoid <- function(x, sinusoid_coeffs) { #for generating underlying shape
              sinusoid_coeffs[1] * sin(2 * pi * x / frequency) +
              sinusoid_coeffs[2] * cos(2 * pi * x / frequency)
              }
  
  reward_mat <- matrix(NA, nrow = time_horizon, ncol = K)
  training_rewards <- matrix(NA, nrow = training, ncol = K)
  coeff_add <- runif(K, 0, delta_coeff)
  
  #create test and train data using underlying shape
  
  for(k in 1:K){
    sinusoid_coeffs <- runif(2, -1, 1)
    reward_mat[,k] <- sapply(input_mat, sinusoid, sinusoid_coeffs = sinusoid_coeffs)
    reward_mat[,k] <- reward_mat[,k] + coeff_add[k]
    
    training_rewards[,k] <- sapply(training_input, sinusoid, sinusoid_coeffs = sinusoid_coeffs)
    training_rewards[,k] <- training_rewards[,k] + coeff_add[k]
    
  }
  
  training_noise <-  rnorm( (training * K), 0, 1) %>% matrix(nrow = training, ncol = K)
  training_rewards <-  training_noise + training_rewards #add noise to training matrix
  
  if(m == 0){  #if no changes to insert, just add noise
    
    noise_mat <- rnorm( (time_horizon * K), 0, 1) %>% matrix(nrow = time_horizon, ncol = K)
    reward_mat <- reward_mat + noise_mat + 8 #adding 8 makes it VERY unlikely to be -ve reward
    
    return(list(change_locations = NA, input_mat = input_mat, reward_mat = reward_mat,
                training_input = training_input, training_rewards = training_rewards))
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
          reward_mat[ ((change_times[i] + 1):change_times[i+1]), k ] <-
            reward_mat[ ((change_times[i] + 1):change_times[i+1]), k ] + runif(1, 0, delta_coeff)
        }
      }
      
    }
    
    
  }else{
    change_points <- sample(1:(time_horizon-1), m)  %>% sort() #times of changes
    change_times <- c(0, change_points, time_horizon)
    segment_lengths <- diff(change_times)
    arms_to_change <- sample(1:K, (m+1), replace = TRUE)
    
    for(i in 1:(m+1)){
      reward_mat[ ((change_times[i] + 1):change_times[i+1]), arms_to_change[i] ] <-
        reward_mat[ ((change_times[i] + 1):change_times[i+1]), arms_to_change[i] ] +
        runif(1, 0, delta_coeff)
    }
    
    change_locations <- vector("list", K) #store changes by arm
    for(i in 1:m){
      change_locations[[arms_to_change[i]]] <- c(change_locations[[arms_to_change[i]]], change_points[i])
    }
  }
  
  #finally, add noise
  noise_mat <- rnorm( (time_horizon * K), 0, 1) %>% matrix(nrow = time_horizon, ncol = K)
  reward_mat <- reward_mat + noise_mat
  
  return(list(change_locations = change_locations, input_mat = input_mat,
              reward_mat = reward_mat, training_input = training_input,
              training_rewards = training_rewards))
  
}
