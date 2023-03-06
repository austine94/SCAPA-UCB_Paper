#contextual logistic scenario generator

contextual_logistic_generator <- function(time_horizon, K, m, n_cont, n_discrete,
                                                 n_binary, delta_feature = 5, delta_coeff = 5,
                                                  training = 100){
  #function to generate random bandit scenarios with multinomial logistic models
  #changes occur to the underlying logistic model, so all arms change at once in this setting
  
  #time_horizon is the length of the time frame
  #K is the number of arms / number of discrete outcomes
  #m is the number of changes
  #n_cont is the number of continuous features
  #n_discrete, and n_binary are the number of discrete and binary features
  #delta_feature is the max value of the features
  #delta_coeff is the max value of the coefficients
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
  
  if(!is.na(training)){
    if(!is.numeric(training) | training < 1 | round(training) != training){
      stop("training must be either NA or the size of the training dataset")
    }
  }else{
    training_features <- training_rewards <- NA
  }
  
  #first generate a linear model then transform to logistic regression
  
  total_features <- n_cont + n_discrete + n_binary + 1 #add 1 for intercept
  
  #create feature mat and base coeff mat (no changes)
  #we use smaller values for the coeffs to give greater prob that change segments are "better"
  
  feature_mat <- r_features(time_horizon, n_cont, n_binary, n_discrete, delta_feature)
  
  coeff_mat <- runif((K * total_features), 1, 2) %>%
    matrix(nrow = total_features, ncol = K)
  
  reward_mat <- feature_mat %*% coeff_mat   #creates rewards from features and models
  
  #create training set
  
  training_features <- r_features(training, n_cont, n_binary, n_discrete, delta_feature)
  training_rewards <- training_features %*% coeff_mat  #use same coefficients as pre-change arms
  
  training_noise <-  rnorm( (training * K), 0, 1) %>% matrix(nrow = training, ncol = K)
  training_rewards <-  training_noise + training_rewards #add noise to training matrix
  
  if(m == 0){  #if no changes to insert, just add noise
    
    noise_mat <- rnorm( (time_horizon * K), 0, 1) %>% matrix(nrow = time_horizon, ncol = K)
    reward_mat <- reward_mat + noise_mat + 5 #adding 5 makes it VERY unlikely to be -ve reward
    
    return(list(change_locations = NA, feature_mat = feature_mat, reward_mat = reward_mat,
                training_features = training_features, training_rewards = training_rewards))
  }
  
  change_locations <- sample(2:time_horizon, m, replace = FALSE) %>% sort()
  
  if(m == 1){ #if only one change no need to loop
    new_coeff_mat <- runif((K * total_features), 0, delta_coeff) %>%
                      matrix(nrow = total_features, ncol = K)
    reward_mat[change_locations:time_horizon,] <- feature_mat[change_locations:time_horizon,] %*%
                                                  coeff_mat #new rewards
    
  }else{
    #otherwise fill between changes using loop then fill final segment after
    for(k in 1:(m-1)){
      new_coeff_mat <- runif((K * total_features), 0, delta_coeff) %>%
        matrix(nrow = total_features, ncol = K)
      reward_mat[change_locations[k]:change_locations[(k+1)],] <- feature_mat[change_locations[k]:change_locations[(k+1)],] %*% coeff_mat #new rewards
    }
    new_coeff_mat <- runif((K * total_features), 0, delta_coeff) %>%
      matrix(nrow = total_features, ncol = K)
    reward_mat[change_locations[m]:time_horizon,] <- feature_mat[change_locations[m]:time_horizon,] %*%
      coeff_mat #new rewards
    
  }

  #finally, add noise
  noise_mat <- rnorm( (time_horizon * K), 0, 1) %>% matrix(nrow = time_horizon, ncol = K)
  reward_mat <- reward_mat + noise_mat
  
  #######
  #now turn into a logistic regression model
  #######
  
  logistic_training_reward_mat <- 1 / (1+exp(-training_rewards)) %>% exp() #exp for softmax
  training_p_mat_row_sums <- apply(logistic_training_reward_mat, 1, sum)
  normalized_training_p_mat <- sweep(logistic_training_reward_mat, 1,
                                     training_p_mat_row_sums, "/")
  training_soft_max_vec <- apply(normalized_training_p_mat, 1, which.max) 
  
  #transform the output as per logistic regression link
  logistic_reward_mat <- 1 / (1 + exp(-reward_mat)) %>% exp() #use exp for softmax
  
  #next take the soft max of thelogistic model output to see which outcome is best choice
  p_mat_row_sums <- apply(logistic_reward_mat, 1, sum)
  normalized_p_mat <- sweep(logistic_reward_mat, 1, p_mat_row_sums, "/")
  soft_max_vec <- apply(normalized_p_mat, 1, which.max) 
  
  reward_mat <- matrix(0, nrow = time_horizon, ncol = K)
  for(i in 1:time_horizon){
    reward_mat[i,soft_max_vec[i]] <- 1  #1 indicates this arm is the correct outcome 
  }
  
  return(list(change_locations = change_locations, feature_mat = feature_mat,
              reward_mat = reward_mat, training_features = training_features,
              training_rewards = training_rewards, outcomes = soft_max_vec,
              training_outcomes = training_soft_max_vec))
  
}
