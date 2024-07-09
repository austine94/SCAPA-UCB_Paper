scapa_ucb_contextual_polynomial_no_change <- function(input, rewards, model_mat, lambda, alpha,
                                                       gamma, n_retrain){
  
  #this is the UCB algorithm with a changepoint test on the rewards of an arm using SCAPA
  #we restart the UCB if we get a collective anomaly
  #we remove point anomalies if they are observed
  
  #input is the T x 1 input vector of inputs for the polynomial over the T time steps
  #rewards is a T x K matrix of rewards for the K arms over the T time steps
  #lambda is for the SCAPA threshold
  #gamma is a value between 0 and 1 for the proportion of uniform samples to take
  #n_retrain is the number of steps to take when retraining
  
  #This code assumes that the order of the polynomial stays fixed for all time -
  #the value of this order is extracted from the model_mat input.
  
  require(tidyverse)
  require(ScapaAnomaly)
  
  if(!is.numeric(input)){
    stop("input should be a vector of numerics for the polynomial domain")
  }
  
  if(!is.numeric(rewards) | !is.matrix(rewards)){
    stop("rewards should be a matrix of numeric rewards")
  }
  
  if(!is.numeric(model_mat) | !is.matrix(model_mat)){
    stop("model mat should be a p x K matrix of model coeffs")
  }
  
  if( !is.numeric(alpha) | alpha <= 0){
    stop("alpha must be a positive numeric")
  }
  
  if( !is.numeric(lambda) | lambda <= 0){
    stop("lambda must be a positive numeric")
  }
  
  if( !is.numeric(n_retrain) | n_retrain <= 0 | n_retrain != round(n_retrain)){
    stop("n_retrain must be a positive integer")
  }
  
  beta <- 1e10
  beta_tilde <- 2*lambda
  
  if( !is.numeric(gamma) | gamma <= 0 | gamma > 1){
    stop("gamma must be a numeric between 0 and 1")
  }
  
  time_horizon <- nrow(rewards)  #time horizon
  K <- ncol(rewards)  #number of arms
  
  if(ncol(model_mat) != K){
    stop("must have a model for each arm")
  }
  
  order <- nrow(model_mat) - 1
  
  most_recent_change <- 0
  tau <- rep(0, K)  #most recent changepoint for each arm
  n_plays <- rep(0, K)
  actions <- rep(NA, time_horizon)
  rewards_received <- rep(0, time_horizon)
  change_locations <- anomalous_actions <- c()
  cumulative_reward <- 0
  
  retrain <- FALSE  #for retraining particular model
  retrain_actions <- c()
  
  residual_mat <- matrix(NA, nrow = time_horizon, ncol = K) #to store received rewards per arm
  #and to use for UCB optimisation and change in expected reward detection
  #for use with Scapa
  
  arm_selection_list <- vector("list", length = K) #for storing when arms are chosen
  for(i in 1:K){
    arm_selection_list[[i]] <- NA  #empty values
  }
  
  for(i in 1:time_horizon){
    if(retrain){
      #if enough points ob served, retrain model for this arm. Else, continue to observe arm
      if(retrain_so_far >= n_retrain){
        retrain_rewards <- as.matrix(retrain_rewards, ncol = 1)
        retrain_model <- lm(retrain_rewards ~ poly(retrain_input, order, raw = TRUE)) #retrain model
        model_mat[,action_to_take] <- retrain_model$coefficients #store new model
        retrain <- FALSE
        
        #once retrained we can backcalculate the residuals
        
        retrained_expected <- sapply(retrain_input, poly_eval,
                                     coeffs = model_mat[,action_to_take])
        
        retrained_residuals <- retrain_rewards - retrained_expected
        
        residual_mat[arm_selection_list[[action_to_take]], action_to_take] <- retrained_residuals
        #once retrained we can continue to play
        
        for(k in 1:K){
          ucb_k[k] <- poly_eval(coeffs = model_mat[,k], input[i])
          penalty <- alpha * sqrt( (i - tau[k]) / n_plays[k])
          ucb_k[k] <- ucb_k[k] + penalty
          #use rewards since last changepoint in a specific arm for the penalty
        }
        action_to_take <- which.max(ucb_k)  #choose arm to max expected rewards
        expected_reward <- ucb_k[action_to_take] #store expected reward
        actions[i] <- action_to_take  #update based on choice
        n_plays[action_to_take] <- n_plays[action_to_take] + 1
        arm_selection_list[[action_to_take]] <- c(arm_selection_list[[action_to_take]], i)
        
        residual_reward <- rewards[i, action_to_take] - expected_reward 
        residual_mat[i,action_to_take] <- residual_reward #matrix used SCAPA
        
        rewards_received[i] <- rewards[i,action_to_take] #update rewards
        cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
        
      }else{
        #if not enough to retrain, continue to observe arm until enough 
        retrain_input <- c(retrain_input, input[i])
        retrain_rewards <- c(retrain_rewards, rewards[i, action_to_take])
        rewards_received[i] <- rewards[i,action_to_take] #update rewards
        cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
        retrain_actions <- c(retrain_actions, i)
        retrain_so_far <- retrain_so_far + 1
        
        actions[i] <- action_to_take  #store action taken
        n_plays[action_to_take] <- n_plays[action_to_take] + 1
        arm_selection_list[[action_to_take]] <- c(arm_selection_list[[action_to_take]], i)
        #we count the plays as we do not start change detection until we have retrained,
        #even if we have enough points to do it
      }
      
    }else{
      A <- (i - most_recent_change) %% (floor(sqrt(K / gamma)))  #uniform sampling; gamma is the proportion that are exploration only
      if( A > 0 & A <= K){  #exploration step
        action_to_take <- A
        actions[i] <- action_to_take  #store action taken
        n_plays[action_to_take] <- n_plays[action_to_take] + 1  #update arm played
        arm_selection_list[[action_to_take]] <- c(arm_selection_list[[action_to_take]], i)
        #updates the time indices at which each arm has been played
        
        expected_reward <- poly_eval(input[i], coeffs = model_mat[,action_to_take])
        residual_reward <- rewards[i, action_to_take] - expected_reward
        
        residual_mat[i,action_to_take] <- residual_reward #matrix used SCAPA
        
        rewards_received[i] <- rewards[i,action_to_take] #update rewards
        cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
        
      }else{
        ucb_k <- rep(NA, K)
        for(k in 1:K){
          ucb_k[k] <- poly_eval(coeffs = model_mat[,k], input[i])
          penalty <- alpha * sqrt( (i - tau[k]) / n_plays[k])
          ucb_k[k] <- ucb_k[k] + penalty
          #use rewards since last changepoint in a specific arm for the penalty
        }
        action_to_take <- which.max(ucb_k)  #choose arm to max expected rewards
        expected_reward <- ucb_k[action_to_take] #store expected reward
        actions[i] <- action_to_take  #update based on choice
        n_plays[action_to_take] <- n_plays[action_to_take] + 1
        arm_selection_list[[action_to_take]] <- c(arm_selection_list[[action_to_take]], i)
        
        residual_reward <- rewards[i, action_to_take] - expected_reward 
        residual_mat[i,action_to_take] <- residual_reward #matrix used SCAPA
        
        rewards_received[i] <- rewards[i,action_to_take] #update rewards
        cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
      }
      
      if(n_plays[action_to_take] >= 30){  #use change detection
        reward_vec <- residual_mat[((tau[action_to_take] + 1) : i),action_to_take] %>%
          na.omit() %>% as.vector()
        change_test <- ucb_scapa_contextual_test(reward_vec, beta, beta_tilde)  
        
        if(change_test$point){ 
          #if a point anomaly then either remove this from future UCB rounds
          #or just record as anomaly 
          residual_mat[i,action_to_take] <- NA
          anomalous_actions <- c(anomalous_actions, i)   
        }
        #note we only return point if NO collective anomalies are present
        #as otherwise we act on the collective anomaly and retrain
        if(change_test$collective){  #reset number of plays, and mark most recent change
          #record changepoint time
          arm_selection_list[[action_to_take]] <- na.omit(arm_selection_list[[action_to_take]])
          
          anomaly_time <- arm_selection_list[[action_to_take]][change_test$anomaly_time]
          arm_selection_list[[action_to_take]] <- arm_selection_list[[action_to_take]][change_test$anomaly_time:length(arm_selection_list[[action_to_take]])]
          
          tau[action_to_take] <- anomaly_time
          most_recent_change <- anomaly_time
          change_locations <- c(change_locations, most_recent_change)
          
          n_plays[action_to_take] <- length(arm_selection_list[[action_to_take]]) 
          #only reset local arm, but account for those pulls since the changepoint
          
          retrain <- TRUE  #store the points since the change 
          retrain_so_far <- length(arm_selection_list[[action_to_take]])
          retrain_input <- input[arm_selection_list[[action_to_take]]]
          retrain_rewards <- rewards[arm_selection_list[[action_to_take]]]
          
        }
      }
    }
    
  }
  
  if(length(change_locations) == 0){
    change_locations <- NA
  }
  if(length(anomalous_actions) == 0){
    anomalous_actions <- NA
  }
  
  return_list <- list(cumulative_reward = cumulative_reward, change_locations = change_locations,
                      rewards_received = rewards_received, actions_taken = actions, 
                      anomalous_actions = anomalous_actions, retrain_actions = retrain_actions)
  return(return_list)
}

