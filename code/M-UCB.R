m_ucb <- function(rewards, w, b, gamma){
  
  require(tidyverse)
  
  if(!is.numeric(rewards) | !is.matrix(rewards)){
    stop("rewards should be a matrix of numeric rewards")
  }
  
  if( !is.numeric(w) | round(w/2) != (w/2) ){
    stop("w should be an even integer")
  }
  
  if( !is.numeric(b) | b <= 0){
    stop("b must be a positive numeric")
  }
  
  if( !is.numeric(gamma) | gamma <= 0 | gamma > 1){
    stop("gamma must be a numeric between 0 and 1")
  }
  
  time_horizon <- nrow(rewards)  #time horizon
  K <- ncol(rewards)  #number of arms
  
  tau <- 0  #most recent changepoint
  n_plays <- rep(0, K)
  actions <- rep(NA, time_horizon)
  change_locations <- c()
  cumulative_reward <- 0
  
  rewards_received <- matrix(NA, nrow = time_horizon, ncol = K) #to store received rewards
  
  for(i in 1:time_horizon){
    A <- (i - tau) %% (K / gamma)   #uniform sampling; gamma is the proportion that are exploration only
    if( A > 0 & A <= K){  #exploration step
      action_to_take <- A
      actions[i] <- action_to_take  #store action taken
      n_plays[action_to_take] <- n_plays[action_to_take] + 1  #update arm played
      rewards_received[i,action_to_take] <- rewards[i,action_to_take]  #update rewards
      cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
    }else{
      penalty <- sqrt( (2*(i - tau)) / (n_plays))
      ucb_k <- colMeans(rewards_received[((tau+1):i),], na.rm = TRUE) + penalty #use rewards since last changepoint
      action_to_take <- which.max(ucb_k)  #choose arm to max expected rewards
      actions[i] <- action_to_take  #update based on choice
      n_plays[action_to_take] <- n_plays[action_to_take] + 1
      rewards_received[i, action_to_take] <- rewards[i, action_to_take]
      cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
    }
    
    if(n_plays[action_to_take] >= w){  #use change detection
      reward_vec <- rewards_received[((tau + 1) : i),action_to_take] %>% na.omit()
      reward_vec <- reward_vec[(n_plays[action_to_take] - w + 1): n_plays[action_to_take]]
      change_test <- ucb_mosum(reward_vec, w, b)  
      
      if(change_test){  #reset number of plays, and mark most recent change
        n_plays <- rep(0, K)
        tau <- i
        change_locations <- c(change_locations, tau)
      }
    }
  }
  
  rewards_received <- t(rewards_received) %>% as.vector() %>% na.omit() %>% as.vector() #return as single vector
  
  if(length(change_locations) == 0){
    change_locations <- NA
  }
  return_list <- list(cumulative_reward = cumulative_reward, change_locations = change_locations,
                      rewards_received = rewards_received, actions_taken = actions)
  return(return_list)
}
