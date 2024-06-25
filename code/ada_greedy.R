#ADA-Greedy - Luo (2018)

ada_greedy <- function(rewards, L, threshold, variation, delta){
  
  require(tidyverse)
  
  if(!is.numeric(rewards) | !is.matrix(rewards)){
    stop("rewards should be a matrix of numeric rewards")
  }
  
  if( !is.numeric(L) | L <= 0){
    stop("L must be a positive numeric")
  }
  
  if( !is.numeric(threshold) | threshold <= 0){
    stop("threshold must be a positive numeric")
  }
  
  
  if( !is.numeric(variation) | variation <= 0){
    stop("variation must be a positive numeric")
  }
  
  if( !is.numeric(delta) | delta <= 0 | delta >= 1){
    stop("delta must be a positive numeric")
  }
  
  time_horizon <- nrow(rewards)  #time horizon
  K <- ncol(rewards)  #number of arms
  I <- max(time_horizon / L, 1)  #number of intervals permitted
  
  actions <- rep(NA, time_horizon)
  rewards_received <- rep(NA, time_horizon)
  cumulative_reward <- 0
  
  #compute mu (for UCB round ) and beta (for nonstationarity detection)
  mu <- min(1/K, (L^(-1/3) * sqrt(log(K/delta)/K)))
  beta <- 2 * sqrt(log(4*time_horizon^2*K/delta)/(mu*I)) + log(4*time_horizon^2*K/delta)/(mu*I)
  
  interval_starts <- 0  #vector to store T_i times - the interval starts
  prev_start <- 0
  t <- 1  #current time point t analyse
  j <- 0
  while(t < time_horizon){
    #this condition means we keep calculating actions until we have got to the time horizon
    j <- j+1
    j_start <- prev_start + 2^(j-1) #for storing the (j dependent) current interval start and end
    j_end <- min(prev_start + 2^j -1, time_horizon)  #interval must end when observations stop
    #compute the best action to take over the interval if we knew the rewards
    oracle_rewards <- colSums(rewards[j_start:j_end, , drop = FALSE])
    oracle_action <- which.max(oracle_rewards)
    for(t in j_start:j_end){ 
      #create values for distribution and sample action from it
      p_vec <- rep(mu, K)
      p_vec[oracle_action] <- p_vec[oracle_action] + (1-K*mu)
      action_dist <- p_vec / sum(p_vec) #create prob distribution to sample from
      actions[t] <- sample(1:K, 1, prob = action_dist) #choose action; most likely oracle action
      rewards_received[t] <- rewards[t,actions[t]]
      cumulative_reward <- cumulative_reward + rewards_received[t]
      
      #test if either max interval length is exceeded or nonstationarity occured:
      if((t >= (prev_start + L)) | nonstationarity_test(rewards, threshold, variation,
                                                        prev_start, j, t, 
                                                        max(oracle_rewards))){
        
        interval_starts <- c(interval_starts, t) #start new interval
        prev_start <- t
        j <- 0  #reset j counter
        break  
      }
      
    }
  }
  return_list <- list(cumulative_reward = cumulative_reward, 
                      rewards_received = rewards_received, 
                      actions_taken = actions,
                      interval_starts = interval_starts)
  #note the interval starts are the times either nonstationarity is detected or when the 
  #current length exceeds the max interval length L. They are not changepoints.
  return(return_list)
}

nonstationarity_test <- function(rewards, threshold, variation, prev_start, j, t, current_max){
  l <- 1
  while(l <= t - prev_start){
    alternative_oracle_max <- colSums(rewards[(t - l +1):t, ,drop = FALSE]) %>% max()
    if(alternative_oracle_max >= 4*(threshold + variation)){
      return(TRUE)
    }else{
      l <- 2*l
    }
  }
  return(FALSE)
}