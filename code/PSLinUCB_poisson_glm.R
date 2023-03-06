#PSLinUCB (Xue 2020)

pslinucb_poisson <- function(features, rewards, window_size, alpha, threshold){
  
  require(tidyverse)
  
  if(!is.numeric(features) | !is.matrix(features)){
    stop("features should be a matrix of numeric features")
  }
  
  if(!is.numeric(rewards) | !is.matrix(rewards)){
    stop("rewards should be a matrix of numeric rewards")
  }
  
  if(!is.numeric(window_size) | window_size <= 0 | round(window_size) != window_size){
    stop("window_size must be a positive integer")
  }
  
  if( !is.numeric(alpha) | alpha <= 0){
    stop("alpha must be a positive numeric")
  }
  
  if( !is.numeric(threshold) | threshold <= 0){
    stop("threshold must be a positive numeric")
  }
  
  time_horizon <- nrow(rewards)  #time horizon
  K <- ncol(rewards)  #number of arms
  p <- ncol(features)  #number of features
  
  rewards <- log(rewards) #apply link function
  
  most_recent_change <- 0
  tau <- rep(0, K)  #most recent changepoint for each arm
  n_plays <- rep(0, K)
  actions <- rep(NA, time_horizon)
  rewards_received <- rep(0, time_horizon)
  change_locations <- anomalous_actions <- c()
  cumulative_reward <- 0
  
  #initial fill of arrays for A, windowed data list, and theta
  
  windowed_data_list <- vector("list", K)  #for each arm store the windowed data
  A_pre_array <- A_cur_array <- A_cum_array <- array(NA, dim = c(p, p, K))
  b_pre_array <- b_cur_array <- b_cum_array <- array(0, dim = c(p, 1, K))
  
  theta_pre <- theta_cum <- matrix(NA, nrow = p, ncol = K) #model parameters
  
  for(k in 1:K){
    A_pre_array[,,k] <- A_cur_array[,,k] <- A_cum_array[,,k] <- diag(1, p, p)
  }
  
  for(i in 1:time_horizon){
    
    #first update the cumulative theta parameters and calculate p_vec
    p_vec <- rep(NA, K)
    
    for(k in 1:K){
      theta_cum[,k] <- solve(A_cum_array[,,k]) %*% b_cum_array[,,k]
      p_vec[k] <- features[i,] %*% theta_cum[,k] + alpha * 
        sqrt(features[i,] %*% solve(A_cum_array[,,k]) %*% as.matrix(features[i,]))
    }
    #select arm with max reward
    action_to_take <- which.max(p_vec)
    #expected_reward <- p_vec[action_to_take]
    actions[i] <- action_to_take  #update based on choice
    n_plays[action_to_take] <- n_plays[action_to_take] + 1 #update times arm played
    
    rewards_received[i] <- rewards[i,action_to_take] #update rewards
    cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
    
    #update windowed_data_list (or initialise if the window is empty)
    windowed_data_list[[action_to_take]] <- rbind(windowed_data_list[[action_to_take]],
                                                  c(features[i,], rewards_received[i]))
    current_window_length <- nrow(windowed_data_list[[action_to_take]])
    
    #update arrays using iterative ridge regression:
    A_cur_array[,,action_to_take] <- A_cur_array[,,action_to_take] +
      as.matrix(features[i,]) %*% features[i,]
    A_cum_array[,,action_to_take] <- A_cum_array[,,action_to_take] +
      as.matrix(features[i,]) %*% features[i,]
    b_cur_array[,,action_to_take] <- b_cur_array[,,action_to_take] + 
      as.matrix(features[i,] * rewards_received[i])
    b_cum_array[,,action_to_take] <- b_cum_array[,,action_to_take] + 
      as.matrix(features[i,] * rewards_received[i])
    
    if(current_window_length >= window_size){ #if window has enough data, do change detection
      
      #update pre-change parameter estimates:
      theta_pre[,action_to_take] <- solve(A_pre_array[,,action_to_take]) %*%
        b_pre_array[,,action_to_take]
      windowed_features <- windowed_data_list[[action_to_take]][,1:p]
      windowed_rewards <- windowed_data_list[[action_to_take]][,(p+1)]
      
      #compute change test statistic
      
      estimated_rewards <- windowed_features %*% as.matrix(theta_pre[,action_to_take])
      residuals <- estimated_rewards - windowed_rewards
      test_stat <- mean(residuals) %>% abs()
      
      if(test_stat > threshold){ #reset estimates, and start cumulative learning with data we have
        
        change_locations <- c(change_locations, i)
        
        A_pre_array[,,action_to_take] <- A_cum_array[,,action_to_take] <- A_cur_array[,,action_to_take]
        b_pre_array[,,action_to_take] <- b_cum_array[,,action_to_take] <- b_cur_array[,,action_to_take]
        A_cur_array[,,action_to_take] <- diag(1, nrow = p, ncol = p)
        b_cur_array[,,action_to_take] <- matrix(0, nrow = p, ncol = 1)
        windowed_data_list[[action_to_take]] <- matrix(c(features[i,], rewards_received[i]),
                                                       nrow = 1)
      }else{
        #sliding window continues, update estimates
        A_pre_array[,,action_to_take] <- A_pre_array[,,action_to_take] + 
          as.matrix(windowed_features[1,]) %*% windowed_features[1,]
        A_cur_array[,,action_to_take] <- A_cur_array[,,action_to_take] - 
          as.matrix(windowed_features[1,]) %*% windowed_features[1,]
        b_pre_array[,,action_to_take] <- b_pre_array[,,action_to_take] + 
          as.matrix(windowed_features[1,] * windowed_rewards[1])
        b_cur_array[,,action_to_take] <- b_cur_array[,,action_to_take] -
          as.matrix(windowed_features[1,] * windowed_rewards[1])
        
        windowed_data_list[[action_to_take]] <- windowed_data_list[[action_to_take]][-1,]
        
      }
    }
    
  }
  
  #undo link function
  #cumulative_reward <- exp(cumulative_reward)
  #rewards_received <- exp(rewards_received)
  
  return_list <- list(cumulative_reward = cumulative_reward, change_locations = change_locations,
                      rewards_received = rewards_received, actions_taken = actions)
  return(return_list)
  
}
