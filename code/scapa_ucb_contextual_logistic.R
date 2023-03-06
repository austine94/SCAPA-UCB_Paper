scapa_ucb_contextual_logistic <- function(features, rewards, model_mat, lambda, alpha, 
                                        gamma, n_retrain){
  
  #this is the UCB algorithm with a changepoint test on the rewards of an arm using SCAPA
  #when the data follows a multinomial logistic regression
  #we restart the UCB if we get a collective anomaly
  #we remove point anomalies if they are observed
  
  #features is the T x n_features matrix of features over the T time steps
  #rewards is a T x K matrix of rewards for the K arms over the T time steps
  #lambda is for the SCAPA threshold
  #gamma is a value between 0 and 1 for the proportion of uniform samples to take
  #n_retrain is the number of steps to take when retraining
  
  require(tidyverse)
  require(anomaly)
  
  if(!is.numeric(features) | !is.matrix(features)){
    stop("features should be a matrix of numeric features")
  }
  
  if(!is.numeric(rewards) | !is.matrix(rewards)){
    stop("rewards should be a matrix of numeric rewards")
  }
  
  if(!is.numeric(model_mat) | !is.matrix(model_mat)){
    stop("model mat should be a p x K matrix of model coeffs")
  }
  
  if( !is.numeric(alpha) | alpha < 0){
    stop("alpha must be a non-negative numeric")
  }
  
  if( !is.numeric(lambda) | lambda <= 0){
    stop("lambda must be a positive numeric")
  }
  
  if( !is.numeric(n_retrain) | n_retrain <= 0 | n_retrain != round(n_retrain)){
    stop("n_retrain must be a positive integer")
  }
  
  beta <- 2 * (1 + lambda + sqrt(2*lambda))
  beta_tilde <- 2*lambda
  
  if( !is.numeric(gamma) | gamma <= 0 | gamma > 1){
    stop("gamma must be a numeric between 0 and 1")
  }
  
  time_horizon <- nrow(rewards)  #time horizon
  K <- ncol(rewards)  #number of arms
  p <- ncol(features)  #number of features
  
  ##NB: if we include the intercept we will have p+1 features; the model_mat also needs a
  ##row for the intercept
  
  if(ncol(model_mat) != K){
    stop("must have a model for each arm")
  }
  
  if(nrow(model_mat) != p){
    stop("must have models with p features")
  }
  
  most_recent_change <- 0
  n_plays <- rep(0, K)
  actions <- rep(NA, time_horizon)
  rewards_received <- rep(0, time_horizon)
  change_locations <- anomalous_actions <- c()
  cumulative_reward <- 0
  
  retrain <- FALSE  #for retraining particular model
  retrain_actions <- c()
  
  residual_vec <- rep(NA, time_horizon) #to store residuals
  #as we are using a logistic regression model we always store the difference between 1 and
  #the probability of the outcome that is observed.
  
  arm_selection_list <- vector("list", length = K) #for storing when arms are chosen
  for(i in 1:K){
    arm_selection_list[[i]] <- NA  #empty values
  }
  
  for(i in 1:time_horizon){
    if(retrain){
      #if enough points observed, retrain model for this arm. Else, continue to observe arm
      if(retrain_so_far >= n_retrain){
        model_mat <- initial_model_multinomial_logistic(retrain_features, 
                                                        retrain_labels, K) #retrain model
        retrain <- FALSE
        #once retrained we can backcalculate the residuals
        retrained_expected <- retrain_features %*% model_mat
        retrained_probs <- exp(retrained_expected)
        for(j in 1:n_retrain){
          #first normalized the probs and then compute residuals
          retrained_probs[j,] <- retrained_probs[j,] / sum(retrained_probs[j,]) 
          residual_vec[i - n_retrain - 1 + j] <- 1 - retrained_probs[j, retrain_labels[j]]
        }
        #once retrained we can continue to play
        expected_reward <- features[i,] %*% model_mat
        unnormalized_probs <- exp(expected_reward)
        expected_probs <- unnormalized_probs / sum(unnormalized_probs)
        ucb_k <- rep(0, K)
        for(k in 1:K){
          penalty <- alpha * sqrt((i - most_recent_change) / (n_plays[k] + 1))
          ucb_k[k] <- expected_probs[k] + penalty
          #use rewards since last changepoint in a specific arm for the penalty
        }
        action_to_take <- which.max(ucb_k)  #choose arm to max expected rewards
        actions[i] <- action_to_take  #update based on choice
        n_plays[action_to_take] <- n_plays[action_to_take] + 1
        arm_selection_list[[action_to_take]] <- c(arm_selection_list[[action_to_take]], i)
        rewards_received[i] <- rewards[i,action_to_take] #update rewards
        cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
        true_outcome <- which.max(rewards[i,]) #store true outcome label
        #compute residuals
        residual_reward <- 1 - expected_probs
        residual_vec[i] <- residuals[true_outcome] #residuals used by SCAPA       
  
      }else{
        #if not enough to retrain, continue to play using previous arm set
        expected_reward <- features[i,] %*% model_mat
        unnormalized_probs <- exp(expected_reward)
        expected_probs <- unnormalized_probs / sum(unnormalized_probs)
        ucb_k <- rep(0, K) 
        for(k in 1:K){
          penalty <- alpha * sqrt((i - most_recent_change) / (n_plays[k] + 1))
          ucb_k[k] <- expected_probs[k] + penalty
          #use rewards since last changepoint in a specific arm for the penalty
        }
        action_to_take <- which.max(ucb_k)  #choose arm to max expected rewards
        #store features, rewards, and retraining info
        retrain_features <- rbind(retrain_features, features[i,])
        rewards_received[i] <- rewards[i,action_to_take] #update rewards
        cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
        retrain_actions <- c(retrain_actions, i)
        retrain_so_far <- retrain_so_far + 1
        retrain_labels <- c(retrain_labels, which.max(rewards[i,])) #store true outcome
        
        actions[i] <- action_to_take  #store action taken
        n_plays[action_to_take] <- n_plays[action_to_take] + 1
        arm_selection_list[[action_to_take]] <- c(arm_selection_list[[action_to_take]], i)
        #we do not store residuals as we back calculate these once the new model is trained
      }
      #if not retraining then perform UCB round of algo
    }else{
      #we either explore or use UCB depending on A:
      A <- (i - most_recent_change) %% (floor(K / gamma))   #uniform sampling; gamma is the proportion that are exploration only
      if( A > 0 & A <= K){  #exploration step
        action_to_take <- A
        actions[i] <- action_to_take  #store action taken
        n_plays[action_to_take] <- n_plays[action_to_take] + 1  #update arm played
        arm_selection_list[[action_to_take]] <- c(arm_selection_list[[action_to_take]], i)
        
        rewards_received[i] <- rewards[i,action_to_take] #update rewards
        cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
        true_outcome <- which.max(rewards[i,]) #correct label
        
        #compute residuals
        expected_reward <- features[i,] %*% model_mat
        unnormalized_probs <- exp(expected_reward)
        expected_probs <- unnormalized_probs / sum(unnormalized_probs)
        residuals <- 1 - expected_probs
        residual_vec[i] <- residuals[true_outcome] #residuals used by SCAPA
    
      }else{  #use UCB to choose max
        expected_reward <- features[i,] %*% model_mat
        unnormalized_probs <- exp(expected_reward)
        expected_probs <- unnormalized_probs / sum(unnormalized_probs)
        ucb_k <- rep(0, K) 
        for(k in 1:K){
          penalty <- alpha * sqrt((i - most_recent_change) / (n_plays[k] + 1))
          ucb_k[k] <- expected_probs[k] + penalty
          #use rewards since last changepoint in a specific arm for the penalty
        }
        action_to_take <- which.max(ucb_k)  #choose arm to max expected rewards
        actions[i] <- action_to_take  #update based on choice
        n_plays[action_to_take] <- n_plays[action_to_take] + 1
        arm_selection_list[[action_to_take]] <- c(arm_selection_list[[action_to_take]], i)
        rewards_received[i] <- rewards[i,action_to_take] #update rewards
        cumulative_reward <- cumulative_reward + rewards[i,action_to_take]
        true_outcome <- which.max(rewards[i,]) #store true outcome label
        #compute residuals
        residual_reward <- 1 - expected_probs
        residual_vec[i] <- residuals[true_outcome] #residuals used by SCAPA
      }
      
      if((i-most_recent_change) >= 100){  #use change detection
        #we record the na's to get the right time, and then omit for use with scapa
        sum_na <- is.na(residual_vec[(most_recent_change + 1):i]) %>% sum() 
        residuals_to_test <- na.omit(residual_vec[(most_recent_change + 1):i])
        change_test <- ucb_scapa_contextual_test(residuals_to_test, beta, beta_tilde,
                                                 transform = function(x){ return(x)})  
        
        if(change_test$point){ 
          #if a point anomaly then either remove this from future UCB rounds
          #or just record as anomaly 
          residual_vec[i] <- NA
          anomalous_actions <- c(anomalous_actions, i)   
        }
        #note we only return point if NO collective anomalies are present
        #as otherwise we act on the collective anomaly and retrain
        if(change_test$collective){  #reset number of plays, and mark most recent change
          #record changepoint time
          anomaly_time <- most_recent_change + change_test$anomaly_time + sum_na- 1
          #the minus one is because we take points including the change
          most_recent_change <- anomaly_time
          change_locations <- c(change_locations, most_recent_change)
          
          n_plays <- rep(0,K)
          #as logistic regression we retrain all arms as we retrain the logistic reg model
          
          retrain <- TRUE  #store the points since the change 
          retrain_so_far <- i - most_recent_change
          retrain_features <- features[(most_recent_change):i,]
          retrain_labels <- apply(rewards[(most_recent_change):i,], 1, which.max)
        }
      }
    }
    if(i %% 500 == 0){
      print(i)
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

