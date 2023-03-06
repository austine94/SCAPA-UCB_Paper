#works out training cost for model parameters for SCAPA-UCB
#assumes each of the K arms is trained using n points

training_cost <- function(rewards, train_steps){
  
  if(!is.numeric(rewards) | !is.matrix(rewards)){
    stop("rewards should be a matrix of numeric rewards")
  }
  
  if(!is.numeric(train_steps) | train_steps < 1 | train_steps != round(train_steps)){
    stop("train_steps must be a positive integer")
  }
  
  time_horizon <- nrow(rewards)  #time horizon
  K <- ncol(rewards)  #number of arms
  
  if( (train_steps * K) > time_horizon){
    stop("need at least train_steps * K points to train model")
  }
  
  oracle_cost <- apply(rewards[1:(train_steps *K),], 1, max) %>% sum()  #best rewards
  #only work out for the K*train_steps we use
  
  train_cost <- 0
  
  for(k in 1:K){
    arm_rewards <- rewards[((k-1)*train_steps) : (k*train_steps), k]
    train_cost <- train_cost + sum(arm_rewards)  #value of choosing these arms whilst training
  }
  
  train_regret = oracle_cost - train_cost
  
  return(list(train_cost = train_cost, train_regret = train_regret))
  
}