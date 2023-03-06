#initial polynomial regression bandit model estimator

initial_model_polynomial <- function(input, rewards, order){
  
  #input and rewards are the input and reward matrices to estimate the model
  #order is the polynomial order
  
  if(!is.numeric(input)){
    stop("input should be a vector of numerics for the polynomial domain")
  }
  
  if(!is.numeric(rewards) | !is.matrix(rewards)){
    stop("rewards should be a matrix of numeric rewards")
  }
  
  if( !is.numeric(order) | order < 0 | round(order) != order){
    stop("order must be a non-negative integer")
  }
  
  time_horizon <- nrow(rewards)  #time horizon
  K <- ncol(rewards)  #number of arms
  #fit regression models
  
  model_mat <- matrix(NA, nrow = (order+1), ncol = K)
    
  for(k in 1:K){
    regression_fit <- lm(rewards[,k] ~ poly(input, order, raw = TRUE) ) #regress on first (reward) col
    
    model_mat[,k] <- regression_fit$coefficients
    
  }
    
  return(model_mat)
  
}