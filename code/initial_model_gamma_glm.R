#initial gamma glm bandit model estimator

initial_model_gamma_glm <- function(features, rewards, intercept = TRUE){
  
  #features and rewards are the feature and reward matrices to estimate the model
  #intercept is a logical indicating if an intercept should be fitted to the model
  #it is assumed that the first column of features is a col of 1s if intercept is true
  
  if(!is.numeric(features) | !is.matrix(features)){
    stop("features should be a matrix of numeric features")
  }
  
  if(!is.numeric(rewards) | !is.matrix(rewards)){
    stop("rewards should be a matrix of numeric rewards")
  }
  
  if(!is.logical(intercept)){
    stop("intercept must be either true or false")
  }
  
  if(intercept){
    features <- features[,-1]  #remove the intercept column
  }
  
  if(is.vector(features)){  #if only one feature, keep it as a matrix
    features <- matrix(features, ncol = 1)
  }
  
  rewards <- log(rewards)  #apply link function
  
  time_horizon <- nrow(rewards)  #time horizon
  K <- ncol(rewards)  #number of arms
  p <- ncol(features)  #number of features
  
  #fit regression models
  
  
  if(intercept){
    model_mat <- matrix(NA, nrow = (p+1), ncol = K)
    
    for(k in 1:K){
      regression_mat <- cbind(rewards[,k], features)
      regression_fit <- lm(data = as.data.frame(regression_mat)) #regress on first (reward) col
      
      model_mat[,k] <- regression_fit$coefficients
      
    }
    
  }else{
    model_mat <- matrix(NA, nrow = p, ncol = K)
    
    for(k in 1:K){
      regression_mat <- cbind(rewards[,k], features)
      regression_fit <- lm(regression_mat[,1] ~. - 1,
                           data = as.data.frame(regression_mat)) #regress on first (reward) col
      model_mat[,k] <- regression_fit$coefficients
      
    }
    
  }
  
  return(model_mat)
  
}