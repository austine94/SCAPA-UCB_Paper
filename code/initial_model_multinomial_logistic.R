#initial multinomial logistic regression model estimator

initial_model_multinomial_logistic <- function(features, outcomes, K){
  
  #features is the feature matrix to estimate the model
  #outcomes is the vector of labels identifying which of the K outcomes happened at a time step
  
  ######
  #note#
  ######
  #we must fit an intercept and so we assume that the first row of features is NOT
  #an intercept, unlike with the rest of the package.
  #to address this we automatically drop the intercept column from features 
  
  require(tidyverse)
  require(nnet)
  
  if(!is.numeric(features) | !is.matrix(features)){
    stop("features should be a matrix of numeric features")
  }
  
  features <- features[,-1]
  
  if(!is.numeric(outcomes)){
    stop("outcomes should be a vector of outcome labels")
  }
  
  if(nrow(features) != length(outcomes)){
    stop("outcome length must equal nrow of features")
  }
  
  if(max(outcomes) > K){
    stop("outcomes must be a vector of positive integers and max cannot exceed K")
  }
  
  if(any(outcomes != round(outcomes)) | any(outcomes <= 0)){
    stop("outcomes must be a vector of positive integers and max cannot exceed K")
  }
  
  time_horizon <- nrow(features)  #time horizon
  p <- ncol(features)  #number of features
  
  #create data frame for model fitting
  full_data_frame <- cbind(outcomes, features) %>% as.data.frame()
  
  multinomial_fit <- multinom(outcomes ~ ., data = full_data_frame) #fit model
  
  model_coefficients <- coefficients(multinomial_fit)
  
  #we need a fail safe to ensure all K possible options have been fitted, or 
  #to give us zeros for a missing set of coeffs:
  coeff_rownames <- rownames(model_coefficients)
  
  model_mat <- matrix(0, nrow = p + 1, ncol = K) #p+1 as we store the intercept
  for(i in 1:K){
    i_char <- as.character(i)
    if(i_char %in% coeff_rownames){ #if we have fitted coeffs for this outcome
      index <- which(coeff_rownames == i_char) #store these coeffs
      model_mat[,i] <- model_coefficients[index,]
    }#else we keep a column of zeros for the coefficients
    
  }
  
  return(model_mat)
  
  
}
