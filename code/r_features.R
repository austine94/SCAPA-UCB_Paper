r_features <- function(n, n_cont, n_binary, n_discrete, delta_feature, intercept = TRUE){
  
  #function to generate features
  #n is the number to generate
  #n_cont is the number of continuous features, etc
  #delta_features is the range of values the features can take 
  
  if(!is.numeric(n) | n < 1 | round(n) != n){
    stop("n must be a positive integer")
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
  
  if(!is.logical(intercept)){
    stop("intercept must be either true or false")
  }
  
  total_features <- n_cont + n_discrete + n_binary  #do not count intercept here
  #we want to have a feature to generate as well as the intercept
  
  if(total_features <= 0){
    stop("must have a positive number of features to generate")
  }
  
  #create feature matrices for each type
  
  if(n_cont > 0){
    continous_mat <- runif( (n_cont * n), 0, delta_feature) %>% matrix(nrow = n, ncol = n_cont)
  }else{
    continous_mat <- matrix(NA, nrow = n, ncol = 1)
  }
  
  if(n_discrete > 0){
    discrete_mat <- sample(0:delta_feature, (n_discrete * n), replace = TRUE) %>%
                    matrix(nrow = n, ncol = n_discrete)
  }else{
    discrete_mat <- matrix(NA, nrow = n, ncol = 1)
  }
  
  if(n_binary > 0){
    binary_mat <- sample(0:1, (n_binary * n), replace = TRUE) %>%
      matrix(nrow = n, ncol = n_binary)
  }else{
    binary_mat <- matrix(NA, nrow = n, ncol = 1)
  }
  
  feature_mat <- cbind(continous_mat, discrete_mat, binary_mat) #combine feature sets
  
  feature_mat <- feature_mat[,colSums(!is.na(feature_mat)) > 0]  #remove all NA cols
  
  if(is.vector(feature_mat)){  #if the removal of NA leaves a vector, change back
    feature_mat <- as.matrix(feature_mat, nrow = n, ncol = 1)
  }
  
  if(intercept){  #if data contains intercept term
    intercept_col <- rep(1, n)
    feature_mat <- cbind(intercept_col, feature_mat)
  }
  
  return(feature_mat)
  
}
