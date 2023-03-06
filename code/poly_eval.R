#function to evaluate polynomial with given coefficients
#similar to poly function, but does not require coeff structure specification

poly_eval <- function(coeffs, new_point){
  
  if(!is.numeric(coeffs)){
    stop("coeffs must be a vector of coefficients specifying the polynomial to evaluate")
  }
  
  if(!is.numeric(new_point)){
    stop("new_point must be a numeric at which the specified polynomial is to be evaluatedm")
  }
  
  order <- length(coeffs) - 1  
  power_vec <- sapply(0:order, function(x) new_point^x)
  poly_val <- sum(power_vec * coeffs)
  
  
  return(poly_val)
  
}
