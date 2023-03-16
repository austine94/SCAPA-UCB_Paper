library(tidyverse)
library(anomaly)
library(kableExtra)

#########################
#import functions for sims
##########################

source("./code/contextual_gamma_glm_generator.R")
source("./code/contextual_gaussian_linear_generator.R")
source("./code/contextual_heavy_linear_generator.R")
source("./code/contextual_gamma_glm_generator.R")
source("./code/contextual_logistic_glm_generator.R")
source("./code/contextual_poisson_glm_generator.R")
source("./code/contextual_polynomial_generator.R")
source("./code/contextual_sinusoidal_generator.R")
source("./code/contextual_zib_linear_generator.R")
source("./code/initial_model_gamma_glm.R")
source("./code/initial_model_linear.R")
source("./code/initial_model_multinomial_logistic.R")
source("./code/initial_model_poisson_glm.R")
source("./code/initial_model_polynomial.R")
source("./code/poly_eval.R")
source("./code/PSLinUCB.R")
source("./code/PSLinUCB_gamma_glm.R")
source("./code/PSLinUCB_poisson_glm.R")
source("./code/r_features.R")
source("./code/SCAPA_Contextual.R")
source("./code/scapa_ucb_contextual_gamma_glm.R")
source("./code/scapa_ucb_contextual_linear.R")
source("./code/scapa_ucb_contextual_logistic.R")
source("./code/scapa_ucb_contextual_poisson_glm.R")
source("./code/scapa_ucb_contextual_polynomial.R")
source("./code/training_cost.R")


####################
#S4.1 Sims
####################

time_horizon <- 1000
K <- 10         #number of arms
m <- 20     #number of changes
train_steps <- 30  #number of obs to fit each arm model for scapa ucb
training_size <- K*train_steps  #number of training points to generate

alpha <- 0.001
lambda <- 3*log(1000)

n_reps <- 100

set.seed(100)

####################

#Gaussian Noise - Scenario One

###################

pslinucb_regret_mat <- scapa_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)

regret_prop_scapa <- regret_prop_lin <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_gaussian_linear_generator(time_horizon, K, m, 15, 0, 5, overlap = FALSE,
                                               training = training_size, delta_feature = 1,
                                               delta_coeff = 3)
  model_mat <- initial_model_linear(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  #lin_cumsum <- cumsum(lin_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_one <- mean(regret_prop_scapa, na.rm = TRUE)
pslin_mean_regret_one <- mean(regret_prop_lin, na.rm = TRUE)

###############
#Heavy Tailed Noise - Two
###############

set.seed(200)

pslinucb_regret_mat <- scapa_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_lin <- rep(NA, n_reps)


for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_heavy_linear_generator(time_horizon, K, m, 15, 0, 5, overlap = FALSE,
                                            training = training_size, delta_feature = 1,
                                            delta_coeff = 3)
  model_mat <- initial_model_linear(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean)

scapa_mean_regret_two <- mean(regret_prop_scapa)
pslin_mean_regret_two <- mean(regret_prop_lin)

####################

#poisson GLM - Three

###################

set.seed(300)

pslinucb_regret_mat <- scapa_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)

regret_prop_scapa <- regret_prop_lin <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_poisson_glm_generator(time_horizon, K, m, 1, 0, 0, overlap = FALSE,
                                           training = training_size, delta_coeff = 1,
                                           delta_feature = 3)
  model_mat <- initial_model_poisson_glm(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps) 
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_poisson_glm(data$feature_mat, data$reward_mat, model_mat,
                                                lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb_poisson(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                              threshold = 1)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% log %>% cumsum 
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  #lin_cumsum <- cumsum(lin_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_three <- mean(regret_prop_scapa, na.rm = TRUE)
pslin_mean_regret_three <- mean(regret_prop_lin, na.rm = TRUE)


####################

#Gamma GLM - Four

###################

set.seed(400)

pslinucb_regret_mat <- scapa_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)

regret_prop_scapa <- regret_prop_lin <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_gamma_glm_generator(time_horizon, K, m, 1, 0, 0, shape = 20, overlap = FALSE,
                                         training = training_size, delta_coeff = 1,
                                         delta_feature = 3)
  model_mat <- initial_model_gamma_glm(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps) 
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_gamma_glm(data$feature_mat, data$reward_mat, model_mat,
                                              lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb_gamma(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                            threshold = 1)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% log %>% cumsum 
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  #lin_cumsum <- cumsum(lin_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_four <- mean(regret_prop_scapa)
pslin_mean_regret_four <- mean(regret_prop_lin)

####################

#Polynomial - Five

###################

set.seed(500)

pslinucb_regret_mat <- scapa_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)

regret_prop_scapa <- regret_prop_lin <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_polynomial_generator(time_horizon, K, m, 0, 2, 4, overlap = FALSE,
                                          training = training_size)
  model_mat <- initial_model_polynomial(data$training_input, data$training_rewards, 4)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- c(data$training_input, data$input_mat) %>% as.matrix(ncol = 1)
  ps_data_features <- cbind(rep(1, nrow(ps_data_features)), ps_data_features)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_polynomial(data$input_mat, data$reward_mat, model_mat,
                                               lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  #lin_cumsum <- cumsum(lin_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_five <- mean(regret_prop_scapa)
pslin_mean_regret_five <- mean(regret_prop_lin)

##############
#Sinusoidal - Six
################

set.seed(600)

pslinucb_regret_mat <- scapa_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)

regret_prop_scapa <- regret_prop_lin <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_sinusoidal_generator(time_horizon, K, m, 0, 2, overlap = FALSE, 
                                          frequency = 1, training = training_size)
  model_mat <- initial_model_polynomial(data$training_input, data$training_rewards, 7)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- c(data$training_input, data$input_mat) %>% as.matrix(ncol = 1)
  ps_data_features <- cbind(rep(1, nrow(ps_data_features)), ps_data_features)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_polynomial(data$input_mat, data$reward_mat, model_mat,
                                               lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  #lin_cumsum <- cumsum(lin_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_six <- mean(regret_prop_scapa, na.rm = TRUE)
pslin_mean_regret_six <- mean(regret_prop_lin, na.rm = TRUE)

####################

#ZIB - Seven

###################

set.seed(700)

pslinucb_regret_mat <- scapa_regret_mat <- matrix(NA, nrow = time_horizon,
                                                  ncol = n_reps)
regret_prop_scapa <- regret_prop_lin <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_zib_linear_generator(time_horizon, K, m, 15, 0, 5, overlap = FALSE,
                                          training = training_size, burst_prob = 0.01, 
                                          delta_feature = 1,
                                          delta_coeff = 3)
  model_mat <- initial_model_linear(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  #lin_cumsum <- cumsum(lin_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_seven <- mean(regret_prop_scapa)
pslin_mean_regret_seven <- mean(regret_prop_lin)

############
#4.1 Table
############
scapa_ucb_results <- c(scapa_mean_regret_one, 
                   scapa_mean_regret_two, 
                   scapa_mean_regret_three, 
                   scapa_mean_regret_four, 
                   scapa_mean_regret_five, 
                   scapa_mean_regret_six, 
                   scapa_mean_regret_seven)
pslin_results <- c(pslin_mean_regret_one, 
                   pslin_mean_regret_two, 
                   pslin_mean_regret_three, 
                   pslin_mean_regret_four, 
                   pslin_mean_regret_five, 
                   pslin_mean_regret_six, 
                   pslin_mean_regret_seven)
scapa_ucb_results_table <- round(scapa_ucb_results * 100, 0)
pslin_results_table <- round(pslin_results * 100, 0)

four_point_one_df <- data.frame(SCAPA_UCB = scapa_ucb_results_table,
                                PSLinUCB = pslin_results_table)
four_point_one_table <- kable(four_point_one_df, "html", 
                              table.attr = "style='width:30%;'") %>%
                              kable_styling(font_size = 20, bootstrap_options = "striped")
save_kable(four_point_one_table, file = "./results/S4.1_Results.png")

#####################
#S4.2 - Varying Parameters
#####################

###################
#m
###################

time_horizon <- 1000
K <- 10       #number of arms
m <- seq(1, 30, by = 2)            #number of changes
train_steps <- 30  #number of obs to fit each arm model for scapa ucb
training_size <- K*train_steps  #number of training points to generate

alpha <- 0.001
lambda <- 3*log(1000)

n_reps <- 100

set.seed(200)

pslinucb_regret_m <- scapa_ucb_regret_m <- rep(NA, length(m))

for(k in 1:length(m)){
  
  pslinucb_regret_current <- scapa_regret_current <- 0
  
  for(i in 1:n_reps){
    data <- contextual_gaussian_linear_generator(time_horizon, K, m[k], 15, 0, 5, overlap = FALSE,
                                                 training = training_size, delta_feature = 1,
                                                 delta_coeff = 3)
    model_mat <- initial_model_linear(data$training_features, data$training_rewards)
    scapa_train_reward <- training_cost(data$training_rewards, train_steps)
    
    #create data features and rewards that include training data to put all of it into pslinucb
    
    ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
    ps_data_features <- rbind(data$training_features, data$feature_mat)
    
    #perform bandit algos
    
    
    scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                             lambda, alpha = alpha, 0.01, 30)
    
    lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                        1)
    
    lin_sum <- sum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
    
    #add to total regret
    
    oracle_reward <- apply(data$reward_mat, 1, max) %>% sum()
    
    scapa_regret_current <- scapa_regret_current + (oracle_reward - scapa_run$cumulative_reward)
    pslinucb_regret_current <- pslinucb_regret_current +
      (oracle_reward - lin_sum)
    
    print(c(k, i))
  }
  #calculate expected regret
  scapa_ucb_regret_m[k] <- scapa_regret_current / n_reps
  pslinucb_regret_m[k] <- pslinucb_regret_current / n_reps
}

#plot result

g_m <- ggplot() + geom_line(aes(x = m, y = scapa_ucb_regret_m), col = "blue") + 
  geom_line(aes(x = m, y = pslinucb_regret_m), col = "black", lty = 2) +
  labs(x = "m", y = "Expedcted Total Regret") + theme_idris()
ggsave("results/m_increase_paper.png", g_m)

##################
#K
##################

time_horizon <- 1000
K <- seq(2, 30, by = 2)          #number of arms
m <- 20           #number of changes
alpha <- 0.001
train_steps <- 30  #number of obs to fit each arm model for scapa ucb

n_reps <- 100

set.seed(200)

pslinucb_regret_K <- scapa_ucb_regret_K <- rep(NA, length(K))

for(k in 1:length(K)){
  
  training_size <- K[k]*train_steps  #number of training points to generate
  
  pslinucb_regret_current <- scapa_regret_current <- 0
  
  for(i in 1:n_reps){
    data <- contextual_gaussian_linear_generator(time_horizon, K[k], m, 15, 0, 5, overlap = FALSE,
                                                 training = training_size, delta_feature = 1,
                                                 delta_coeff = 3)
    model_mat <- initial_model_linear(data$training_features, data$training_rewards)
    scapa_train_reward <- training_cost(data$training_rewards, train_steps)
    
    #create data features and rewards that include training data to put all of it into pslinucb
    
    ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
    ps_data_features <- rbind(data$training_features, data$feature_mat)
    
    #perform bandit algos
    
    
    scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                             lambda, alpha, 0.01, 30)
    
    lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                        threshold = 1)
    
    lin_sum <- sum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
    
    #add to total regret
    
    oracle_reward <- apply(data$reward_mat, 1, max) %>% sum()
    
    scapa_regret_current <- scapa_regret_current + (oracle_reward - scapa_run$cumulative_reward)
    pslinucb_regret_current <- pslinucb_regret_current +
      (oracle_reward - lin_sum)
    
    print(c(k, i))
  }
  #calculate expected regret
  scapa_ucb_regret_K[k] <- scapa_regret_current / n_reps
  pslinucb_regret_K[k] <- pslinucb_regret_current / n_reps
}


#plot

g_k <- ggplot() + geom_line(aes(x = K, y = pslinucb_regret_K), col = "black", lty = 2) +
  geom_line(aes(x = K, y = scapa_ucb_regret_K), col = "blue") +
  labs(x = "K", y = "Expected Total Regret") + theme_idris()
ggsave("results/K_Increase_paper.png", g_k)

################
#R
################


time_horizon <- 1000
K <- 10        #number of arms
m <- 10           #number of changes
alpha <- 0.001
train_steps <- 30  #number of obs to fit each arm model for scapa ucb
training_size <- K*train_steps  #number of training points to generate
R_vec <- seq(10, 100, by = 5)

n_reps <- 100

set.seed(200)

scapa_ucb_regret <- rep(0, length(R_vec))

for(i in 1:n_reps){
  
  data <- contextual_gaussian_linear_generator(time_horizon, K, m, 15, 0, 5, overlap = TRUE,
                                               training = training_size, delta_feature = 1,
                                               delta_coeff = 3)
  model_mat <- initial_model_linear(data$training_features, data$training_rewards)
  
  for(k in 1:length(R_vec)){
    
    
    #perform bandit algo
    
    scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                             20, alpha, 0.01, R_vec[k])
    #add to total regret
    
    oracle_reward <- apply(data$reward_mat, 1, max) %>% sum()
    
    scapa_ucb_regret[k] <- scapa_ucb_regret[k] + (oracle_reward - scapa_run$cumulative_reward)
    
    print(c(i, k))
  }
  #calculate expected regret
}

scapa_ucb_regret_R <- scapa_ucb_regret / n_reps

#plot

g_r <- ggplot() + geom_line(aes(x = R_vec, y = scapa_ucb_regret_R), col = "blue") + 
  labs(x = "R", y = "Expected Total Regret") + theme_idris()
ggsave("results/R_Increase_paper.png", g_r)

