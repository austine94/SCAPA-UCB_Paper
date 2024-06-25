##################
#Ablation Study
###################

time_horizon <- 1000
K <- 10         #number of arms
m <- 20     #number of changes
train_steps <- 30  #number of obs to fit each arm model for scapa ucb
training_size <- K*train_steps  #number of training points to generate

alpha <- 0.001
lambda <- 3*log(1000)

n_reps <- 100
anomaly_prob <- 0.01

set.seed(100)

####################

#Gaussian Noise - Scenario One

###################

scapa_no_test_regret_mat <- scapa_regret_mat <- scapa_no_penalty_regret_mat <- scapa_no_gamma_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_scapa_no_test <- regret_prop_scapa_no_penalty <- regret_prop_scapa_no_gamma <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_gaussian_linear_generator(time_horizon, K, m, 15, 0, 5, overlap = FALSE,
                                               training = training_size, delta_feature = 1,
                                               delta_coeff = 3)
  model_mat <- initial_model_linear(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #add point anomalies
  point_anomaly_indices <- rbinom((time_horizon*K), 1, anomaly_prob)
  data$reward_mat[which(point_anomaly_indices == 1)] <- 0  
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  scapa_no_test_run <-scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                                  1e6, alpha, 0.01, 30)
  
  scapa_no_penalty_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                                      lambda, 1e-10, 0.01, 30)
  
  scapa_no_gamma_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 1e-10, 30)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  scapa_no_test_cumsum <- cumsum(scapa_no_test_run$rewards_received)
  scapa_no_penalty_cumsum <- cumsum(scapa_no_penalty_run$rewards_received)
  scapa_no_gamma_cumsum <- cumsum(scapa_no_gamma_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for scapa_no_test to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  scapa_no_test_regret <- oracle_cumsum - scapa_no_test_cumsum
  scapa_no_penalty_regret <- oracle_cumsum - scapa_no_penalty_cumsum
  scapa_no_gamma_regret <- oracle_cumsum - scapa_no_gamma_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  scapa_no_test_regret_mat[,i] <- scapa_no_test_regret
  scapa_no_penalty_regret_mat[,i] <- scapa_no_penalty_regret
  scapa_no_gamma_regret_mat[,i] <- scapa_no_gamma_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_test[i] <- scapa_no_test_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_penalty[i] <- scapa_no_penalty_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_gamma[i] <- scapa_no_gamma_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_test_cum_regret <- apply(scapa_no_test_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_penalty_cum_regret <- apply(scapa_no_penalty_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_gamma_cum_regret <- apply(scapa_no_gamma_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_one <- mean(regret_prop_scapa, na.rm = TRUE)
scapa_no_test_mean_regret_one <- mean(regret_prop_scapa_no_test, na.rm = TRUE)
scapa_no_penalty_mean_regret_one <- mean(regret_prop_scapa_no_penalty, na.rm = TRUE)
scapa_no_gamma_mean_regret_one <- mean(regret_prop_scapa_no_gamma, na.rm = TRUE)

###############
#Heavy Tailed Noise - Two
###############

set.seed(200)

scapa_no_test_regret_mat <- scapa_regret_mat <- scapa_no_penalty_regret_mat <- scapa_no_gamma_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_scapa_no_test <- regret_prop_scapa_no_penalty <- regret_prop_scapa_no_gamma <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_heavy_linear_generator(time_horizon, K, m, 15, 0, 5, overlap = FALSE,
                                            training = training_size, delta_feature = 1,
                                            delta_coeff = 3)
  model_mat <- initial_model_linear(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #add point anomalies
  point_anomaly_indices <- rbinom((time_horizon*K), 1, anomaly_prob)
  data$reward_mat[which(point_anomaly_indices == 1)] <- 0  
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  scapa_no_test_run <-scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                                  1e6, alpha, 0.01, 30)
  
  scapa_no_penalty_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                                      lambda, 1e-10, 0.01, 30)
  
  scapa_no_gamma_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                                    lambda, alpha, 1e-10, 30)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  scapa_no_test_cumsum <- cumsum(scapa_no_test_run$rewards_received)
  scapa_no_penalty_cumsum <- cumsum(scapa_no_penalty_run$rewards_received)
  scapa_no_gamma_cumsum <- cumsum(scapa_no_gamma_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for scapa_no_test to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  scapa_no_test_regret <- oracle_cumsum - scapa_no_test_cumsum
  scapa_no_penalty_regret <- oracle_cumsum - scapa_no_penalty_cumsum
  scapa_no_gamma_regret <- oracle_cumsum - scapa_no_gamma_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  scapa_no_test_regret_mat[,i] <- scapa_no_test_regret
  scapa_no_penalty_regret_mat[,i] <- scapa_no_penalty_regret
  scapa_no_gamma_regret_mat[,i] <- scapa_no_gamma_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_test[i] <- scapa_no_test_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_penalty[i] <- scapa_no_penalty_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_gamma[i] <- scapa_no_gamma_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_test_cum_regret <- apply(scapa_no_test_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_penalty_cum_regret <- apply(scapa_no_penalty_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_gamma_cum_regret <- apply(scapa_no_gamma_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_two <- mean(regret_prop_scapa, na.rm = TRUE)
scapa_no_test_mean_regret_two <- mean(regret_prop_scapa_no_test, na.rm = TRUE)
scapa_no_penalty_mean_regret_two <- mean(regret_prop_scapa_no_penalty, na.rm = TRUE)
scapa_no_gamma_mean_regret_two <- mean(regret_prop_scapa_no_gamma, na.rm = TRUE)

####################

#poisson GLM - Three

###################

set.seed(300)

scapa_no_test_regret_mat <- scapa_regret_mat <- scapa_no_penalty_regret_mat <- scapa_no_gamma_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_scapa_no_test <- regret_prop_scapa_no_penalty <- regret_prop_scapa_no_gamma <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_poisson_glm_generator(time_horizon, K, m, 1, 0, 0, overlap = FALSE,
                                           training = training_size, delta_coeff = 1,
                                           delta_feature = 3)
  model_mat <- initial_model_poisson_glm(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps) 
  
  #add point anomalies
  point_anomaly_indices <- rbinom((time_horizon*K), 1, anomaly_prob)
  data$reward_mat[which(point_anomaly_indices == 1)] <- 0  
  
  #perform bandit algos
  
  scapa_run <- scapa_ucb_contextual_poisson_glm(data$feature_mat, data$reward_mat, model_mat,
                                                lambda, alpha, 0.01, 30)
  
  scapa_no_test_run <-scapa_ucb_contextual_poisson_glm(data$feature_mat, data$reward_mat, model_mat,
                                                  1e6, alpha, 0.01, 30)
  
  scapa_no_penalty_run <- scapa_ucb_poisson_glm(data$feature_mat, data$reward_mat, model_mat,
                                                      lambda, 1e-10, 0.01, 30)
  
  scapa_no_gamma_run <- scapa_ucb_poisson_glm(data$feature_mat, data$reward_mat, model_mat,
                                                    lambda, alpha, 1e-10, 30)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  scapa_no_test_cumsum <- cumsum(scapa_no_test_run$rewards_received)
  scapa_no_penalty_cumsum <- cumsum(scapa_no_penalty_run$rewards_received)
  scapa_no_gamma_cumsum <- cumsum(scapa_no_gamma_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for scapa_no_test to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  scapa_no_test_regret <- oracle_cumsum - scapa_no_test_cumsum
  scapa_no_penalty_regret <- oracle_cumsum - scapa_no_penalty_cumsum
  scapa_no_gamma_regret <- oracle_cumsum - scapa_no_gamma_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  scapa_no_test_regret_mat[,i] <- scapa_no_test_regret
  scapa_no_penalty_regret_mat[,i] <- scapa_no_penalty_regret
  scapa_no_gamma_regret_mat[,i] <- scapa_no_gamma_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_test[i] <- scapa_no_test_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_penalty[i] <- scapa_no_penalty_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_gamma[i] <- scapa_no_gamma_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_test_cum_regret <- apply(scapa_no_test_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_penalty_cum_regret <- apply(scapa_no_penalty_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_gamma_cum_regret <- apply(scapa_no_gamma_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_three <- mean(regret_prop_scapa, na.rm = TRUE)
scapa_no_test_mean_regret_three <- mean(regret_prop_scapa_no_test, na.rm = TRUE)
scapa_no_penalty_mean_regret_three <- mean(regret_prop_scapa_no_penalty, na.rm = TRUE)
scapa_no_gamma_mean_regret_three <- mean(regret_prop_scapa_no_gamma, na.rm = TRUE)

####################

#Gamma GLM - Four

###################

set.seed(400)

scapa_no_test_regret_mat <- scapa_regret_mat <- scapa_no_penalty_regret_mat <- scapa_no_gamma_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_scapa_no_test <- regret_prop_scapa_no_penalty <- regret_prop_scapa_no_gamma <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_gamma_glm_generator(time_horizon, K, m, 1, 0, 0, shape = 20, overlap = FALSE,
                                         training = training_size, delta_coeff = 1,
                                         delta_feature = 3)
  model_mat <- initial_model_gamma_glm(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps) 
  
  #add point anomalies
  point_anomaly_indices <- rbinom((time_horizon*K), 1, anomaly_prob)
  data$reward_mat[which(point_anomaly_indices == 1)] <- 0  
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_gamma_glm(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  scapa_no_test_run <-scapa_ucb_contextual_gamma_glm(data$feature_mat, data$reward_mat, model_mat,
                                                  1e6, alpha, 0.01, 30)
  
  scapa_no_penalty_run <- scapa_ucb_contextual_gamma_glm(data$feature_mat, data$reward_mat, model_mat,
                                                      lambda, 1e-10, 0.01, 30)
  
  scapa_no_gamma_run <- scapa_ucb_contextual_gamma_glm(data$feature_mat, data$reward_mat, model_mat,
                                                    lambda, alpha, 1e-10, 30)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  scapa_no_test_cumsum <- cumsum(scapa_no_test_run$rewards_received)
  scapa_no_penalty_cumsum <- cumsum(scapa_no_penalty_run$rewards_received)
  scapa_no_gamma_cumsum <- cumsum(scapa_no_gamma_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for scapa_no_test to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  scapa_no_test_regret <- oracle_cumsum - scapa_no_test_cumsum
  scapa_no_penalty_regret <- oracle_cumsum - scapa_no_penalty_cumsum
  scapa_no_gamma_regret <- oracle_cumsum - scapa_no_gamma_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  scapa_no_test_regret_mat[,i] <- scapa_no_test_regret
  scapa_no_penalty_regret_mat[,i] <- scapa_no_penalty_regret
  scapa_no_gamma_regret_mat[,i] <- scapa_no_gamma_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_test[i] <- scapa_no_test_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_penalty[i] <- scapa_no_penalty_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_gamma[i] <- scapa_no_gamma_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_test_cum_regret <- apply(scapa_no_test_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_penalty_cum_regret <- apply(scapa_no_penalty_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_gamma_cum_regret <- apply(scapa_no_gamma_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_four <- mean(regret_prop_scapa, na.rm = TRUE)
scapa_no_test_mean_regret_four <- mean(regret_prop_scapa_no_test, na.rm = TRUE)
scapa_no_penalty_mean_regret_four <- mean(regret_prop_scapa_no_penalty, na.rm = TRUE)
scapa_no_gamma_mean_regret_four <- mean(regret_prop_scapa_no_gamma, na.rm = TRUE)

####################

#Polynomial - Five

###################

set.seed(500)

scapa_no_test_regret_mat <- scapa_regret_mat <- scapa_no_penalty_regret_mat <- scapa_no_gamma_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_scapa_no_test <- regret_prop_scapa_no_penalty <- regret_prop_scapa_no_gamma <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_polynomial_generator(time_horizon, K, m, 0, 2, 4, overlap = FALSE,
                                          training = training_size)
  model_mat <- initial_model_polynomial(data$training_input, data$training_rewards, 4)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #add point anomalies
  point_anomaly_indices <- rbinom((time_horizon*K), 1, anomaly_prob)
  data$reward_mat[which(point_anomaly_indices == 1)] <- 0  
  
  #perform bandit algos
  
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$input_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  scapa_no_test_run <-scapa_ucb_contextual_linear(data$input_mat, data$reward_mat, model_mat,
                                                  1e6, alpha, 0.01, 30)
  
  scapa_no_penalty_run <- scapa_ucb_contextual_linear(data$input_mat, data$reward_mat, model_mat,
                                                      lambda, 1e-10, 0.01, 30)
  
  scapa_no_gamma_run <- scapa_ucb_contextual_linear(data$input_mat, data$reward_mat, model_mat,
                                                    lambda, alpha, 1e-10, 30)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  scapa_no_test_cumsum <- cumsum(scapa_no_test_run$rewards_received)
  scapa_no_penalty_cumsum <- cumsum(scapa_no_penalty_run$rewards_received)
  scapa_no_gamma_cumsum <- cumsum(scapa_no_gamma_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for scapa_no_test to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  scapa_no_test_regret <- oracle_cumsum - scapa_no_test_cumsum
  scapa_no_penalty_regret <- oracle_cumsum - scapa_no_penalty_cumsum
  scapa_no_gamma_regret <- oracle_cumsum - scapa_no_gamma_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  scapa_no_test_regret_mat[,i] <- scapa_no_test_regret
  scapa_no_penalty_regret_mat[,i] <- scapa_no_penalty_regret
  scapa_no_gamma_regret_mat[,i] <- scapa_no_gamma_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_test[i] <- scapa_no_test_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_penalty[i] <- scapa_no_penalty_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_scapa_no_gamma[i] <- scapa_no_gamma_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_test_cum_regret <- apply(scapa_no_test_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_penalty_cum_regret <- apply(scapa_no_penalty_regret_mat, 1, mean, na.rm = TRUE)
scapa_no_gamma_cum_regret <- apply(scapa_no_gamma_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_five <- mean(regret_prop_scapa, na.rm = TRUE)
scapa_no_test_mean_regret_five <- mean(regret_prop_scapa_no_test, na.rm = TRUE)
scapa_no_penalty_mean_regret_five <- mean(regret_prop_scapa_no_penalty, na.rm = TRUE)
scapa_no_gamma_mean_regret_five <- mean(regret_prop_scapa_no_gamma, na.rm = TRUE)


############
#Table
############
scapa_ucb_results <- c(scapa_mean_regret_one, 
                       scapa_mean_regret_two, 
                       scapa_mean_regret_three, 
                       scapa_mean_regret_four, 
                       scapa_mean_regret_five)
psscapa_no_test_results <- c(scapa_no_test_mean_regret_one, 
                   scapa_no_test_mean_regret_two, 
                   scapa_no_test_mean_regret_three, 
                   scapa_no_test_mean_regret_four, 
                   scapa_no_test_mean_regret_five)
scapa_no_penalty_results <- c(scapa_no_penalty_mean_regret_one, 
                 scapa_no_penalty_mean_regret_two, 
                 scapa_no_penalty_mean_regret_three, 
                 scapa_no_penalty_mean_regret_four, 
                 scapa_no_penalty_mean_regret_five)
scapa_no_gamma_results <- c(scapa_no_gamma_mean_regret_one, 
                   scapa_no_gamma_mean_regret_two, 
                   scapa_no_gamma_mean_regret_three, 
                   scapa_no_gamma_mean_regret_four, 
                   scapa_no_gamma_mean_regret_five)

scapa_ucb_results_table <- round(scapa_ucb_results * 100, 0)
scapa_no_test_results_table <- round(scapa_no_test_results * 100, 0)
scapa_no_penalty_results_table <- round(scapa_no_penalty_results * 100, 0)
scapa_no_gamma_results_table <- round(scapa_no_gamma_results * 100, 0)

ablation_df <- data.frame(SCAPA_UCB = scapa_ucb_results_table,
                                  scapa_no_test = scapa_no_test_results_table, 
                                  scapa_no_penaltyGreedy = scapa_no_penalty_results_table,
                                  scapa_no_gamma = scapa_no_gamma_results_table)

ablation_table <- kable(ablation_df, "html", 
                                table.attr = "style='width:30%;'") %>%
  kable_styling(font_size = 20, bootstrap_options = "striped")
save_kable(ablation_table, file = "./results/Ablation_Results.png")
