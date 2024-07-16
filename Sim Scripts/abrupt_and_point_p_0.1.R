##################
#Abrupt Change With Point Anomalies
###################

time_horizon <- 1000
K <- 10         #number of arms
m <- 20     #number of changes
train_steps <- 30  #number of obs to fit each arm model for scapa ucb
training_size <- K*train_steps  #number of training points to generate

alpha <- 0.001
lambda <- log(1000)


ada_L <- 100000
ada_threshold <- 100
ada_variation <- 100000^(-1/3)
ada_delta <- 0.001

m_ucb_w <- 250 #set these using formula given in Cao(2019)
m_ucb_b <- 266 

n_reps <- 100
anomaly_prob <- 0.1

set.seed(150)

####################

#Gaussian Noise - Scenario One

###################

pslinucb_regret_mat <- scapa_regret_mat <- ada_regret_mat <- m_ucb_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_lin <- regret_prop_ada <- regret_prop_m_ucb <- rep(NA, n_reps)

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
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  ada_run <- ada_greedy(data$reward_mat, ada_L, ada_threshold, ada_variation, ada_delta)
  
  m_ucb_run <- m_ucb(data$reward_mat, m_ucb_w, m_ucb_b, 0.05)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  ada_cumsum <- cumsum(ada_run$rewards_received)
  m_ucb_cumsum <- cumsum(m_ucb_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  ada_regret <- oracle_cumsum - ada_cumsum
  m_ucb_regret <- oracle_cumsum - m_ucb_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  ada_regret_mat[,i] <- ada_regret
  m_ucb_regret_mat[,i] <- m_ucb_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_ada[i] <- ada_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_m_ucb[i] <- m_ucb_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)
ada_cum_regret <- apply(ada_regret_mat, 1, mean, na.rm = TRUE)
m_ucb_cum_regret <- apply(m_ucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_one <- mean(regret_prop_scapa, na.rm = TRUE)
pslin_mean_regret_one <- mean(regret_prop_lin, na.rm = TRUE)
ada_mean_regret_one <- mean(regret_prop_ada, na.rm = TRUE)
m_ucb_mean_regret_one <- mean(regret_prop_m_ucb, na.rm = TRUE)

###############
#Heavy Tailed Noise - Two
###############

set.seed(151)

pslinucb_regret_mat <- scapa_regret_mat <- ada_regret_mat <- m_ucb_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_lin <- regret_prop_ada <- regret_prop_m_ucb <- rep(NA, n_reps)

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
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  ada_run <- ada_greedy(data$reward_mat, ada_L, ada_threshold, ada_variation, ada_delta)
  
  m_ucb_run <- m_ucb(data$reward_mat, m_ucb_w, m_ucb_b, 0.05)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  ada_cumsum <- cumsum(ada_run$rewards_received)
  m_ucb_cumsum <- cumsum(m_ucb_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  ada_regret <- oracle_cumsum - ada_cumsum
  m_ucb_regret <- oracle_cumsum - m_ucb_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  ada_regret_mat[,i] <- ada_regret
  m_ucb_regret_mat[,i] <- m_ucb_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_ada[i] <- ada_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_m_ucb[i] <- m_ucb_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)
ada_cum_regret <- apply(ada_regret_mat, 1, mean, na.rm = TRUE)
m_ucb_cum_regret <- apply(m_ucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_two <- mean(regret_prop_scapa, na.rm = TRUE)
pslin_mean_regret_two <- mean(regret_prop_lin, na.rm = TRUE)
ada_mean_regret_two <- mean(regret_prop_ada, na.rm = TRUE)
m_ucb_mean_regret_two <- mean(regret_prop_m_ucb, na.rm = TRUE)

####################

#poisson GLM - Three

###################

set.seed(152)

pslinucb_regret_mat <- scapa_regret_mat <- ada_regret_mat <- m_ucb_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_lin <- regret_prop_ada <- regret_prop_m_ucb <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_poisson_glm_generator(time_horizon, K, m, 1, 0, 0, overlap = FALSE,
                                           training = training_size, delta_coeff = 1,
                                           delta_feature = 3)
  model_mat <- initial_model_poisson_glm(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps) 
  
  # #add point anomalies
  # point_anomaly_indices <- rbinom((time_horizon*K), 1, anomaly_prob)
  # data$reward_mat[which(point_anomaly_indices == 1)] <- 1 #avoid zero-inflation 
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  ada_run <- ada_greedy(data$reward_mat, ada_L, ada_threshold, ada_variation, ada_delta)
  
  m_ucb_run <- m_ucb(data$reward_mat, m_ucb_w, m_ucb_b, 0.05)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  ada_cumsum <- cumsum(ada_run$rewards_received)
  m_ucb_cumsum <- cumsum(m_ucb_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  ada_regret <- oracle_cumsum - ada_cumsum
  m_ucb_regret <- oracle_cumsum - m_ucb_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  ada_regret_mat[,i] <- ada_regret
  m_ucb_regret_mat[,i] <- m_ucb_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_ada[i] <- ada_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_m_ucb[i] <- m_ucb_regret[time_horizon] / oracle_cumsum[time_horizon] 
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)
ada_cum_regret <- apply(ada_regret_mat, 1, mean, na.rm = TRUE)
m_ucb_cum_regret <- apply(m_ucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_three <- mean(regret_prop_scapa, na.rm = TRUE)
pslin_mean_regret_three <- mean(regret_prop_lin, na.rm = TRUE)
ada_mean_regret_three <- mean(regret_prop_ada, na.rm = TRUE)
m_ucb_mean_regret_three <- mean(regret_prop_m_ucb, na.rm = TRUE)

####################

#Gamma GLM - Four

###################

set.seed(153)

pslinucb_regret_mat <- scapa_regret_mat <- ada_regret_mat <- m_ucb_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_lin <- regret_prop_ada <- regret_prop_m_ucb <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_gamma_glm_generator(time_horizon, K, m, 1, 0, 0, shape = 20, overlap = FALSE,
                                         training = training_size, delta_coeff = 1,
                                         delta_feature = 3)
  model_mat <- initial_model_gamma_glm(data$training_features, data$training_rewards)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps) 
  
  #add point anomalies
  point_anomaly_indices <- rbinom((time_horizon*K), 1, anomaly_prob)
  data$reward_mat[which(point_anomaly_indices == 1)] <- 1 
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- rbind(data$training_features, data$feature_mat)
  
  #perform bandit algos
  
  scapa_run <- scapa_ucb_contextual_linear(data$feature_mat, data$reward_mat, model_mat,
                                           lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  ada_run <- ada_greedy(data$reward_mat, ada_L, ada_threshold, ada_variation, ada_delta)
  
  m_ucb_run <- m_ucb(data$reward_mat, m_ucb_w, m_ucb_b, 0.05)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  ada_cumsum <- cumsum(ada_run$rewards_received)
  m_ucb_cumsum <- cumsum(m_ucb_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  ada_regret <- oracle_cumsum - ada_cumsum
  m_ucb_regret <- oracle_cumsum - m_ucb_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  ada_regret_mat[,i] <- ada_regret
  m_ucb_regret_mat[,i] <- m_ucb_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_ada[i] <- ada_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_m_ucb[i] <- m_ucb_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)
ada_cum_regret <- apply(ada_regret_mat, 1, mean, na.rm = TRUE)
m_ucb_cum_regret <- apply(m_ucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_four <- mean(regret_prop_scapa, na.rm = TRUE)
pslin_mean_regret_four <- mean(regret_prop_lin, na.rm = TRUE)
ada_mean_regret_four <- mean(regret_prop_ada, na.rm = TRUE)
m_ucb_mean_regret_four <- mean(regret_prop_m_ucb, na.rm = TRUE)

####################

#Polynomial - Five

###################

set.seed(154)

pslinucb_regret_mat <- scapa_regret_mat <- ada_regret_mat <- m_ucb_regret_mat <- matrix(NA, nrow = time_horizon, ncol = n_reps)
regret_prop_scapa <- regret_prop_lin <- regret_prop_ada <- regret_prop_m_ucb <- rep(NA, n_reps)

for(i in 1:n_reps){
  #generate data and fit initial model for SCAPA
  data <- contextual_polynomial_generator(time_horizon, K, m, 0, 2, 4, overlap = FALSE,
                                          training = training_size)
  model_mat <- initial_model_polynomial(data$training_input, data$training_rewards, 4)
  scapa_train_reward <- training_cost(data$training_rewards, train_steps)
  
  #add point anomalies
  point_anomaly_indices <- rbinom((time_horizon*K), 1, anomaly_prob)
  data$reward_mat[which(point_anomaly_indices == 1)] <- 0  
  
  #create data features and rewards that include training data to put all of it into pslinucb
  
  ps_data_rewards <- rbind(data$training_rewards, data$reward_mat)
  ps_data_features <- c(data$training_input, data$input_mat) %>% as.matrix(ncol = 1)
  ps_data_features <- cbind(rep(1, nrow(ps_data_features)), ps_data_features)
  
  #perform bandit algos
  
  
  scapa_run <- scapa_ucb_contextual_polynomial(data$input_mat,
                                               data$reward_mat, model_mat,
                                               lambda, alpha, 0.01, 30)
  
  lin_run <- pslinucb(ps_data_features, ps_data_rewards, window_size = 100, alpha = 24,
                      threshold = 1)
  
  ada_run <- ada_greedy(data$reward_mat, ada_L, ada_threshold, ada_variation, ada_delta)
  
  m_ucb_run <- m_ucb(data$reward_mat, m_ucb_w, m_ucb_b, 0.05)
  
  #calculate cumulative rewards and then the regret
  oracle_cumsum <- apply(data$reward_mat, 1, max) %>% cumsum
  scapa_cumsum <- cumsum(scapa_run$rewards_received)
  lin_cumsum <- cumsum(lin_run$rewards_received[(training_size + 1): length(lin_run$rewards_received)])
  ada_cumsum <- cumsum(ada_run$rewards_received)
  m_ucb_cumsum <- cumsum(m_ucb_run$rewards_received)
  
  #note we count regret from when the test period starts; we allow for PSLinUCB to train 
  #on the same amount of data as SCAPA-UCB before counting regret
  
  scapa_regret <- oracle_cumsum - scapa_cumsum
  lin_regret <- oracle_cumsum - lin_cumsum
  ada_regret <- oracle_cumsum - ada_cumsum
  m_ucb_regret <- oracle_cumsum - m_ucb_cumsum
  
  scapa_regret_mat[,i] <- scapa_regret
  pslinucb_regret_mat[,i] <- lin_regret
  ada_regret_mat[,i] <- ada_regret
  m_ucb_regret_mat[,i] <- m_ucb_regret
  
  regret_prop_scapa[i] <- scapa_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_lin[i] <- lin_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_ada[i] <- ada_regret[time_horizon] / oracle_cumsum[time_horizon]
  regret_prop_m_ucb[i] <- m_ucb_regret[time_horizon] / oracle_cumsum[time_horizon]
  
  print(i)
  
}

scapa_cum_regret <- apply(scapa_regret_mat, 1, mean, na.rm = TRUE)
pslin_cum_regret <- apply(pslinucb_regret_mat, 1, mean, na.rm = TRUE)
ada_cum_regret <- apply(ada_regret_mat, 1, mean, na.rm = TRUE)
m_ucb_cum_regret <- apply(m_ucb_regret_mat, 1, mean, na.rm = TRUE)

scapa_mean_regret_five <- mean(regret_prop_scapa, na.rm = TRUE)
pslin_mean_regret_five <- mean(regret_prop_lin, na.rm = TRUE)
ada_mean_regret_five <- mean(regret_prop_ada, na.rm = TRUE)
m_ucb_mean_regret_five <- mean(regret_prop_m_ucb, na.rm = TRUE)


############
#Table
############
scapa_ucb_results <- c(scapa_mean_regret_one, 
                       scapa_mean_regret_two, 
                       scapa_mean_regret_three, 
                       scapa_mean_regret_four, 
                       scapa_mean_regret_five)
pslin_results <- c(pslin_mean_regret_one, 
                   pslin_mean_regret_two, 
                   pslin_mean_regret_three, 
                   pslin_mean_regret_four, 
                   pslin_mean_regret_five)
ada_results <- c(ada_mean_regret_one, 
                 ada_mean_regret_two, 
                 ada_mean_regret_three, 
                 ada_mean_regret_four, 
                 ada_mean_regret_five)
m_ucb_results <- c(m_ucb_mean_regret_one, 
                   m_ucb_mean_regret_two, 
                   m_ucb_mean_regret_three, 
                   m_ucb_mean_regret_four, 
                   m_ucb_mean_regret_five)

scapa_ucb_results_table <- round(scapa_ucb_results * 100, 0)
pslin_results_table <- round(pslin_results * 100, 0)
ada_results_table <- round(ada_results * 100, 0)
m_ucb_results_table <- round(m_ucb_results * 100, 0)

abrupt_and_point_df <- data.frame(SCAPA_UCB = scapa_ucb_results_table,
                                  PSLinUCB = pslin_results_table, 
                                  ADA = ada_results_table,
                                  M_UCB = m_ucb_results_table)

abrupt_and_point_table <- kable(abrupt_and_point_df, "html", 
                                table.attr = "style='width:30%;'") %>%
  kable_styling(font_size = 20, bootstrap_options = "striped")
save_kable(abrupt_and_point_table, file = "./results/p_0.1_Abrupt_And_Point_Results.png")
