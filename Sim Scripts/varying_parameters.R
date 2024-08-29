#####################
#Varying Parameters
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

g_m <- ggplot() + geom_line(aes(x = m[1:14], y = scapa_ucb_regret_m[1:14]), col = "blue") + 
  #geom_line(aes(x = m, y = pslinucb_regret_m), col = "black", lty = 2) +
  labs(x = "m", y = "Expedcted Total Regret") + theme_idris()
ggsave("results/m_increase_paper.png", g_m)

##################
#K
##################

time_horizon <- 1000
K <- seq(2, 20, by = 2)          #number of arms
m <- 20           #number of changes
alpha <- 0.001
train_steps <- 30  #number of obs to fit each arm model for scapa ucb

n_reps <- 100

set.seed(100)

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
    
    lin_sum <- sum(lin_run$rewards_received)
    
    #add to total regret
    
    oracle_train <- apply(data$training_rewards, 1, max) %>% sum()
    oracle_reward <- apply(data$reward_mat, 1, max) %>% sum()
    
    scapa_regret_current <- scapa_regret_current + (oracle_reward - scapa_run$cumulative_reward) +
      scapa_train_reward$train_regret
    pslinucb_regret_current <- pslinucb_regret_current + oracle_train +(oracle_reward - lin_sum)
    
    print(c(k, i))
  }
  #calculate expected regret
  scapa_ucb_regret_K[k] <- scapa_regret_current / n_reps
  pslinucb_regret_K[k] <- pslinucb_regret_current / n_reps
}


#plot

g_k <- ggplot() + 
  geom_line(aes(x = K, y = scapa_ucb_regret_K), col = "blue") +
  labs(x = "K", y = "Expected Total Regret") + theme_idris()
ggsave("results/K_Increase_paper.png", g_k)

################
#R
################


time_horizon <- 1000
K <- 10        #number of arms
m <- 20           #number of changes
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

