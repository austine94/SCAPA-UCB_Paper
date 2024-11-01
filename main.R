library(tidyverse)
library(ScapaAnomaly)
library(kableExtra)

#########################
#import functions for sims
##########################

source("./code/ada_greedy.R")
source("./code/contextual_gamma_glm_generator.R")
source("./code/contextual_gaussian_linear_generator.R")
source("./code/contextual_gaussian_linear_ar1_generator.R")
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
source("./code/M-UCB.R")
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
source("./code/SCAPA_Contextual.R")
source("./code/scapa_ucb_contextual_gamma_glm_no_anomaly.R")
source("./code/scapa_ucb_contextual_linear_no_anomaly.R")
source("./code/scapa_ucb_contextual_linear_no_change.R")
source("./code/scapa_ucb_contextual_linear_no_gamma.R")
source("./code/scapa_ucb_contextual_poisson_glm_no_anomaly.R")
source("./code/scapa_ucb_contextual_polynomial_no_anomaly.R")
source("./code/scapa_ucb_contextual_polynomial_no_change.R")
source("./code/scapa_ucb_contextual_polynomial_no_gamma.R")
source("./code/training_cost.R")

theme_idris <- function() {
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "grey20"),
    panel.border =  element_rect(fill = NA,
                                 colour = "grey20"),
  )
}

#######
#Code for Sims
#######

source("./Sim Scripts/ablation_study.R")
source("./Sim Scripts/abrupt_and_point_p_0.1.R")
source("./Sim Scripts/abrupt_and_point_p_0.05.R")
source("./Sim Scripts/abrupt_and_point.R")
source("./Sim Scripts/abrupt_change.R")
source("./Sim Scripts/abrupt_magnitude.R")
source("./Sim Scripts/drift_nonstationarity.R")
source("./Sim Scripts/varying_parameters.R")




