# This is the main simulator file
rm(list=ls())
library(simulator)
library(knockoff)
library(cheapknockoff)

source("model_functions.R")
source("method_functions.R")
source("eval_functions.R")

p <- 30
n <- 200
nsim <- 100
k <- 10
# correlation between a feature being expensive and non-null
prob_list <- as.list(c(0, 0.25, 0.5, 0.75, 1))
#prob_list <- 0.5
# cost of very expensive feature
expensive <- 6
# signal-to-noise ratio
snr <- 4

# simulation on path of wfdp and its upper bounds
name_of_simulation <- "alt_tau"
label_of_simulation <- "alt_tau"

sim <- new_simulation(name = name_of_simulation,
                      label = label_of_simulation) %>%
  generate_model(make_paper, seed = 123,
                 n = n, p = p, k = k, snr = snr,  
                 prob_exp_null = prob_list,
                 expensive = expensive,
                 vary_along = "prob_exp_null") %>%
  simulate_from_model(nsim = nsim) %>%
  run_method(c(regular, mk)) %>%
  evaluate(metrics = c(wfdp, ub, ubo, ubr))
save_simulation(sim)

# simulation on prediction error and cost
sim <- load_simulation("alt_tau") %>% rename("alt_tau_regression") %>% relabel("alt_tau_regression") %>%
  run_method(c(regular + refit, mk + refit)) %>%
  subset_simulation(methods = c("regular_refit", "mk_refit")) %>%
  evaluate(metrics = c(cost, prederr))
save_simulation(sim)