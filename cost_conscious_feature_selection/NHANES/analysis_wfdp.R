#### This is the source code for doing Table 3 (validating Theorem 1 holding for NHANES dataset)
set.seed(123)
rm(list = ls())
library(glmnet)
library(mknockoff)
library(knockoff)
library(ROCR)
load("diabetes.RData")

# for simplicity, just consider all the real-valued features
x <- raw$features[, -raw$one_hot_idx]
#x[x <= 1e-10] <- 0
# blurry
# x <- x + rnorm(length(x), sd = 0.1)

# the response has three classes normal (0), pre-diabetes (1), and diabetes (2)
# for simplicity, combine pre-diabetes with diabetes and make it a 2-classification problem
y <- raw$targets
y[y == 2] <- 1

costs <- raw$costs[-raw$one_hot_idx]
name <- raw$name[-raw$one_hot_idx]
type <- raw$type[-raw$one_hot_idx]

# use sample mean / covariance as estimates
mu <- colMeans(x)
Sigma <- cov(x)
# make Sigma pd
Sigma <- Sigma + diag(0.1, ncol(Sigma))

## run multiple knockoff procedure
run_mk <- function(x, y, x_te, mu, Sigma, omega, family){
  # construct multiple knockoffs
  x_k <- mknockoff::knockoff_Gaussian(X = x, 
                                      mu = mu,
                                      Sigma = Sigma,
                                      omega = omega)
  # compute knockoff statistics
  stat <- mknockoff::stat_glmnet_coef(X = x,
                                      X_k = x_k, 
                                      y = y,
                                      omega = omega, nlam = 100, family = family)
  
  # mk filter: compute the path of select variables
  path <- mknockoff::mk_path(kappa = stat$kappa, tau = stat$tau)
  
  # given the fitted path of variables, do prediction on the left out data
  result <- mknockoff::refit(path = path, x = x, y = y, newdata = x_te, family = family)
  return(result)
}

## run regular knockoff procedure
run_rk <- function(x, y, x_te, mu, Sigma, family){
  knockoffs <- function(x) 
    create.gaussian(x, mu, Sigma)
  
  w <- function(x, x_k, y) 
    suppressWarnings(stat.glmnet_coefdiff(x, x_k, y, nfolds = 5, family = "binomial"))
  
  out = knockoff.filter(x, y, knockoffs = knockoffs, statistic = w)
  # compute the set of select variables
  idx = out$selected
  
  # added a tiny ridge
  mod <- glm(formula = y ~ ., family = family, data = data.frame(y = y, x = x[, idx]))
  pred <- as.numeric(predict(mod, newdata = data.frame(x = x_te[, idx]), type = "response"))
  
  result <- list(path = idx, mod = mod, pred = pred)
  return(result) 
}

## run logistic regression
run_lr <- function(x, y, x_te, family){
  mod <- glm(formula = y~., family = family, data = data.frame(y = y, x = x))
  #mod <- glmnet(x = x, y = y, family = family, alpha = 0, lambda = 1e-4)
  # given the fitted path of variables, do prediction on the left out data
  pred <- as.numeric(predict(object = mod, newx = x_te, type = "response"))
  
  result <- list()
  result$mod <- mod
  result$pred <- pred
  return(result) 
}

## compute weighted fdp
compute_measure <- function(out, costs, truth, alpha = 0.2, y_te){
  p <- length(out$path)
  wfdp <- rep(NA, length = p)
  fdp <- rep(NA, length = p)
  ub <- rep(NA, length = p)
  ubr <- rep(NA, length = p)
  ubo <- rep(NA, length = p)
  cost <- rep(NA, length = p)
  auc <- rep(NA, length = p)
  for(i in seq(p)){
    select <- out$path[[i]]
    cost[i] <- sum(costs[select])
    if(length(select) > 0){
      # index of false discovery
      idx <- setdiff(select, truth)
      wfdp[i] <- sum(costs[idx]) / sum(costs[select])
      fdp[i] <- length(idx) / length(select)
    }
    else{
      wfdp[i] <- 0
      fdp[i] <- 0
    }
    # upper bounds for ours and Katesvich&Ramdas
    ub[i] <- (1 + (i - length(select))) / max(sum(costs[select]), 1)
    ubr[i] <- (1 + (i - length(select))) / max(length(select), 1)
    # prediction performance in terms of AUC
    pred <- prediction(out$pred[[i]], y_te)
    auc[i] <- performance(pred, measure = "auc")@y.values[[1]]
  }
  ub <- max(costs / log(costs - (costs - 1) * (alpha))) * (-log(alpha)) * ub
  ubr <- (-log(alpha)) / log(2 - alpha)  * ubr
  wo <- costs[-truth]
  ubo <- max(wo / log(wo - (wo - 1) * (alpha))) * (-log(alpha)) * ub
  
  return(list(wfdp = wfdp, fdp = fdp, ub = ub, ubr = ubr, ubo = ubo, cost = cost, auc = auc))
}

#################################################
# start the simulation
n <- nrow(x)
p <- length(costs)

# we use 72062 randomly selected sample and run LR on it to get the "truth"
idx_tr <- sample(n, 72062)
x_tr <- x[idx_tr, ]
y_tr <- as.numeric(y[idx_tr])
# first run LR to get the "truth"
truth <- glm(formula = y~., family = "binomial", data = data.frame(y = y_tr, x = x_tr))
thresh <- 0.01
# S is the set of variables that we "believe" to be the "truth"
S <- as.integer(which(summary(truth)$coefficients[-1, 4] <= thresh))

name[head(order(summary(truth)$coefficients[-1, 4]), length(S))]

# now we randomly devide the rest 20000 samples into 50 test dataset, each consisting of 400 samples
idx_te <- seq(n)[-idx_tr]
nte <- 400
nrep <- 20000 / nte
list_test <- split(sample(idx_te), ceiling(seq_along(idx_te)/nte))

ours <- list()
uw <- list()
for(i in seq(length(list_test))){
  x_te <- x[list_test[[i]], ]
  y_te <- as.numeric(y[list_test[[i]]])
  
  ## our method:
  ours[[i]] <- run_mk(x = x_te, y = y_te, x_te = x_te, mu = mu, Sigma = Sigma, omega = costs, family = "binomial")
  ## mknockoff with no weights
  uw[[i]] <- run_mk(x = x_te, y = y_te, x_te = x_te, mu = mu, Sigma = Sigma, omega = rep(2, length(costs)), family = "binomial")
}


###############################################
# record result
alpha_list <- seq(0.05, 0.5, by = 0.05)
res_ours <- rep(NA, length(alpha_list))
res_uw <- rep(NA, length(alpha_list))
for (j in seq(length(alpha_list))){
  out_ours <- list()
  out_uw <- list()
  for(i in seq(nrep)){
    out_ours[[i]] <- compute_measure(out = ours[[i]], costs = costs, truth = S, alpha = alpha_list[j], y_te = y_te)
    out_uw[[i]] <- compute_measure(out = uw[[i]], costs = costs, truth = S, alpha = alpha_list[j], y_te = y_te)
  }
  ratio_ours_w <- matrix(NA, nrep, p)
  #ratio_ours_r <- matrix(NA, nrep, p)
  #ratio_uw_w <- matrix(NA, nrep, p)
  ratio_uw_r <- matrix(NA, nrep, p)
  for (i in seq(nrep)){
    ratio_ours_w[i, ] <- out_ours[[i]]$wfdp / out_ours[[i]]$ubo
    #ratio_ours_r[i, ] <- out_ours[[i]]$fdp / out_ours[[i]]$ubr
    #ratio_uw_w[i, ] <- out_ours[[i]]$wfdp / out_ours[[i]]$ubo
    ratio_uw_r[i, ] <- out_ours[[i]]$fdp / out_ours[[i]]$ubr
  }
  
  res_ours[j] <- mean(apply(ratio_ours_w, 1, max) >=1 )
  res_uw[j] <- mean(apply(ratio_uw_r, 1, max) >= 1)
}