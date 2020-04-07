#### This is the source code for plotting Figure 3 (model size, model cost, and prediction performance for NHANES dataset)
set.seed(123)
rm(list = ls())
library(glmnet)
library(mknockoff)
library(knockoff)
library(ROCR)
load("diabetes.RData")

# for simplicity, just consider all the real-valued features
x <- raw$features[, -raw$one_hot_idx]

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


#################################################
# start the simulation
n <- nrow(x)
p <- length(costs)

## some measures
# total model cost
measure_cost <- matrix(NA, nrow = 4, ncol = p)
# prediction accuracy
measure_accu <- matrix(NA, nrow = 4, ncol = p)
# number of selected feature
measure_card <- matrix(NA, nrow = 4, ncol = p)

## start recording classification accuracy and cost
# consider a subset of data as training set
# idx_tr <- c(sample(id1, floor(ntr * 0.5)), sample(id0, floor(ntr * 0.5)))
idx_tr <- seq(n)
x_tr <- x[idx_tr, ]
y_tr <- as.numeric(y[idx_tr])

# x_te <- x[-idx_tr, ]
# y_te <- as.numeric(y[-idx_tr])
x_te <- x
y_te <- as.numeric(y)

## our method:
ours <- run_mk(x = x_tr, y = y_tr, x_te = x_te, mu = mu, Sigma = Sigma, omega = costs, family = "binomial")

# costs_amp
costs_amp <- costs^2
ours_exp <- run_mk(x = x_tr, y = y_tr, x_te = x_te, mu = mu, Sigma = Sigma, omega = costs_amp, family = "binomial")

## mknockoff with no weights
uw <- run_mk(x = x_tr, y = y_tr, x_te = x_te, mu = mu, Sigma = Sigma, omega = rep(2, length(costs)), family = "binomial")

## run regular knockoff filter
rk <- run_rk(x = x_tr, y = y_tr, x_te = x_te, mu = mu, Sigma = Sigma, family = "binomial")

## run logistic regression on all the 30 variables
lr <- run_lr(x = x_tr, y = y_tr, x_te = x_te, family = "binomial")

for(j in seq(p)){
  # record cardinality
  measure_card[1, j] <- length(ours$path[[j]])
  measure_card[2, j] <- length(uw$path[[j]])
  measure_card[3, j] <- length(ours_exp$path[[j]])
  measure_card[4, j] <- length(uw$path[[j]])
  
  # record cost
  measure_cost[1, j] <- sum(costs[ours$path[[j]]])
  measure_cost[2, j] <- sum(costs[uw$path[[j]]])
  measure_cost[3, j] <- sum(costs_amp[ours_exp$path[[j]]])
  measure_cost[4, j] <- sum(costs_amp[uw$path[[j]]])
  
  # record auc
  pred_ours <- prediction(ours$pred[[j]], y_te)
  measure_accu[1, j] <- performance(pred_ours, measure = "auc")@y.values[[1]]
  
  pred_uw <- prediction(uw$pred[[j]], y_te)
  measure_accu[2, j] <- performance(pred_uw, measure = "auc")@y.values[[1]]
  
  pred_ours_exp <- prediction(ours_exp$pred[[j]], y_te)
  measure_accu[3, j] <- performance(pred_ours_exp, measure = "auc")@y.values[[1]]
  
  measure_accu[4, j] <- measure_accu[2, j]
}

# for regular knockoff and logistic regression
rk_card <- length(rk$path)
rk_cost <- sum(costs[rk$path])
rk_pred <- prediction(rk$pred, y_te)
rk_accu <- performance(rk_pred, measure = "auc")@y.values[[1]]

lr_card <- p
lr_cost <- sum(costs)
lr_pred <- prediction(lr$pred, y_te)
lr_accu <- performance(lr_pred, measure = "auc")@y.values[[1]]

## plot results
#######  cheap costs
pdf("NHANES.pdf", height = 4, width = 13)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 1, 1))
# accuracy
xlim <- range(c(measure_card[1:2, ], rk_card, lr_card))
ylim <- range(c(measure_accu[1:2, ], rk_accu, lr_accu))
plot(x = NULL, y = NULL, xlim = xlim, ylim = ylim, xlab = expression(paste("|", R[k], "|")), 
     ylab = "AUC", cex.axis = 2, cex.lab = 2)
lines(measure_card[1, ], measure_accu[1, ], 
      type = "b", col = "black", ylim = ylim, lwd = 3, pch = 15, cex = 1)

lines(measure_card[2, ], measure_accu[2, ], 
      col = "red", lwd = 3, pch = 17, type = "b", cex = 1)

#points(rk_card, rk_accu, col = "blue", pch = 16, cex = 2.5)
points(lr_card, lr_accu, col = "orange", pch = 8, cex = 2.5)

# cost
xlim <- range(c(measure_card[1:2, ], rk_card, lr_card))
ylim <- range(c(measure_cost[1:2, ], rk_cost, lr_cost))
plot(x = NULL, y = NULL, xlim = xlim, ylim = ylim, xlab = expression(paste("|", R[k], "|")), 
     ylab = "Cost", cex.axis = 2, cex.lab = 2)
lines(measure_card[1, ], measure_cost[1, ], 
      type = "b", col = "black", ylim = ylim, lwd = 3, pch = 15, cex = 1)

lines(measure_card[2, ], measure_cost[2, ], 
      col = "red", lwd = 3, pch = 17, type = "b", cex = 1)

#points(rk_card, rk_cost, col = "blue", pch = 16, cex = 2.5)
points(lr_card, lr_cost, col = "orange", pch = 8, cex = 2.5)

# cost-accu
xlim <- range(c(measure_cost[1:2, ], rk_cost, lr_cost))
ylim <- range(c(measure_accu[1:2, ], rk_accu, lr_accu))
plot(x = NULL, y = NULL, xlim = xlim, ylim = ylim, xlab = "Cost", 
     ylab = "AUC", cex.axis = 2, cex.lab = 2)
lines(measure_cost[1, ], measure_accu[1, ], 
      type = "b", col = "black", ylim = ylim, lwd = 3, pch = 15, cex = 1)

lines(measure_cost[2, ], measure_accu[2, ], 
      col = "red", lwd = 3, pch = 17, type = "b", cex = 1)

#points(rk_cost, rk_accu, col = "blue", pch = 16, cex = 2.5)
points(lr_cost, lr_accu, col = "orange", pch = 8, cex = 2.5)

legend("bottomright", legend = c("Our proposal", "Katsevich&Ramdas(2018)", "Logistic regression"), col = c("black", "red", "orange"), pch = c(15, 17, 8), cex = 2)
dev.off()

###### expensive costs
lr_cost_exp <- sum(costs_amp)
pdf("NHANES_exp.pdf", height = 4, width = 13)
par(mfrow = c(1, 3), mar = c(4.5, 4.5, 1, 1))
# accuracy
xlim <- range(c(measure_card[3:4, ], lr_card))
ylim <- range(c(measure_accu[3:4, ], lr_accu))
plot(x = NULL, y = NULL, xlim = xlim, ylim = ylim, xlab = expression(paste("|", R[k], "|")), 
     ylab = "AUC", cex.axis = 2, cex.lab = 2)
lines(measure_card[3, ], measure_accu[3, ], 
      type = "b", col = "black", ylim = ylim, lwd = 3, pch = 15, cex = 1)

lines(measure_card[4, ], measure_accu[4, ], 
      col = "red", lwd = 3, pch = 17, type = "b", cex = 1)

#points(rk_card, rk_accu, col = "blue", pch = 16, cex = 2.5)
points(lr_card, lr_accu, col = "orange", pch = 8, cex = 2.5)

# cost
xlim <- range(c(measure_card[3:4, ], rk_card, lr_card))
ylim <- range(c(measure_cost[3:4, ], rk_cost, lr_cost_exp))
plot(x = NULL, y = NULL, xlim = xlim, ylim = ylim, xlab = expression(paste("|", R[k], "|")), 
     ylab = "Cost", cex.axis = 2, cex.lab = 2)
lines(measure_card[3, ], measure_cost[3, ], 
      type = "b", col = "black", ylim = ylim, lwd = 3, pch = 15, cex = 1)

lines(measure_card[4, ], measure_cost[4, ], 
      col = "red", lwd = 3, pch = 17, type = "b", cex = 1)

#points(rk_card, rk_cost, col = "blue", pch = 16, cex = 2.5)
points(lr_card, lr_cost_exp, col = "orange", pch = 8, cex = 2.5)

# cost-accu
xlim <- range(c(measure_cost[3:4, ], rk_cost, lr_cost_exp))
ylim <- range(c(measure_accu[3:4, ], rk_accu, lr_accu))
plot(x = NULL, y = NULL, xlim = xlim, ylim = ylim, xlab = "Cost", 
     ylab = "AUC", cex.axis = 2, cex.lab = 2)
lines(measure_cost[3, ], measure_accu[3, ], 
      type = "b", col = "black", ylim = ylim, lwd = 3, pch = 15, cex = 1)

lines(measure_cost[4, ], measure_accu[4, ], 
      col = "red", lwd = 3, pch = 17, type = "b", cex = 1)

#points(rk_cost, rk_accu, col = "blue", pch = 16, cex = 2.5)
points(lr_cost_exp, lr_accu, col = "orange", pch = 8, cex = 2.5)

legend("bottomright", legend = c("Our proposal", "Katsevich&Ramdas(2018)", "Logistic regression"), col = c("black", "red", "orange"), pch = c(15, 17, 8), cex = 2)
dev.off()


######## Analysis of the path
# analyze the path results
path_our <- c()
path_uw <- c()
path_our_exp <- c()
for(j in seq(p)){
  path_our <- c(path_our, paste(sort(ours$path[[j]]), collapse = " "))
  path_uw <- c(path_uw, paste(sort(uw$path[[j]]), collapse = " "))
  path_our_exp <- c(path_our_exp, paste(sort(ours_exp$path[[j]]), collapse = " "))
}

inc_our <- c(ours$path[[1]])
inc_uw <- c(uw$path[[1]])
inc_our_exp <- c(ours_exp$path[[1]])
for(j in seq(2, p)){
  if(!setequal(ours$path[[j]], ours$path[[j - 1]]))
    inc_our <- c(inc_our, setdiff(ours$path[[j]], ours$path[[j - 1]]))
  if(!setequal(uw$path[[j]], uw$path[[j - 1]]))
    inc_uw <- c(inc_uw, setdiff(uw$path[[j]], uw$path[[j - 1]]))
  if(!setequal(ours_exp$path[[j]], ours_exp$path[[j - 1]]))
    inc_our_exp <- c(inc_our_exp, setdiff(ours_exp$path[[j]], ours_exp$path[[j - 1]]))
}

save(inc_our, inc_uw, inc_our_exp, type, name, costs, measure_accu, measure_cost, file = "path.RData")

### The plotting functions below has been moved to "analysis_all_plot.R"

# # Install ggplot2
# library("ggplot2")
# # Install ggrepel
# library("ggrepel")
# library("RColorBrewer")
# 
# type_ez <- type[inc_our]
# fac_type <- levels(factor(type_ez))
# type_ez[type_ez == fac_type[[1]]] <- "Demographics(2)"
# type_ez[type_ez == fac_type[[2]]] <- "Examination(5)"
# type_ez[type_ez == fac_type[[3]]] <- "Laboratory(9)"
# type_ez[type_ez == fac_type[[4]]] <- "Questionnaire(4)"
# type_ez <- factor(type_ez)
# levels(type_ez) <- levels(type_ez)[c(1, 4, 2, 3)]
# 
# 
# mydf <- data.frame(Cost = unique(measure_cost[1, ]),
#                    AUC = unique(measure_accu[1, ]),
#                    Type = type_ez)
# row.names(mydf) <- name[inc_our]
# 
# 
# pdf("path.pdf", height = 11, width = 12)
# p <- ggplot(mydf, aes(Cost, AUC)) + ggtitle("Our proposal") +
#   geom_line(linetype = "dashed")+
#   geom_point(color = 'red', size = 3) +
#   theme_classic(base_size = 20) + ylim (0.53, 0.8) + 
#   geom_label_repel(aes(label = rownames(mydf), fill = Type), 
#                      box.padding = 1,
#                      point.padding = 1,
#                      #label.padding = 1,
#                      segment.color = "pink",
#                      direction = "both",
#                      nudge_x = ifelse(mydf$Type == "Questionnaire", 1, 0),
#                      nudge_y = ifelse(mydf$Type == "Questionnaire", 4, ifelse(mydf$Type == "Examination", -4, 0)),
#                      color = "black", size = 5) 
# 
# # remove the 'a' in the legend
# p + scale_fill_manual(values=c("#D1E5F0","#92C5DE","#4393C3","#2166AC")) + guides(fill = guide_legend(title = "Type(cost)", override.aes = list(label = "", fill = c("#D1E5F0","#92C5DE","#4393C3","#2166AC")))) + theme(legend.position = c(0.8, 0.2))
# 
# dev.off()
# 
# ##### unweighted version
# type_ez <- type[inc_uw]
# fac_type <- levels(factor(type_ez))
# type_ez[type_ez == fac_type[[1]]] <- "Demographics(2)"
# type_ez[type_ez == fac_type[[2]]] <- "Examination(5)"
# type_ez[type_ez == fac_type[[3]]] <- "Laboratory(9)"
# type_ez[type_ez == fac_type[[4]]] <- "Questionnaire(4)"
# type_ez <- factor(type_ez)
# levels(type_ez) <- levels(type_ez)[c(1, 4, 2, 3)]
# 
# uwdf <- data.frame(Cost = unique(measure_cost[2, ]),
#                    AUC = unique(measure_accu[2, ]),
#                    Type = type_ez)
# row.names(uwdf) <- name[inc_uw]
# 
# 
# pdf("path_uw.pdf", height = 11, width = 12)
# p <- ggplot(uwdf, aes(Cost, AUC)) + ggtitle("Katsevich&Ramdas(2018)") +
#   geom_line(linetype = "dashed")+
#   geom_point(color = 'red', size = 3) +
#   theme_classic(base_size = 20) + ylim (0.65, 0.8) + 
#   geom_label_repel(aes(label = rownames(uwdf), fill = Type), 
#                    box.padding = 1,
#                    point.padding = 1,
#                    #label.padding = 1,
#                    segment.color = "pink",
#                    direction = "both",
#                    nudge_x = ifelse(uwdf$Type == "Questionnaire", 1, 0),
#                    nudge_y = ifelse(uwdf$Type == "Questionnaire", 4, ifelse(uwdf$Type == "Examination", -4, 0)),
#                    color = "black", size = 5) 
# 
# # remove the 'a' in the legend
# p + scale_fill_manual(values=c("#D1E5F0","#92C5DE","#4393C3","#2166AC")) + guides(fill = guide_legend(title = "Type(cost)", override.aes = list(label = "", fill = c("#D1E5F0","#92C5DE","#4393C3","#2166AC")))) + theme(legend.position = c(0.8, 0.2))
# 
# dev.off()
# 
# ##### weighted version with amplified costs
# type_ez <- type[inc_our_exp]
# fac_type <- levels(factor(type_ez))
# type_ez[type_ez == fac_type[[1]]] <- "Demographics(4)"
# type_ez[type_ez == fac_type[[2]]] <- "Examination(25)"
# type_ez[type_ez == fac_type[[3]]] <- "Laboratory(81)"
# type_ez[type_ez == fac_type[[4]]] <- "Questionnaire(16)"
# type_ez <- factor(type_ez)
# levels(type_ez) <- levels(type_ez)[c(1, 4, 2, 3)]
# 
# mydf <- data.frame(Cost = unique(measure_cost[3, ]),
#                    AUC = unique(measure_accu[3, ]),
#                    Type = type_ez)
# row.names(mydf) <- name[inc_our_exp]
# 
# 
# pdf("path_exp.pdf", height = 11, width = 12)
# p <- ggplot(mydf, aes(Cost, AUC)) + ggtitle("Our proposal (squared costs)") +
#   geom_line(linetype = "dashed")+
#   geom_point(color = 'red', size = 3) +
#   theme_classic(base_size = 20) + ylim (0.53, 0.8) + 
#   geom_label_repel(aes(label = rownames(mydf), fill = Type), 
#                    box.padding = 1,
#                    point.padding = 1,
#                    #label.padding = 1,
#                    segment.color = "pink",
#                    direction = "both",
#                    nudge_x = ifelse(mydf$Type == "Questionnaire", 1, 0),
#                    nudge_y = ifelse(mydf$Type == "Questionnaire", 4, ifelse(mydf$Type == "Examination", -4, 0)),
#                    color = "black", size = 5) 
# 
# # remove the 'a' in the legend
# p + scale_fill_manual(values=c("#D1E5F0","#92C5DE","#4393C3","#2166AC")) + guides(fill = guide_legend(title = "Type(cost)", override.aes = list(label = "", fill = c("#D1E5F0","#92C5DE","#4393C3","#2166AC")))) + theme(legend.position = c(0.8, 0.2))
# 
# dev.off()
# 
# ##### unweighted version with amplified costs
# type_ez <- type[inc_uw]
# fac_type <- levels(factor(type_ez))
# type_ez[type_ez == fac_type[[1]]] <- "Demographics(4)"
# type_ez[type_ez == fac_type[[2]]] <- "Examination(25)"
# type_ez[type_ez == fac_type[[3]]] <- "Laboratory(81)"
# type_ez[type_ez == fac_type[[4]]] <- "Questionnaire(16)"
# type_ez <- factor(type_ez)
# levels(type_ez) <- levels(type_ez)[c(1, 4, 2, 3)]
# 
# mydf <- data.frame(Cost = unique(measure_cost[4, ]),
#                    AUC = unique(measure_accu[4, ]),
#                    Type = type_ez)
# row.names(mydf) <- name[inc_uw]
# 
# pdf("path_uw_exp.pdf", height = 11, width = 12)
# p <- ggplot(mydf, aes(Cost, AUC)) + ggtitle("Katsevich&Ramdas(2018)") +
#   geom_line(linetype = "dashed")+
#   geom_point(color = 'red', size = 3) +
#   theme_classic(base_size = 20) + ylim (0.65, 0.8) + 
#   geom_label_repel(aes(label = rownames(mydf), fill = Type), 
#                    box.padding = 1,
#                    point.padding = 1,
#                    #label.padding = 1,
#                    segment.color = "pink",
#                    direction = "both",
#                    nudge_x = ifelse(mydf$Type == "Questionnaire", 1, 0),
#                    nudge_y = ifelse(mydf$Type == "Questionnaire", 4, ifelse(mydf$Type == "Examination", -4, 0)),
#                    color = "black", size = 5) 
# 
# # remove the 'a' in the legend
# p + scale_fill_manual(values=c("#D1E5F0","#92C5DE","#4393C3","#2166AC")) + guides(fill = guide_legend(title = "Type(cost)", override.aes = list(label = "", fill = c("#D1E5F0","#92C5DE","#4393C3","#2166AC")))) + theme(legend.position = c(0.8, 0.2))
# 
# dev.off()
