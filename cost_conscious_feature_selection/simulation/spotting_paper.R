##### This is the source code used to plot Figure 1 and Table 1 of the paper

rm(list=ls())
source("for_plot.R")
library(scales)
# we could also compute the probability that sup_k ratio_k > 1
compute_prob <- function(evals, metric_idx){
  pp <- rep(NA, length(evals))
  names(pp) <- names(evals)
  nsim <- nrow(evals[[1]][[1]])
  for(i in seq(length(pp))){
    vec <- rep(0, nsim)
    ee <- evals[[i]][[1]] / evals[[i]][[metric_idx]]
    for(j in seq(nsim)){
      vec[j] <- any(ee[j, ] > 1)
    }
    pp[i] <- mean(vec)
  }
  return(pp)
}

################ plot ###################
sim <- load_simulation("alt_tau")
lty_seq <- c(5, 1)
col_seq_original <- c("black", "red")
col_seq <- unlist(lapply(col_seq_original, scales::alpha, 0.3))
lwd_seq <- c(0.8, 0.4)
cex_seq <- c(0.5, 0.5)
cex_legend <- 1.6

pdf(file = "ratio_alt_tau.pdf", width = 10, height = 7)
mat <- matrix(seq(6), 2)
layout(mat, c(9.7, 9, 9.01), c(1, 1, 1))
par(cex.main = 1.6, cex.lab = 1.6, cex.axis = 1.2)

ylab <- "Mean squared error"
xlab <- expression(k)

# first column
par(mar = c(0, 5, 3, 0))
sub <- subset_simulation(sim = sim, subset = 1)
e <- extract_evals(sub)

# we construct three evals
# make the evals for mk contains (wfdp, practical bound)
# make the evals for regular contains (wfdp, regular bound)
#e[[1]] <- e[[1]][c(1, 2)]
#e[[2]] <- e[[2]][c(1, 4)]
ee <- list()
ee[[1]] <- list(wfdp = jitter(e[[1]][[1]], factor = 1, amount = 0), bound = e[[1]][[2]], ratio = jitter(e[[1]][[1]] / e[[1]][[2]], factor = 1, amount = 0))
ee[[2]] <- list(wfdp = jitter(e[[2]][[1]], factor = 1, amount = 0), bound = e[[2]][[2]], ratio = jitter(e[[2]][[1]] / e[[2]][[2]], factor = 1, amount = 0))

ylab <- expression(list(bar(U)(R[k], 1)^{-1}*wFDP(R[k])))
draw_one_eval(evals = ee, metric_idx = 3, xlab = "k", ylab = ylab,
           xaxis = FALSE,
           col = col_seq, lty = lty_seq, lwd = lwd_seq, cex = cex_seq,
           main = expression(list(gamma == 0)))
abline(h = 1, lwd = 1, col = "Gray")

par(mar = c(4, 5, 0, 0))
ylab <- expression(list(wFDP(R[k])))
draw_one_eval(evals = ee, metric_idx = 1, xlab = "k", ylab = ylab,
           col = col_seq, lty = lty_seq, lwd = lwd_seq, cex = cex_seq,
           main = NULL)

# second column
par(mar = c(0, 2.5, 3, 0))
sub <- subset_simulation(sim = sim, subset = 3)
e <- extract_evals(sub)
# make the evals for mk contains (wfdp, bound)
# make the evals for regular contains (wfdp, regular bound)
ee <- list()
ee[[1]] <- list(wfdp = jitter(e[[1]][[1]], factor = 1, amount = 0), bound = e[[1]][[2]], ratio = jitter(e[[1]][[1]] / e[[1]][[2]], factor = 1, amount = 0))
ee[[2]] <- list(wfdp = jitter(e[[2]][[1]], factor = 1, amount = 0), bound = e[[2]][[2]], ratio = jitter(e[[2]][[1]] / e[[2]][[2]], factor = 1, amount = 0))

ylab <- expression(list(bar(U)[k]^{-1}*wFDP[k]))

draw_one_eval(evals = ee, metric_idx = 3, xlab = "k", ylab = ylab,
           xaxis = FALSE,
           col = col_seq, lty = lty_seq, lwd = lwd_seq, cex = cex_seq,
           main = expression(list(gamma == 0.5)))
abline(h = 1, lwd = 1, col = "Gray")
                     
par(mar = c(4, 2.5, 0, 0))
ylab <- expression(list(wFDP[k]))
draw_one_eval(evals = ee, metric_idx = 1, xlab = "k", ylab = ylab, 
           col = col_seq, lty = lty_seq, lwd = lwd_seq, cex = cex_seq,
           main = NULL)

# third column
sub <- subset_simulation(sim = sim, subset = 5)
e <- extract_evals(sub)
# make the evals for mk contains (wfdp, bound)
# make the evals for regular contains (wfdp, regular bound)
ee <- list()
ee[[1]] <- list(wfdp = jitter(e[[1]][[1]], factor = 1, amount = 0), bound = e[[1]][[2]], ratio = jitter(e[[1]][[1]] / e[[1]][[2]], factor = 1, amount = 0))
ee[[2]] <- list(wfdp = jitter(e[[2]][[1]], factor = 1, amount = 0), bound = e[[2]][[2]], ratio = jitter(e[[2]][[1]] / e[[2]][[2]], factor = 1, amount = 0))

par(mar = c(0, 2.5, 3, 0.2))
ylab <- expression(list(bar(U)[k]^{-1}*wFDP[k]))

draw_one_eval(evals = ee, metric_idx = 3, xlab = "k", ylab = ylab,
           xaxis = FALSE,
           col = col_seq, lty = lty_seq, lwd = lwd_seq, cex = cex_seq,
           main = expression(list(gamma == 1)))

abline(h = 1, lwd = 1, col = "Gray")
#legend("topleft", legend = c("multiple", "regular"), 
#       col = col_seq_original, lwd = lwd_seq,
#       lty = lty_seq, cex = cex_legend)
                     
par(mar = c(4, 2.5, 0, 0.2))
ylab <- expression(list(wFDP[k]))
draw_one_eval(evals = ee, metric_idx = 1, xlab = "k", ylab = ylab,
           col = col_seq, lty = lty_seq, lwd = lwd_seq, cex = cex_seq,
           main = NULL)
dev.off()



# compute the probability
matres <- matrix(NA, ncol = 2, nrow = 5)
for(i in seq(5)){
  sub <- subset_simulation(sim = sim, subset = i)
  e <- extract_evals(sub)
  
  # we construct three evals
  # make the evals for mk contains (wfdp, practical bound)
  # make the evals for regular contains (wfdp, regular bound)
  e[[1]] <- e[[1]][c(1, 2)]
  e[[2]] <- e[[2]][c(1, 4)]
  
  matres[i,] <- compute_prob(evals = e, metric_idx = 2)
}
