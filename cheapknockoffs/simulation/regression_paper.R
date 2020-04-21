rm(list=ls())
##### This is the source code used to plot Figure 2 of the paper

source("for_plot.R")
sim <- load_simulation("alt_tau_regression")
############ plot #################
pdf(file = "alt_tau_regression.pdf", width = 10, height = 4)
mat <- matrix(c(1, 2, 3), 1)
layout(mat)
par(cex.main = 1.6, cex.lab = 1.6, cex.axis = 1.2)

ylab <- "root mean squared prediction error"
xlab <- "cost"

pch_seq <- c(16, 4)
col_seq <- c(1, 2)
lwd_seq <- c(1, 1)
cex_seq <- c(1, 1)

# first columns
par(mar = c(4, 5, 3, 0))
sub <- subset_simulation(sim, subset = 1)
e <- extract_evals(sim = sub)

draw_two_evals(evals = e, xlab = xlab, ylab = ylab, main = expression(list(gamma == 0)), col = col_seq, pch = pch_seq, lwd = lwd_seq, cex = cex_seq)

# second column
par(mar = c(4, 2.5, 3, 0))
sub <- subset_simulation(sim, subset = 3)
e <- extract_evals(sim = sub)

draw_two_evals(evals = e, xlab = xlab, ylab = ylab, main = expression(list(gamma == 0.5)), col = col_seq, pch = pch_seq, lwd = lwd_seq, cex = cex_seq)

# third column
par(mar = c(4, 2.5, 3, 1))
sub <- subset_simulation(sim, subset = 5)
e <- extract_evals(sim = sub)

draw_two_evals(evals = e, xlab = xlab, ylab = ylab, main = expression(list(gamma == 1)), col = col_seq, pch = pch_seq, lwd = lwd_seq, cex = cex_seq)

#legend("topright", legend = c("multiple", "regular"), col = col_seq, lwd = lwd_seq, pch = pch_seq, cex = 1.6)
dev.off()
