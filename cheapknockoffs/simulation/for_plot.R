extract_evals <- function(sim){
  # this function extract the evals as a list
  # with each element corresponding to all the evaluations of a method
  nsim <- length(draws(sim)@draws)
  e <- evals(sim)
  
  result <- list()
  for(i in seq(length(e@method_name))){
    result[[i]] <- list()
    for(j in seq(length(e@metric_name) - 1)){
      result[[i]][[j]] <- matrix(NA, nrow = nsim, ncol = model(sim)@params$p)
      for (k in seq(nsim)){
        result[[i]][[j]][k, ] <- e@evals[[i]][[k]][[j]]
      }
    }
    names(result[[i]]) <- e@metric_label[1:(length(e@metric_name) - 1)]
  }
  names(result) <- e@method_label
  return(result)
}

draw_one_eval <- function(evals, metric_idx, xlab, ylab, xaxis = TRUE, main, col, lty, lwd, cex){
  # evals is a list, with each cell representing the evaluations of a method
  # the evaluation of a method is a matrix
  n_method <- length(evals)
  K <- ncol(evals[[1]][[1]])
  nsim <- nrow(evals[[1]][[1]])
  # determine the x, y limits
  xmin <- 0
  xmax <- K
  ymin <- 0
  ymax <- 0
  for(i in seq(n_method)){
    ymax <- max(ymax, max(evals[[i]][[metric_idx]]))
  }
  if(xaxis)
    plot(0, 0, xlim = c(xmin, xmax), ylim = c(ymin, ymax), main = main, type = "n", xlab = xlab, ylab = ylab)
  else
    plot(0, 0, xlim = c(xmin, xmax), ylim = c(ymin, ymax), xaxt = "n", main = main, type = "n", xlab = xlab, ylab = ylab)
  
  for (i in seq(n_method)){
    # compute the ratio instead
    ee <- evals[[i]][[metric_idx]]
    for (l in seq(nsim)){
      points(x = seq(K), y = ee[l, ], col = col[i], lty = lty[i], type = "l", lwd = lwd[i], cex = cex[i])
    }
  }
}

draw_evals <- function(evals, metric_idx, xlab, ylab, xaxis = TRUE, main, col, lty, lwd, cex){
  # evals is a list, with each cell representing the evaluations of a method
  # the evaluation of a method is a matrix
  n_method <- length(evals)
  K <- ncol(evals[[1]][[1]])
  nsim <- nrow(evals[[1]][[1]])
  # determine the x, y limits
  xmin <- 0
  xmax <- K
  ymin <- 0
  ymax <- 1
  for(i in seq(n_method)){
    ymax <- max(ymax, max(evals[[i]][[1]] / evals[[i]][[metric_idx]]))
  }
  if(xaxis)
    plot(0, 0, xlim = c(xmin, xmax), ylim = c(ymin, ymax), main = main, type = "n", xlab = xlab, ylab = ylab)
  else
    plot(0, 0, xlim = c(xmin, xmax), ylim = c(ymin, ymax), xaxt = "n", main = main, type = "n", xlab = xlab, ylab = ylab)
    
  for (i in seq(n_method)){
    # compute the ratio instead
    ee <- evals[[i]][[1]] / evals[[i]][[metric_idx]]
    for (l in seq(nsim)){
      points(x = seq(K), y = ee[l, ], col = col[i], lty = lty[i], type = "l", lwd = lwd[i], cex = cex[i])
    }
  }
}

draw_two_evals <- function(evals, xlab, ylab, main, col, pch, lwd, cex){
  # evals is a list, with each cell representing the evaluations of a method
  # the evaluation of a method is a matrix
  n_method <- length(evals)
  
  # determine the x, y limits
  xmin <- 0
  xmax <- 0
  ymin <- 0
  ymax <- 0
  for(i in seq(n_method)){
    xmax <- max(xmax, max(colMeans(evals[[i]][[1]])))
    ymax <- max(ymax, max(colMeans(evals[[i]][[2]])))
  }
  
  plot(0, 0, xlim = c(xmin, xmax), ylim = c(ymin, ymax), main = main, type = "n", xlab = xlab, ylab = ylab)
  
  for (i in seq(n_method)){
    xx <- colMeans(evals[[i]][[1]])
    yy <- colMeans(evals[[i]][[2]])
    points(x = xx, y = yy, col = col[i], pch = pch[i], type = "b", lwd = lwd[i], cex = cex[i])
  }
}