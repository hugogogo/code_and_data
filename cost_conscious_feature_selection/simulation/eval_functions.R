## @knitr metrics
wfdp <- new_metric("wfdp", "wfdp", metric = function(model, out){
  wfdp <- rep(NA, length = model$p)
  for(i in seq(model$p)){
    select <- out$path[[i]]
    if(length(select) > 0){
      # index of false discovery
      idx <- select[model$beta[select] == 0]
      idx_alternative <- setdiff(select, which(model$beta != 0))
      stopifnot(all.equal(idx, idx_alternative))
      wfdp[i] <- sum(model$omega[idx]) / sum(model$omega[select])
    }
    else{
      wfdp[i] <- 0
    }
  }
  return(wfdp)
})

ub <- new_metric("ub", "bound", metric = function(model, out){
  alpha <- 0.2
  c <- 1
  weight <- model$omega
  multi <- max(weight / log(weight - (weight - 1) * (alpha^c))) * (-log(alpha))
  ub <- rep(NA, length = model$p)
  for(i in seq(model$p)){
    select <- out$path[[i]]
    ub[i] <- (1 + c * (i - length(select))) / max(sum(weight[select]), 1)
  }
  ub <- ub * multi
  return(ub)
})

ubr <- new_metric("ubr", "regular bound", metric = function(model, out){
  alpha <- 0.2
  ub <- rep(NA, length = model$p)
  multi <- (-log(alpha)) / log(2 - alpha)
  for(i in seq(model$p)){
    select <- out$path[[i]]
    ub[i] <- (1 + (i - length(select))) / max(length(select), 1)
  }
  ub <- ub * multi
  return(ub)
})

ubo <- new_metric("ubo", "oracle bound", metric = function(model, out){
  alpha <- 0.2
  c <- 1
  weight <- model$omega
  wo <- model$omega[model$beta == 0]
  multi <- max(wo / log(wo - (wo - 1) * (alpha^c))) * (-log(alpha))
  ub <- rep(NA, length = model$p)
  for(i in seq(model$p)){
    select <- out$path[[i]]
    ub[i] <- (1 + c * (i - length(select))) / max(sum(weight[select]), 1)
  }
  ub <- ub * multi
  return(ub)
})

prederr <- new_metric("prederr", "(root mean squared) Prediction error", metric = function(model, out){
  err <- rep(NA, length = model$p)
  for(i in seq(model$p)){
    err[i] <- sqrt(mean((model$signal - out$pred[, i])^2))
  }
  return(err)
}
)

cost <- new_metric("cost", "cost", metric = function(model, out){
  cost <- rep(NA, length = model$p)
  for(i in seq(model$p)){
    cost[i] <- sum(model$omega[out$beta[, i] != 0])
  }
  return(cost)
}
)