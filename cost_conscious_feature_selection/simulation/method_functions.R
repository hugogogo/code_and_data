regular <- new_method(name = "regular", label = "regular", method = function(model, draw) {
  # construct knockoff variables
  X_k <- knockoff::create.gaussian(model$X, model$mu, model$Sigma)
  # compute the knockoff statistics using lasso with cross-validation
  W <- suppressWarnings(knockoff::stat.glmnet_coefdiff(model$X, X_k, draw, nfolds = 10))
  # compute the path of select variables
  kappa <- (W > 0) * 1
  kappa[kappa == 0] <- 2
  path <- cheapknockoff::generate_path(kappa = kappa, tau = abs(W))
  return(list(X_k = X_k, stat = list(kappa = kappa, tau = abs(W)), path = path))
})

# this is a generic wrapper method to only generate multiple knockoffs
mk <- new_method(name = "mk", label = "multiple", method = function(model, draw) {
  # construct multiple knockoff variables
  X_k <- cheapknockoff::multiple_knockoff_Gaussian(X = model$X, 
                                                   mu = model$mu,
                                                   Sigma = model$Sigma,
                                                   omega = model$omega)
  # compute knockoff statistics
  stat <- cheapknockoff::stat_glmnet_coef(X = model$X,
                                          X_k = X_k, 
                                          y = draw,
                                          omega = model$omega)
  # mk filter
  # compute the path of select variables
  path <- cheapknockoff::generate_path(kappa = stat$kappa, tau = stat$tau)

  return(list(X_k = X_k, stat = stat, path = path))
})

# method extension that takes a path of selected variables and fit least squares on these variables
refit <- new_method_extension(name = "refit", label = "refitted",
                              method_extension = function(model, draw, out,
                                                          base_method) {
                                # beta is a matrix, with each column an estimate
                                # pred is a n by p matrix, with each column a prediction
                                beta <- matrix(0, model$p, model$p)
                                pred <- matrix(NA, model$n, model$p)
                                for(i in seq(model$p)){
                                  idx <- out$path[[i]]
                                  y_mean <- mean(draw)
                                  if(length(idx) == 0){
                                    pred[, i] <- y_mean
                                  }
                                  else{
                                    x <- model@params$X[, idx, drop = FALSE]
                                    x <- scale(x, center = TRUE, scale = FALSE)
                                    xtx <- crossprod(x)
                                    # add small ridge in case solution has more
                                    # than n nonzeros:
                                    diag(xtx) <- as.matrix(diag(xtx) + 1e-4)
                                    bb <- solve(xtx, crossprod(x, draw))
                                    beta[idx, i] <- bb
                                    # prediction
                                    pred[, i] <- as.numeric(y_mean + x %*% bb)
                                  }
                                }
                                return(list(beta = beta, pred = pred))
                              })