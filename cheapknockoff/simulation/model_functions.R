reject_sample <- function(n, pmat)
{
  result <- data.frame(nonnull = numeric(n), exp = numeric(n))
  i <- 0
  while (i<n)
  {
    xval <- 1 * (runif(1, 0, 1) <= 0.5)
    yval <- 1 * (runif(1, 0, 1) <= 0.5)
    if (runif(1) < pmat[xval + 1, yval + 1] / max(pmat))
    {
      i <- i + 1
      result[i, ] <- c(xval, yval)
    }
  }
  return(result)
}

# a general model
make_model <- function(n, p, ph, pe, rho, expensive, snr, feature_cor, delta = 2) {
  # ph: probability that a hypothesis is non-null
  # pe: probability that a hypothesis is expensive
  # rho: the correlation between a hypothesis being non-null and expensive
  # expensive: the cost of an expensive hypothesis (cheap feature cost 2)
  # delta: the value of non-zero beta's
  
  # construct mean vector
  beta <- rep(0, p)
  # construct cost vector
  omega <- rep(0, p)
  
  pmat <- matrix(NA, 2, 2)
  p11 <- rho * sqrt(pe * ph * (1 - pe) * (1 - ph)) + pe * ph
  p10 <- ph - p11
  p01 <- pe - p11
  p00 <- 1 - p11 - p10 - p01
  pmat[1, 1] <- p00
  pmat[1, 2] <- p01
  pmat[2, 1] <- p10
  pmat[2, 2] <- p11
  
  re <- reject_sample(p, pmat)
  # specify beta and omega
  beta <- delta * re$nonnull
  omega <- expensive * re$exp + 2 * (re$exp == 0)
  
  # Covariance matrix of the Gaussian design
  # Sigma <- toeplitz(feature_cor^(0:(p-1)))
  Sigma <- diag(1, p, p)
  # mu is the mean of the population Gaussian distribution
  mu <- rep(0, p)
  
  X <- scale(matrix(rnorm(n * p), n) %*% chol(Sigma))
  
  signal <- as.numeric(X %*% beta)
  sigma <- sqrt(sum(signal^2) / (n * snr))
  if(sigma == 0) sigma <- 1
  
  # generate model
  new_model(name = "basic",
            label = sprintf("pe = %s, ph = %s, rho = %s", pe, ph, rho),
            params = list(n = n, p = p, ph = ph, pe = pe, snr = snr, rho = rho,
                          expensive = expensive, delta = delta,
                          omega = omega,
                          sigma = sigma, 
                          mu = mu,
                          Sigma = Sigma,
                          feature_cor = feature_cor,
                          beta = beta,
                          signal = signal,
                          X = X),
            simulate = function(n, signal, sigma, nsim) {
              # this function must return a list of length nsim
              y <- signal + sigma * matrix(rnorm(nsim * n), n, nsim) 
              return(split(y, col(y)))
            })
}

# a example that multiple should gain much more than regular in prederr-vs-cost
make_paper <- function(n, p, k, snr, prob_exp_null, expensive) {
  # mean vector of the Guassian design
  mu <- rep(0, p)
  
#  # Cholesky factor of Sigma
#  cholesky <- diag(1, nrow = p, ncol = p)
#  diag(cholesky[-p, -1]) <- feature_cor
#  
#  # Covariance matrix of the Gaussian design
#  Sigma <- crossprod(cholesky)
#  X <- scale(matrix(rnorm(n * p), n) %*% cholesky)
  
  Sigma <- diag(1, nrow = p, ncol = p)
  X <- scale(matrix(rnorm(n * p), n))
  
  idx <- seq(k)
  beta <- rep(0, p)
  beta[idx] <- 2
  
  signal <- as.numeric(X %*% beta)
  sigma <- sqrt(sum(signal^2) / (n * snr))
  
  # weights
  omega <- rep(NA, p)
  omega[idx] <- c(rep.int(expensive, times = k / 2), rep.int(2, times = k / 2))
  exp_idx <- rbinom(n = p - length(idx), size = 1, prob = prob_exp_null)
  omega[-idx] <- exp_idx * expensive + (1 - exp_idx) * 2
  
  # generate model
  new_model(name = "paper",
            label = sprintf("snr = %s, k = %s, prob = %s", snr, k, prob_exp_null),
            params = list(n = n, p = p, k = k, snr = snr, 
                          prob_exp_null = prob_exp_null,
                          expensive = expensive,
                          omega = omega,
                          sigma = sigma,
                          Sigma = Sigma,
                          mu = mu,
                          beta = beta,
                          signal = signal,
                          X = X),
            simulate = function(n, signal, sigma, nsim) {
              # this function must return a list of length nsim
              y <- signal + sigma * matrix(rnorm(nsim * n), n, nsim) 
              return(split(y, col(y)))
            })
}