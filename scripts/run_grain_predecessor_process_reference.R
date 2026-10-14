#!/usr/bin/env Rscript
# Known-model interface checks, not new human findings or full DSEM/selection-IRT.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: script.R OUTPUT UPSTREAM_SOURCE_DIR")
out <- args[[1]]
upstream <- args[[2]]
if (dir.exists(out)) stop("Use a new output directory")
dir.create(out, recursive = TRUE)
set.seed(731)
capture.output(sessionInfo(), file = file.path(out, "R_session.txt"))
records <- data.frame()
add <- function(model, quantity, estimate, truth = NA_real_, scope = "known_model_simulation") {
  records <<- rbind(records, data.frame(model, quantity, estimate, truth, scope))
}

# Random intercepts/slopes: a person can differ in level and in condition response.
suppressPackageStartupMessages(library(lme4))
n <- 100L
conditions <- 8L
occasions <- 4L
d <- expand.grid(author = seq_len(n), condition = seq_len(conditions), occasion = seq_len(occasions))
a <- rnorm(n, sd = 0.8)
b <- rnorm(n, sd = 0.6)
c <- rnorm(conditions, sd = 0.4)
d$z <- seq(-1.5, 1.5, length.out = conditions)[d$condition]
d$score <- 0.3 + a[d$author] + (0.5 + b[d$author]) * d$z + c[d$condition] + rnorm(nrow(d), sd = 0.5)
d$author <- factor(d$author)
d$condition <- factor(d$condition)
fit <- lmer(score ~ z + (1 + z | author) + (1 | condition), data = d,
            control = lmerControl(check.conv.singular = "ignore"))
effects <- ranef(fit)$author
add("lme4_response", "author_intercept_correlation", cor(effects[,1], a))
add("lme4_response", "author_slope_correlation", cor(effects[,2], b))
add("lme4_response", "residual_sd", sigma(fit), 0.5)
add("lme4_response", "singular", as.numeric(isSingular(fit)))
write.csv(as.data.frame(VarCorr(fit)), file.path(out, "variance_components.csv"), row.names = FALSE)
saveRDS(fit, file.path(out, "response_model.rds"))

# Opportunity is observed in this known world. The offset is not an IRT substitute.
preference <- rnorm(n, sd = 0.45)
exposure <- rep(c(20, 200), length.out = n)
selected <- rpois(n, exposure * exp(preference))
add("observed_exposure", "count_preference_correlation", cor(selected, preference))
add("observed_exposure", "rate_preference_correlation", cor(selected/exposure, preference))
writeLines("Algebraic scope: count alone identifies exposure times rate, not both factors; this is not a numeric test.",
           file.path(out, "opportunity_scope.txt"))

# An OU process with known observations: exact irregular-time transition likelihood.
theta <- 0.7
diffusion <- 0.5
times <- c(0, cumsum(runif(500, 0.05, 2)))
dt <- diff(times)
x <- numeric(length(times))
for (i in seq_along(dt)) {
  f <- exp(-theta*dt[[i]])
  x[[i+1]] <- f*x[[i]] + rnorm(1, sd = sqrt(diffusion^2/(2*theta)*(1-f^2)))
}
ou_loss <- function(parameters) {
  rate <- exp(parameters[[1]])
  sigma <- exp(parameters[[2]])
  f <- exp(-rate*dt)
  variance <- sigma^2/(2*rate)*(1-f^2)
  0.5*sum(log(2*pi*variance) + (x[-1]-f*x[-length(x)])^2/variance)
}
ou <- optim(log(c(0.5, 0.5)), ou_loss, method = "BFGS", hessian = TRUE)
add("irregular_OU", "decay_rate", exp(ou$par[[1]]), theta)
add("irregular_OU", "diffusion", exp(ou$par[[2]]), diffusion)
add("irregular_OU", "optimizer_convergence_code", ou$convergence, 0)
add("irregular_OU", "naive_unit_lag_coefficient", coef(lm(x[-1] ~ x[-length(x)]))[[2]], NA_real_,
    "a_unit_lag_coefficient_is_not_a_per_time_decay_rate")

# Run the actual upstream sJIVE estimator on its intended shared/view-specific model.
source(file.path(upstream, "sJIVE_functions.r"), local = TRUE)
k <- 140L
scores <- matrix(rnorm(k*3), 3, k)
scores <- t(qr.Q(qr(t(scores)))) * sqrt(k)
load1 <- qr.Q(qr(matrix(rnorm(12*2), 12, 2)))
load2 <- qr.Q(qr(matrix(rnorm(10*2), 10, 2)))
x1 <- load1[,1] %o% scores[1,] + load1[,2] %o% scores[2,] + matrix(rnorm(12*k, sd = 0.04), 12, k)
x2 <- load2[,1] %o% scores[1,] + load2[,2] %o% scores[3,] + matrix(rnorm(10*k, sd = 0.04), 10, k)
y <- scores[1,] + 0.6*scores[2,] - 0.5*scores[3,] + rnorm(k, sd = 0.15)
tr <- 1:100
te <- 101:k
views <- list(x1, x2)
means <- lapply(views, function(v) rowMeans(v[,tr]))
centered <- Map(function(v, m) sweep(v, 1, m, "-"), views, means)
ym <- mean(y[tr])
sj <- sJIVE.converge(lapply(centered, function(v) v[,tr]), y[tr]-ym,
                    rankJ = 1, rankA = c(1,1), eta = 0.5,
                    center.scale = FALSE, reduce.dim = FALSE,
                    max.iter = 500, threshold = 1e-7, show.message = FALSE)
pred <- sJIVE.predict(sj, lapply(centered, function(v) v[,te]), threshold = 1e-7)
add("upstream_sJIVE", "heldout_target_r", cor(as.numeric(pred$Ypred), y[te]))
add("upstream_sJIVE", "shared_score_abs_r", abs(cor(as.numeric(pred$Sj), scores[1,te])))
add("upstream_sJIVE", "view1_specific_abs_r", abs(cor(as.numeric(pred$Si[[1]]), scores[2,te])))
add("upstream_sJIVE", "view2_specific_abs_r", abs(cor(as.numeric(pred$Si[[2]]), scores[3,te])))
add("upstream_sJIVE", "fit_iterations", sj$iterations)
saveRDS(sj, file.path(out, "sjive_model.rds"))
write.csv(records, file.path(out, "known_model_results.csv"), row.names = FALSE)
print(records)
writeLines("COMPLETE: source model checks only, not psychological validity", file.path(out, "status.txt"))
