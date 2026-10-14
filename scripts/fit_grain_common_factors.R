#!/usr/bin/env Rscript
# Native psych MINRES and promax; this is not the unpublished BLT implementation.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Usage: fit_grain_common_factors.R input.csv output_dir factors")
if (!requireNamespace("psych", quietly = TRUE) ||
    !requireNamespace("GPArotation", quietly = TRUE)) stop("psych/GPArotation unavailable")
out <- args[2]
dir.create(out, recursive = TRUE, showWarnings = FALSE)
x <- as.matrix(read.csv(args[1], check.names = FALSE))
if (any(!is.finite(x)) || any(apply(x, 2, sd) < 1e-12)) stop("Invalid source matrix")
warnings_seen <- character()
fit <- withCallingHandlers(
  psych::fa(x, nfactors = as.integer(args[3]), fm = "minres", rotate = "promax",
            scores = "regression", max.iter = 500, min.err = 1e-5, smooth = FALSE),
  warning = function(w) {
    warnings_seen <<- c(warnings_seen, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
emit <- function(value, name) {
  write.csv(as.matrix(value), file.path(out, paste0(name, ".csv")), row.names = FALSE)
}
phi <- if (is.null(fit$Phi)) diag(ncol(fit$loadings)) else fit$Phi
pattern <- unclass(fit$loadings)
structure <- pattern %*% phi
common <- pattern %*% phi %*% t(pattern)
z <- scale(x)
replayed <- z %*% fit$weights
if (max(abs(replayed - fit$scores)) > 1e-7) stop("Frozen score formula disagrees with psych")
emit(pattern, "pattern")
emit(phi, "phi")
emit(structure, "structure")
emit(fit$weights, "weights")
emit(fit$uniquenesses, "uniquenesses")
emit(fit$scores, "source_scores")
emit(cor(x), "observed_correlation")
off <- upper.tri(common)
summary <- c(
  authors = nrow(x), variables = ncol(x), factors = ncol(pattern),
  minimum_uniqueness = min(fit$uniquenesses),
  heywood_count = sum(fit$uniquenesses < -1e-8),
  off_diagonal_rms = sqrt(mean((cor(x)[off] - common[off])^2)),
  maximum_abs_factor_correlation = if (ncol(phi) > 1) max(abs(phi[upper.tri(phi)])) else 0,
  source_score_replay_max_error = max(abs(replayed - fit$scores)),
  minimum_observed_eigenvalue = min(eigen(cor(x), symmetric = TRUE)$values)
)
write.csv(data.frame(key = names(summary), value = unname(summary)),
          file.path(out, "diagnostics.csv"), row.names = FALSE)
writeLines(unique(warnings_seen), file.path(out, "warnings.txt"))
writeLines(c(paste("R", getRversion()), paste("psych", packageVersion("psych")),
             paste("GPArotation", packageVersion("GPArotation")),
             "Estimator: minres; native psych promax starts from varimax.",
             "psych does not return an optimizer convergence flag here; inspect warnings/diagnostics."),
           file.path(out, "versions.txt"))
