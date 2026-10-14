#!/usr/bin/env Rscript
# Actual pinned sJIVE, fixed-rank local adapter, not original-paper replication.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: script.R OUTPUT UPSTREAM_SOURCE_DIR")
out <- args[[1]]
source(file.path(args[[2]], "sJIVE_functions.r"), local = TRUE)
capture.output(sessionInfo(), file = file.path(out, "R_session.txt"))
results <- data.frame()
status <- data.frame()
for (fold in 0:4) {
  d <- read.csv(file.path(out, paste0("private_fold", fold, ".csv")), check.names = FALSE)
  train <- d$partition == "train"
  views <- lapply(c("semantic_", "grain_"), function(prefix) {
    t(as.matrix(d[,grep(paste0("^", prefix), names(d)), drop = FALSE]))
  })
  for (trait in 0:4) {
    set.seed(731 + fold * 10 + trait)
    y <- d[[paste0("y_", trait)]]
    ym <- mean(y[train]); ys <- sd(y[train])
    start <- proc.time()[[3]]
    fitted <- tryCatch({
      fit <- sJIVE.converge(lapply(views, function(x) x[,train]), (y[train]-ym)/ys,
        rankJ = 2, rankA = c(2,2), eta = 0.5, max.iter = 1000,
        threshold = 0.001, center.scale = FALSE, reduce.dim = FALSE,
        show.message = FALSE)
      pred <- sJIVE.predict(fit, lapply(views, function(x) x[,!train]),
                            threshold = 1e-5, max.iter = 500)
      list(fit = fit, pred = pred)
    }, error = function(e) e)
    if (inherits(fitted, "error")) {
      status <- rbind(status, data.frame(fold, trait, success = FALSE,
        fit_iterations = NA, predict_iterations = NA, limit_hit = NA,
        seconds = proc.time()[[3]]-start, message = conditionMessage(fitted)))
    } else {
      p <- fitted$pred
      shared <- as.numeric(fitted$fit$theta1 %*% p$Sj) * ys
      semantic <- as.numeric(fitted$fit$theta2[[1]] %*% p$Si[[1]]) * ys
      grain <- as.numeric(fitted$fit$theta2[[2]] %*% p$Si[[2]]) * ys
      total <- as.numeric(p$Ypred) * ys + ym
      stopifnot(max(abs(total-(ym+shared+semantic+grain))) < 1e-8)
      results <- rbind(results, data.frame(row_id = d$row_id[!train], fold, trait,
        prediction = total, intercept = ym, shared, semantic, grain))
      limit <- fitted$fit$iterations >= 1000 || p$iteration >= 500
      status <- rbind(status, data.frame(fold, trait, success = TRUE,
        fit_iterations = fitted$fit$iterations, predict_iterations = p$iteration,
        limit_hit = limit, seconds = proc.time()[[3]]-start, message = ""))
      saveRDS(fitted, file.path(out, paste0("private_sjive_fold",fold,"_trait",trait,".rds")))
    }
    write.csv(results, file.path(out, "private_sjive_predictions.csv"), row.names = FALSE)
    write.csv(status, file.path(out, "sjive_status.csv"), row.names = FALSE)
    print(tail(status, 1))
  }
}
