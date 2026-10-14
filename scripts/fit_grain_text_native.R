#!/usr/bin/env Rscript
# Native text training on the existing development cohort; one held-out fold.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 2)
out <- normalizePath(args[1])
.libPaths(c(normalizePath(args[2]),.libPaths()))
setwd(out)
library(text)
library(tibble)
library(workflows)
x <- as_tibble(read.csv("private_semantic.csv",check.names=FALSE))
names(x) <- paste0("Dim",seq_len(ncol(x)))
y <- read.csv("private_labels.csv",check.names=FALSE)
folds <- read.csv("private_folds.csv")$fold
train <- folds != 0
stopifnot(sum(train)==504, sum(!train)==120, nrow(x)==nrow(y))
held_x <- x[!train,]
held_x$id_nr <- which(!train)
summary <- list()
for (trait in names(y)) {
  model <- textTrain(x=x[train,], y=y[train,trait], force_train_method="regression",
    cv_method="cv_folds", outside_folds=3, inside_folds=2,
    outside_strata=FALSE, inside_strata=FALSE, preprocess_PCA=64,
    penalty=c(.001,.01,.1,1), mixture=0, multi_cores=1, seed=731,
    save_output="all", model_description="GRAIN native package reference; source-only selection; original official fold0 held out")
  pred <- predict(model$final_model, new_data=held_x)$.pred
  stopifnot(length(pred)==sum(!train),all(is.finite(pred)))
  write.csv(data.frame(row_index=which(!train)-1, y=y[!train,trait],prediction=pred),
            paste0("private_predictions_",trait,".csv"),row.names=FALSE)
  saveRDS(model,paste0("native_model_",trait,".rds"))
  capture.output(print(model$model_description),file=paste0("model_description_",trait,".txt"))
  summary[[trait]] <- data.frame(trait=trait,n=sum(!train),r=cor(pred,y[!train,trait]),
    source_nested_r=unname(model$results$estimate),
    final_penalty=workflows::extract_fit_parsnip(model$final_model)$spec$args$penalty)
  write.csv(do.call(rbind,summary),"metrics.csv",row.names=FALSE)
  cat("COMPLETED_TRAIT",trait,"\n")
}
capture.output(sessionInfo(),file="sessionInfo.txt")
writeLines("COMPLETE","status.txt")
