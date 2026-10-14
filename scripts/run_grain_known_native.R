#!/usr/bin/env Rscript
# Run installed text and the unmodified author MEM function on frozen inputs.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3)
out <- normalizePath(args[1])
.libPaths(c(normalizePath(args[2]), .libPaths()))
source_file <- normalizePath(args[3])
setwd(out)
library(text)
library(psych)
library(GPArotation)
library(MASS)

# Evaluate function definitions only, not the template's machine-specific I/O.
author <- new.env(parent=globalenv())
required <- c("kmo", "Bartlett.sphericity.test", "remove_zero_var", "PCAFunction")
for (expr in parse(source_file)) {
  if (is.call(expr) && as.character(expr[[1]]) %in% c("=", "<-") &&
      is.symbol(expr[[2]]) && as.character(expr[[2]]) %in% required &&
      is.call(expr[[3]]) && identical(expr[[3]][[1]], as.name("function"))) {
    eval(expr, envir=author)
  }
}
stopifnot(all(vapply(required, exists, logical(1), envir=author, inherits=FALSE)))
train <- read.csv("mem_train.csv", check.names=FALSE)
held <- as.matrix(read.csv("mem_held.csv", check.names=FALSE))
k <- as.integer(readLines("mem_rank.txt"))
mem <- author$PCAFunction(train, k)
held_scores <- scale(held, center=colMeans(train), scale=apply(train,2,sd)) %*% mem$weights
write.csv(unclass(mem$loadings), "native_mem_loadings.csv", row.names=FALSE)
write.csv(mem$scores, "native_mem_train_scores.csv", row.names=FALSE)
write.csv(held_scores, "native_mem_held_scores.csv", row.names=FALSE)
saveRDS(mem, "native_mem_model.rds")

# Author-provided embedded example, no new encoder/model download.
data("word_embeddings_4", package="text")
data("Language_based_assessment_data_8", package="text")
x <- word_embeddings_4$texts$harmonytext
y <- Language_based_assessment_data_8$hilstotal
stopifnot(nrow(x) == length(y), all(is.finite(y)))
example <- textTrain(x=x, y=y, cv_method="cv_folds", outside_folds=3,
  inside_folds=2, outside_strata=FALSE, inside_strata=FALSE,
  preprocess_PCA=5, penalty=c(.1,1), mixture=0,
  multi_cores=1, seed=731, save_output="all",
  model_description="Author bundled demonstration; bounded grid; not published accuracy reproduction")
saveRDS(example, "native_text_example.rds")
test <- example$results
two_sided <- cor.test(example$predictions$predictions, example$predictions$y,
                      alternative="two.sided")
write.csv(data.frame(n=nrow(x), r=unname(test$estimate),
  package_p=test$p.value, package_alternative=test$alternative,
  two_sided_p=two_sided$p.value, ci_low=two_sided$conf.int[1],
  ci_high=two_sided$conf.int[2]), "native_text_example_results.csv", row.names=FALSE)
write.csv(example$predictions, "native_text_example_predictions.csv", row.names=FALSE)
capture.output(str(example, max.level=2), file="native_text_output_structure.txt")

# The default center=TRUE is not the raw cosine used for anchor retrieval.
sx <- read.csv("similarity_x.csv")
sy <- read.csv("similarity_y.csv")
names(sx) <- paste0("Dim", seq_len(ncol(sx)))
names(sy) <- names(sx)
write.csv(textSimilarity(x=sx, y=sy, center=FALSE, scale=FALSE),
          "native_text_raw_cosine.csv", row.names=FALSE)
capture.output(sessionInfo(), file="sessionInfo.txt")
write.csv(data.frame(package=rownames(installed.packages()),
                    version=installed.packages()[,"Version"]),
          "installed_versions.csv", row.names=FALSE)
writeLines("COMPLETE", "status.txt")
