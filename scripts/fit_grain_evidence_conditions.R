#!/usr/bin/env Rscript
# Observed-comment evidence availability, not latent choice or personality.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("input.csv output-directory required")
suppressPackageStartupMessages(library(lme4))
d <- read.csv(args[1], stringsAsFactors=FALSE)
out <- args[2]
stopifnot(all(d$matched %in% c(0,1)), all(is.finite(d$log_words_z)))
d$author <- factor(d$author)
d$condition <- factor(d$condition)
messages <- character()
fit <- withCallingHandlers(
  glmer(matched ~ log_words_z + (1|author) + (1|condition), data=d,
    family=binomial(), nAGQ=1,
    control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=20000))),
  warning=function(w) {messages <<- c(messages, conditionMessage(w)); invokeRestart("muffleWarning")}
)
fixed <- as.data.frame(coef(summary(fit)))
fixed$term <- rownames(fixed)
write.csv(fixed, file.path(out,"fixed_effects.csv"), row.names=FALSE)
write.csv(as.data.frame(VarCorr(fit)), file.path(out,"variance_components.csv"), row.names=FALSE)
baseline <- glm(matched ~ log_words_z, family=binomial(), data=d)
write.csv(data.frame(
  model=c("length_glm","author_condition_glmm"),
  AIC=c(AIC(baseline), AIC(fit)),
  logLik=c(as.numeric(logLik(baseline)),as.numeric(logLik(fit)))
),file.path(out,"descriptive_fit.csv"),row.names=FALSE)
write.csv(data.frame(
  singular=isSingular(fit), optimizer_code=fit@optinfo$conv$opt,
  lme4_convergence_messages=paste(unlist(fit@optinfo$conv$lme4), collapse="; "),
  warnings=paste(messages,collapse="; "),
  lme4_version=as.character(packageVersion("lme4")),
  observations=nrow(d), authors=nlevels(d$author), conditions=nlevels(d$condition)
),file.path(out,"diagnostics.csv"),row.names=FALSE)
saveRDS(fit,file.path(out,"private_availability_model.rds"))
