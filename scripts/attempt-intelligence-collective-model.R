# Run only the author helper and chosen configuration from the locked checkout.
# Usage: Rscript attempt-intelligence-collective-model.R CHECKOUT three|eight OUTPUT
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=3 || !args[2] %in% c('three','eight')) stop('expected CHECKOUT three|eight OUTPUT')
checkout <- normalizePath(args[1], mustWork=TRUE)
outFile <- args[3]
if(!grepl('^/',outFile)) stop('OUTPUT must be an absolute path')
library(metaSEM)
library(OpenMx)
setwd(checkout)
mainDataFile <- 'DATA/full.csv'
inFile <- if(args[2]=='three') 'CONFIGURATION/conf_AllStudies3Tasks_new.R' else 'CONFIGURATION/conf_AllStudies8Tasks_new.R'
runRerun <- FALSE
# Presentation output is unnecessary for the fit; this replaces only texreg display.
screenreg <- function(...) invisible(NULL)
source('R_CODE/helper_fitMetaAnalysis.R')
cat('\n=== Session ===\n'); print(sessionInfo())
cat('\n=== Stage-one optimizer status ===\n'); print(random1Diag$mx.fit$output$status)
cat('\n=== Stage-one summary ===\n'); print(summary(random1Diag))
cat('\n=== Stage-two summary ===\n'); print(summary(random2Eight))
write.csv(summary(random2Eight)$coef, paste0(outFile,'.coefficients.csv'))
write.csv(summary(random2Eight)$stat, paste0(outFile,'.fit.csv'))
