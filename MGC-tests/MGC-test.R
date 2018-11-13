#!/usr/bin/env Rscript
library(mgc)
library(reshape2)
library(ggplot2)
plot_sim_func <- function(X, Y, Xf, Yf, name, geom='line') {
  if (!is.null(dim(Y))) {
    Y <- Y[, 1]
    Yf <- Yf[, 1]
  }
  if (geom == 'points') {
    funcgeom <- geom_point
  } else {
    funcgeom <- geom_line
  }
  data <- data.frame(x1=X[,1], y=Y)
  data_func <- data.frame(x1=Xf[,1], y=Yf)
  ggplot(data, aes(x=x1, y=y)) +
    funcgeom(data=data_func, aes(x=x1, y=y), color='red', size=3) +
    geom_point() +
    xlab("x") +
    ylab("y") +
    ggtitle(name) +
    theme_bw()
}
plot_mtx <- function(Dx, main.title="Local Correlation Map", xlab.title="# X Neighbors", ylab.title="# Y Neighbors") {
  data <- melt(Dx)
  ggplot(data, aes(x=Var1, y=Var2, fill=value)) +
    geom_tile() +
    scale_fill_gradientn(name="l-corr",
                         colours=c("#f2f0f7", "#cbc9e2", "#9e9ac8", "#6a51a3"),
                         limits=c(min(Dx), max(Dx))) +
    xlab(xlab.title) +
    ylab(ylab.title) +
    theme_bw() +
    ggtitle(main.title)
}

set.seed(12345)
dat <- read.csv("KHop_Data.csv", header = TRUE)
dat <- array(as.numeric(unlist(dat)), dim=c(97,97))
dat <- dat[,-1]
maxval <- 0
Xmax <- 0
Ymax <- 0
X <- 1

zeromat <- matrix(1:96, nrow = 96, ncol=6)
mgctests <- array(0L, dim(zeromat))
for (Y in 2:97) {
    Xdat <- dat[X,]
    YDat <- dat[Y,]
    res <- mgc.test(Xdat,YDat, rep=20)
    val <- res$statMGC
    p_val <- res$pMGC
    scale <- res$optimalScale
    print(scale)
    plot_mtx(res$localCorr, main.title="Local Correlation Map")
    if ( val > maxval){
        maxval <- val
        Xmax <- X
        Ymax <- Y
    }
    mgctests[Y-1,1] = val
    mgctests[Y-1,2] = X
    mgctests[Y-1,3] = Y
    mgctests[Y-1,4] = p_val
    mgctests[Y-1,5] = scale$x
    mgctests[Y-1,6] = scale$y
}
#Xdat = dat[1,]
#YDat = dat[2,]
#res <- mgc.test(Xdat, YDat, rep=20)
#maxval = res$statMGC

print(mgctests)
write.csv(mgctests, file = 'mgc-test-stats.csv')
