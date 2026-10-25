# Independent weighted QR estimation and prediction limits. No prediction-code
# export is generated or executed. Inputs and terminal partitions are fixtures.
args <- commandArgs(trailingOnly=TRUE)
mode <- args[1]; fitting <- strsplit(args[2],",",fixed=TRUE)[[1]]
constant <- args[3] == "constant"; cv <- args[4] == "cv"
d <- read.csv("data.csv"); train <- as.logical(scan("training.txt",quiet=TRUE))
leaf <- scan("leaves.txt",quiet=TRUE);n <- nrow(d)
w <- if("weight" %in% names(d)) d$weight else rep(1,n)
global_range <- range(d$y[train])
flags <- fitting[vapply(fitting,function(name)any(is.na(d[[name]])),logical(1))]
estimate <- function(ids,rows) {
    if (constant) return(rep(weighted.mean(d$y[ids],w[ids]),length(rows)))
    x <- matrix(1,length(ids),1); future <- matrix(1,length(rows),1)
    for (name in fitting) {
        observed <- !is.na(d[[name]][ids]); m <- if(any(observed)) weighted.mean(d[[name]][ids][observed],w[ids][observed]) else 0
        observed_values <- d[[name]][ids];observed_values[!observed] <- m
        values <- d[[name]][rows];values[is.na(values)] <- m
        x <- cbind(x,observed_values);future <- cbind(future,values)
    }
    for (name in flags) {
        x <- cbind(x,as.numeric(is.na(d[[name]][ids])))
        future <- cbind(future,as.numeric(is.na(d[[name]][rows])))
    }
    beta <- lm.wfit(x,d$y[ids],w[ids])$coefficients;beta[is.na(beta)] <- 0
    prediction <- as.vector(future %*% beta)
    if (mode != "none") {
        bounds <- if(mode=="global_range")global_range else range(d$y[ids])
        if(mode=="node_range_expanded")bounds <- bounds+c(-1,1)*diff(bounds)*0.1
        prediction <- pmin(bounds[2],pmax(bounds[1],prediction))
    }
    prediction
}
prediction <- numeric(n)
for (node in unique(leaf)) prediction[leaf==node] <- estimate(which(train & leaf==node),which(leaf==node))
writeLines(format(prediction,digits=17),"oracle-predictions.txt")
writeLines(R.version.string,"oracle-version.txt")
if (cv) {
    errors <- matrix(0,sum(train),2)
    for (held in which(train)) {
        training <- which(train & seq_len(n)!=held)
        # These two CV fixtures have a binary split-only g predictor; both
        # classes remain after leaving out any row, fixing the stump partition.
        ids <- training[leaf[training]==leaf[held]]
        errors[held,1] <- n*w[held]/sum(w[train])*(d$y[held]-estimate(ids,held))^2
        errors[held,2] <- n*w[held]/sum(w[train])*(d$y[held]-estimate(training,held))^2
    }
    means <- colMeans(errors);ses <- sqrt(colSums(sweep(errors,2,means)^2))/sum(train)
    writeLines(format(c(means,ses),digits=17),"oracle-cv.txt")
}
