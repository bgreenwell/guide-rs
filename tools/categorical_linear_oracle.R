# Independent weighted least squares on frozen terminal partitions, plus an
# exhaustive four-level split check. No generated prediction code is used.
args <- commandArgs(trailingOnly=TRUE)
mode <- args[1]
categorical <- if(args[2]=="-") character() else strsplit(args[2],",",fixed=TRUE)[[1]]
numeric <- if(args[3]=="-") character() else strsplit(args[3],",",fixed=TRUE)[[1]]
cv <- args[4]=="cv"
d <- read.csv("data.csv",colClasses="character")
for(name in c("y","weight",numeric)) if(name %in% names(d)) d[[name]] <- as.numeric(d[[name]])
train <- as.logical(scan("training.txt",quiet=TRUE));leaf <- scan("leaves.txt",quiet=TRUE)
n <- nrow(d);w <- if("weight" %in% names(d)) d$weight else rep(1,n)
# Preserve numeric-looking labels; all these DSC fixtures use the NA token.
for(name in categorical) d[[name]][is.na(d[[name]])] <- "NA"
levels <- lapply(d[categorical],function(v)sort(unique(v),method="radix"))
flags <- numeric[vapply(numeric,function(name)any(is.na(d[[name]])),logical(1))]
p <- length(numeric)+sum(lengths(levels)-1)
minimum <- if(length(flags)==0) p+7 else p+length(flags)+3
estimate <- function(ids,rows,truncate=TRUE) {
    if(length(ids)<minimum) return(rep(weighted.mean(d$y[ids],w[ids]),length(rows)))
    x <- matrix(1,length(ids),1);future <- matrix(1,length(rows),1)
    for(name in categorical) {
        local <- levels[[name]][levels[[name]] %in% d[[name]][ids]]
        reference <- if(levels[[name]][1] %in% local) levels[[name]][1] else tail(local,1)
        for(level in setdiff(local,reference)) {
            x <- cbind(x,as.numeric(d[[name]][ids]==level))
            future <- cbind(future,as.numeric(d[[name]][rows]==level))
        }
    }
    for(name in numeric) {
        v <- d[[name]][ids];q <- d[[name]][rows];observed <- !is.na(v)
        m <- if(any(observed))weighted.mean(v[observed],w[ids][observed]) else 0
        v[is.na(v)] <- m;q[is.na(q)] <- m
        x <- cbind(x,v);future <- cbind(future,q)
        if(any(!observed) && any(observed)) {
            x <- cbind(x,as.numeric(!observed));future <- cbind(future,as.numeric(is.na(d[[name]][rows])))
        }
    }
    beta <- lm.wfit(x,d$y[ids],w[ids])$coefficients;beta[is.na(beta)] <- 0
    prediction <- as.vector(future %*% beta)
    if(truncate && mode!="none") {
        bounds <- if(mode=="global_range")range(d$y[train]) else range(d$y[ids])
        if(mode=="node_range_expanded")bounds <- bounds+c(-1,1)*diff(bounds)*0.1
        prediction <- pmin(bounds[2],pmax(bounds[1],prediction))
    }
    prediction
}
prediction <- rep(NA_real_,n)
for(node in unique(leaf))prediction[leaf==node] <- estimate(which(train & leaf==node),which(leaf==node))
writeLines(format(prediction,digits=17),"oracle-predictions.txt")
if(cv) {
    errors <- matrix(0,sum(train),2)
    held_predictions <- matrix(0,sum(train),2)
    for(held in which(train)) {
        ids <- which(train & seq_len(n)!=held)
        local <- ids[leaf[ids]==leaf[held]]
        if(!"g" %in% names(d)) {
            # The orientation CV control has one categorical split predictor.
            # Independently enumerate every partition in every training fold.
            observed <- sort(unique(d$group[ids]),method="radix")
            best <- Inf
            for(mask in 0:(2^(length(observed)-1)-2)) {
                left <- c(observed[1],observed[-1][bitwAnd(mask,bitwShiftL(1,seq_len(length(observed)-1)-1))!=0])
                a <- ids[d$group[ids] %in% left];b <- setdiff(ids,a)
                risk <- sum(w[a]*(d$y[a]-estimate(a,a,FALSE))^2)+sum(w[b]*(d$y[b]-estimate(b,b,FALSE))^2)
                if(risk<best) {best<-risk;local<-if(d$group[held] %in% left)a else b}
            }
        }
        held_predictions[held,1] <- estimate(local,held)
        held_predictions[held,2] <- estimate(ids,held)
        errors[held,] <- n*w[held]/sum(w[train])*(d$y[held]-held_predictions[held,])^2
    }
    means <- colMeans(errors);ses <- sqrt(colSums(sweep(errors,2,means)^2))/sum(train)
    writeLines(format(c(means,ses),digits=17),"oracle-cv.txt")
    write.table(held_predictions,"oracle-heldout.txt",row.names=FALSE,col.names=FALSE)
}
if(length(categorical)==1 && length(levels[[1]])==4 && !"NA" %in% levels[[1]]) {
    risk <- sapply(1:7,function(mask) {
        left <- levels[[1]][1:3][bitwAnd(mask,bitwShiftL(1,0:2))!=0]
        membership <- d[[categorical]][which(train)] %in% left
        ids <- which(train)
        a <- ids[membership];b <- ids[!membership]
        sum(w[a]*(d$y[a]-estimate(a,a,FALSE))^2)+sum(w[b]*(d$y[b]-estimate(b,b,FALSE))^2)
    })
    writeLines(format(risk,digits=17),"oracle-subset-risks.txt")
}
