# Independent base-R least-squares/exhaustive-search and weakest-link oracle.
# Input: data.csv; command arguments: depth, minimum leaf size, fit x (0/1).
# No Rust code or native GUIDE output is used to compute these quantities.
args <- commandArgs(trailingOnly=TRUE)
depth_limit <- as.integer(args[1]); minimum <- as.integer(args[2]); fit_x <- args[3] == "1"
d <- read.csv("data.csv"); n <- nrow(d)
w <- if ("weight" %in% names(d)) d$weight else rep(1,n)
fit_columns <- if (fit_x) c("x","z") else "z"
near <- function(a,b) is.finite(a) && is.finite(b) && abs(a-b) <= 64*.Machine$double.eps*max(abs(a),abs(b))

fit_node <- function(ids) {
    x <- matrix(1,length(ids),1); means <- numeric(); columns <- character()
    for (name in fit_columns) {
        v <- d[[name]][ids]; seen <- !is.na(v)
        if (!any(seen)) next
        m <- weighted.mean(v[seen],w[ids][seen]); v[!seen] <- m
        means[name] <- m
        if (length(unique(v)) > 1) { x <- cbind(x,v); columns <- c(columns,name) }
        if (any(!seen) && any(seen)) {
            x <- cbind(x,as.numeric(!seen)); columns <- c(columns,paste0(name,".NA"))
        }
    }
    fit <- lm.wfit(x,d$y[ids],w[ids]); beta <- fit$coefficients; beta[is.na(beta)] <- 0
    list(beta=beta,columns=columns,means=means,risk=sum(w[ids]*fit$residuals^2))
}
predict_fit <- function(fit,row) {
    values <- vapply(fit$columns,function(name) {
        if (endsWith(name,".NA")) return(as.numeric(is.na(d[[sub(".NA$","",name)]][row])))
        value <- d[[name]][row]
        if (is.na(value)) fit$means[name] else value
    },numeric(1))
    sum(fit$beta*c(1,values))
}
grow <- function(ids,depth=0,id=1) {
    model <- fit_node(ids); node <- list(id=id,fit=model,left=NULL,right=NULL,cut=NA_real_)
    if (depth >= depth_limit || length(ids) < 2*minimum) return(node)
    values <- sort(unique(d$x[ids])); cuts <- (head(values,-1)+tail(values,-1))/2
    best <- model$risk; partition <- NULL
    for (cut in cuts) {
        left <- ids[d$x[ids]<=cut]; right <- ids[d$x[ids]>cut]
        if (length(left)<minimum || length(right)<minimum) next
        risk <- fit_node(left)$risk + fit_node(right)$risk
        if (risk < best && !near(risk,best)) {
            best <- risk; partition <- list(left,right,cut)
        }
    }
    if (!is.null(partition)) {
        node$cut <- partition[[3]]
        node$left <- grow(partition[[1]],depth+1,2*id)
        node$right <- grow(partition[[2]],depth+1,2*id+1)
    }
    node
}
terminal <- function(node) is.null(node$left)
leaves <- function(node) if (terminal(node)) 1 else leaves(node$left)+leaves(node$right)
leaf_risk <- function(node) if (terminal(node)) node$fit$risk else leaf_risk(node$left)+leaf_risk(node$right)
candidate <- function(node,total,flawed=FALSE) {
    best <- list(alpha=Inf,id=0)
    visit <- function(branch) {
        if (terminal(branch)) return(invisible(NULL))
        alpha <- max(0,(branch$fit$risk-leaf_risk(branch))/(leaves(branch)-1)/total)
        if (alpha < best$alpha && !near(alpha,best$alpha) && (!flawed || branch$id>best$id))
            best <<- list(alpha=alpha,id=branch$id)
        if (flawed) { visit(branch$right);visit(branch$left) }
        else { visit(branch$left);visit(branch$right) }
    }
    visit(node);best
}
collapse <- function(node,id) {
    if (node$id == id) { node$left <- NULL;node$right <- NULL;node$cut <- NA_real_;return(node) }
    if (!terminal(node)) { node$left <- collapse(node$left,id);node$right <- collapse(node$right,id) }
    node
}
path <- function(tree,total) {
    stages <- list(list(alpha=0,tree=tree,pruned_node=0))
    while (!terminal(tree)) {
        next_node <- candidate(tree,total)
        stopifnot(next_node$id > 0)
        tree <- collapse(tree,next_node$id)
        stages[[length(stages)+1]] <- list(alpha=next_node$alpha,tree=tree,pruned_node=next_node$id)
    }
    stages
}
prune_at <- function(tree,total,alpha,flawed=FALSE) {
    while (!terminal(tree)) {
        next_node <- candidate(tree,total,flawed)
        stopifnot(next_node$id > 0)
        if (next_node$alpha > alpha) break
        tree <- collapse(tree,next_node$id)
    }
    tree
}
predict_node <- function(tree,row) {
    while (!terminal(tree)) tree <- if(d$x[row]<=tree$cut) tree$left else tree$right
    c(node=tree$id,prediction=predict_fit(tree$fit,row))
}
full <- grow(seq_len(n)); stages <- path(full,sum(w))
eval_alpha <- vapply(seq_along(stages),function(i) {
    if (i == length(stages)) Inf else if (i == 1) 0 else sqrt(stages[[i]]$alpha*stages[[i+1]]$alpha)
},numeric(1))
errors <- matrix(0,n,length(stages)); flawed_errors <- errors
held_predictions <- errors; flawed_predictions <- errors
for (held in seq_len(n)) {
    ids <- setdiff(seq_len(n),held); tree <- grow(ids)
    for (j in seq_along(stages)) {
        correct <- if(j == 1) tree else prune_at(tree,sum(w[ids]),eval_alpha[j])
        flawed <- if(j == 1) tree else prune_at(tree,sum(w[ids]),eval_alpha[j],TRUE)
        held_predictions[held,j] <- predict_node(correct,held)["prediction"]
        flawed_predictions[held,j] <- predict_node(flawed,held)["prediction"]
        errors[held,j] <- n*w[held]/sum(w)*(d$y[held]-held_predictions[held,j])^2
        flawed_errors[held,j] <- n*w[held]/sum(w)*(d$y[held]-flawed_predictions[held,j])^2
    }
}
means <- colMeans(errors); ses <- sqrt(colSums(sweep(errors,2,means)^2))/n
flawed_means <- colMeans(flawed_errors); flawed_ses <- sqrt(colSums(sweep(flawed_errors,2,flawed_means)^2))/n
summary <- data.frame(leaves=vapply(stages,function(s)leaves(s$tree),numeric(1)),
    alpha=vapply(stages,function(s)s$alpha,numeric(1)),
    pruned_node=vapply(stages,function(s)s$pruned_node,numeric(1)),
    mean_loss=means,standard_error=ses,native_scan_mean=flawed_means,native_scan_se=flawed_ses)
write.csv(summary,"oracle-scores.csv",row.names=FALSE)
for (i in seq_along(stages)) {
    predictions <- t(vapply(seq_len(n),function(row)predict_node(stages[[i]]$tree,row),numeric(2)))
    write.csv(predictions,paste0("oracle-tree-",i,".csv"),row.names=FALSE)
}
write.csv(held_predictions,"oracle-held-predictions.csv",row.names=FALSE)
write.csv(flawed_predictions,"oracle-native-scan-predictions.csv",row.names=FALSE)
writeLines(R.version.string,"oracle-version.txt")
