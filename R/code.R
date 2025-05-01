#### OjaQuantile ----
#' Oja Quantile
#'
#' Iterative computation of the Oja quantiles for multivariate data:
#' \itemize{
#' \item A stochastic gradient descent algorithm, and 
#' \item an exact optimization algorithm using function \link[stats]{optim}.
#' }
#' 
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' @param u A single unit vector of dimension \code{d}, or a numerical matrix
#' of dimension \code{nq}-times-\code{d} of \code{nq} unit vectors in dimension 
#' \code{d}. Each row of the matrix corresponds to one unit direction in which
#' the Oja quantile should be calclulated.
#' @param alpha Magnitudes \code{alpha} of the Oja quantiles to be calculated.
#' A single value in the interval \code{[0,1]} if \code{u} is a 
#' single vector, or a numerical vector of values inside \code{[0,1]} of length
#' \code{nq} if \code{u} is a matrix with \code{nq} rows. The \code{i}-th 
#' element of \code{alpha} corresponds to the \code{i}-th row of \code{u}.
#' @param method An idicator of the method used to compute the Oja quantiles. 
#' Can be either \code{method="SGD"} for stochastic gradient descent (default), 
#' or \code{method="optim"} for a numerical optimization method using function
#' \link[stats]{optim}. 
#' 
#' \emph{All the further arguments to this function apply only in the case} 
#' \code{method="SGD"}.
#' 
#' @param averaging The averaging method used for the final estimates using the 
#' stochastic gradient method. By default set to \code{averaging="all"}, meaning
#' that the resulting estimate is a weighted average of all the \code{B} 
#' updates. Other possible values are \code{averaging="suffix"}, where the 
#' weighted average of only the last \code{B/4} updates is considered, or
#' \code{averaging="none"}, where only the final \code{B}th update is taken, 
#' without any averaging.
#'  
#' @param B Number of iterations of the stochastic gradient descent method. A 
#' positive integer. By default taken to be \code{B=1e5}.
#' @param batch Size of mini-batches for the stochastic gradient descent 
#' algorithm. In each iteration of the algorithm, \code{batch} of 
#' randomly selected \code{d}-tuples of points from \code{X} are taken, and 
#' the gradients of the summands of the objective function \code{f} associated 
#' with these \code{batch} subsets are used to approximate the gradient 
#' of \code{f}. By default set to \code{NULL}, in which case 
#' \code{batch} is taken as \code{max(1,round(min(100,sqrt(n))))}, where 
#' \code{n} is the sample size.
#' @param gamma The initial step value for the stochastic gradient descent. By 
#' default set to \code{NULL}, which means that \code{gamma} is evaluated as
#' \code{1/4} of the maximum range of the dataset \code{X} among its \code{d}
#' coordinates.
#' @param x0 The initial value for the iterative procedure. By default set to 
#' \code{NULL}, meaning that \code{x0} is taken to be the sample mean of 
#' \code{X}.
#' @param trck Indicator of whether as an output of the procedure, also the
#' whole history of \code{B} iterated solutions should be given. 
#' By default set to \code{FALSE}.
#' 
#' @note The algorithm \code{method="optim"} typically gives more accurate 
#' results. On the other hand, the speed of the two algorithms differs 
#' drastically:
#' \itemize{
#' \item The algorithm using \code{method="SGD"} is very fast, and can be used 
#' also for higher values of \code{n} and \code{d}. 
#' \item Using \code{method="optim"} involves the exact evaluation of the 
#' complete objective function of an Oja quantile (see function \link{objf}), 
#' which is of order \code{O(n^d)}. Thus, running with \code{method="optim"} and
#' higher \code{d} or \code{n} is extremely slow. 
#' }
#'
#' @return A list with the following components:
#' \itemize{
#' \item \code{q}: A matrix of the approximate Oja quantiles of size 
#' \code{nq}-times-\code{d}, one row per one row of \code{u} and an element of 
#' \code{alpha}. 
#' \item \code{qtrck}: If \code{trck=FALSE}, an array of size 
#' \code{nq}-times-\code{d}-times-\code{1} which contains the same matrix as in
#' \code{q}. For \code{method="SGD"} and \code{trck=TRUE}, an array of size 
#' \code{nq}-times-\code{d}-times-\code{B}, where in the last dimension we have
#' the whole history of the \code{B} iterations of the approximate procedure for 
#' the Oja quantiles.
#' }
#'
#' @examples
#' n = 100
#' d = 2
#' X = matrix(rnorm(n*d),ncol=d)
#' 
#' alpha = 0.7
#' u = -c(1,rep(0,d-1))
#' 
#' # Computation using stochastic gradient descent
#' resSGD = OjaQuantile(X, u, alpha)
#' 
#' # Computation using function optim
#' resoptim = OjaQuantile(X, u, alpha, method="optim")
#' 
#' plot(X,pch=16,ann=FALSE)
#' points(resSGD$q[,1],resSGD$q[,2],col="orange",pch=16,cex=2)
#' points(resoptim$q[,1],resoptim$q[,2],col="violet",pch=1,cex=3,lwd=3)
#' 
#' # Multiple quantiles
#' nq = 10
#' alpha = rep(0.7,nq)
#' theta = seq(-pi,pi,length=nq)
#' u = cbind(cos(theta),sin(theta))
#' 
#' # Computation using stochastic gradient descent
#' resSGD = OjaQuantile(X, u, alpha)
#' 
#' # Computation using function optim
#' resoptim = OjaQuantile(X, u, alpha, method="optim")
#' 
#' plot(X,pch=16,ann=FALSE)
#' points(resSGD$q[,1],resSGD$q[,2],col="orange",pch=16,cex=2)
#' points(resoptim$q[,1],resoptim$q[,2],col="violet",pch=1,cex=3,lwd=3)
#' 
#' # Multiple quantiles including tracing the algorithm
#' resSGD = OjaQuantile(X, u, alpha, B=100, trck=TRUE)
#' plot(X,pch=16,ann=FALSE)
#' points(resSGD$q[,1],resSGD$q[,2],col="orange",pch=16,cex=2)
#' for(i in 1:nq) 
#' lines(resSGD$qtrck[i,1,],resSGD$qtrck[i,2,],col="orange",lwd=2)

OjaQuantile = function(X, u, alpha, 
                       method="SGD",
                       averaging="all",
                       B=1e5, batch=NULL, 
                       gamma=NULL, x0=NULL,
                       trck=FALSE){
  
  n = nrow(X)
  d = ncol(X)
  method = match.arg(method,c("SGD","optim"))
  averaging = match.arg(averaging,c("all","suffix","none"))
  if(averaging!="none") trck=TRUE
  if(is.matrix(u)) u = t(u)
  if(!is.matrix(u)) u = matrix(u,ncol=1)
  nqs = ncol(u)
  if(length(alpha)==1) alpha = rep(alpha,nqs)
  if(length(alpha)!=nqs) stop("The number of rows in u and the length of alpha must equal.")
  
  if(method=="SGD"){
    if(is.null(gamma)){
      gamma0 = apply(X,2,range)
      gamma = max(gamma0[2,]-gamma0[1,])/4
    }
    if(is.null(x0)) x0 = colMeans(X)
    if(is.null(batch)) batch = max(1,round(min(100,sqrt(n))))
    x0 = matrix(x0,nrow=d,ncol=nqs)
    
    if(!trck){
      res = OjaSGD(t(X), u, alpha, B, batch, 
                   gamma, x0, nrow(X), ncol(X), nqs, trck)
      return(list(q = t(res$x0), 
                  qtrck = array(t(res$x0),dim = c(nqs,ncol(X),1)),
                  gamma = gamma
                  ))
    }
    if(trck){
      res = OjaSGD(t(X), u, alpha, B, batch, 
                   gamma, x0, nrow(X), ncol(X), nqs, trck)
      if(averaging=="none") q = t(res$x0)
      if(averaging=="all"){
        q = matrix(NA,nrow=nqs,ncol=ncol(X))
        qtrck = aperm(res$x0trck,c(2,1,3))
        gammas = gamma/sqrt(1:B)
          for(i in 1:(dim(qtrck)[1])){
            if(d>1){
              Q2 = t(qtrck[i,,])
              q[i,] = colSums(Q2*gammas)/sum(gammas)
            }
            if(d==1){
              Q2 = qtrck[i,,]
              q[i,] = sum(Q2*gammas)/sum(gammas)  
            }
          }
      }
      if(averaging=="suffix"){
        q = matrix(NA,nrow=nqs,ncol=ncol(X))
        qtrck = aperm(res$x0trck,c(2,1,3))
        gammas = gamma/sqrt(1:B)
        for(i in 1:(dim(qtrck)[1])){
          if(d>1){
            Q2 = t(qtrck[i,,])
            q[i,] = colSums(Q2[floor(3*B/4):B,]*
                            gammas[floor(3*B/4):B])/
            sum(gammas[floor(3*B/4):B])
          }
          if(d==1){
            Q2 = qtrck[i,,]
            q[i,] = sum(Q2[floor(3*B/4):B]*
                              gammas[floor(3*B/4):B])/
              sum(gammas[floor(3*B/4):B])            
          }
        }        
      }
      return(list(q = q, 
                  qtrck = aperm(res$x0trck,c(2,1,3)),
                  gamma = gamma
                  ))
    }
  }
  if(method=="optim"){
    d = ncol(X)
    if(d>5) stop("The exact computation works only in dimension d<=5.")
    res = matrix(nrow=nqs, ncol=d)
    for(i in 1:nqs){
      if(d>1) res[i,] = 
          optim(c(colMeans(X)), function(x) objf(x,X,u[,i],alpha[i]))$par
      if(d==1) res[i,] = 
          optim(c(colMeans(X)), function(x) objf(x,X,u[,i],alpha[i]), 
                method="Brent", lower = min(X)-1/2, upper=max(X)+1/2)$par
    }
    return(list(q = res, qtrck = array(res, dim = c(nqs,ncol(X),1))))
  }
}

#### Objective function ----
#' Objective Function for the Oja Quantiles
#'
#' The complete objective function to be minimized for computing the Oja 
#' quantiles for multivariate data. 
#'
#' @param mu Arguments at which the objective function is to be evaluated. 
#' A numerical vector of dimension \code{d}, or a matrix of dimension 
#' \code{m}-times-\code{d}, each row per one argument of the objective function.
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' @param u A single unit vector of dimension \code{d}. Corresponds to the unit 
#' direction in which the Oja quantile should be calclulated.
#' @param alpha The magnitude \code{alpha} of the Oja quantile to be 
#' calculated. A single value in the interval \code{[0,1]}.
#'
#' @note Since this is an exact evaluation of the objective function, running
#' this function with higher \code{d} (that is, \code{d>2}) or \code{m} 
#' (\code{m>500}) can be extremely slow.
#'
#' @return A numerical vector of length \code{m} of the values of the objective
#' function. Each value corresponds to one row of \code{mu}.
#'
#' @examples
#' n = 100
#' d = 2
#' X = matrix(rnorm(n*d),ncol=d)
#' 
#' alpha = 0.7
#' u = -c(1,rep(0,d-1))
#' 
#' resSGD = OjaQuantile(X, u, alpha)
#' objf(resSGD$q, X, u, alpha)
#' 
#' neval = 101
#' xgr = seq(min(X[,1])-0.5,max(X[,1])+0.5,length=neval)
#' ygr = seq(min(X[,2])-0.5,max(X[,2])+0.5,length=neval)
#' mu = as.matrix(expand.grid(xgr,ygr))
#' objmat = matrix(objf(mu,X,u,alpha),ncol=neval)
#' 
#' contour(xgr,ygr,objmat,ann=FALSE)
#' points(X,pch=16)
#' points(resSGD$q[,1],resSGD$q[,2],col="orange",pch=16,cex=2)

objf = function(mu, X, u, alpha){
  n = nrow(X)
  d = ncol(X)
  if(is.matrix(mu)) mu = t(mu)
  if(!is.matrix(mu)){
    if(length(mu)==d) mu = matrix(mu,ncol=1)
    if(d==1) mu = matrix(mu,nrow=1)
  }
  if(nrow(mu)!=d) stop("Dimensions of mu and X do not coincide.")
  if(d>5) stop("The exact computation works only in dimension d<=5.")  
  return(objfC(mu, t(X), u, alpha, n, d, ncol(mu))/
           choose(n,d))
}

objf_rank = function(mu, X, eval){
  n = nrow(X)
  d = ncol(X)
  if(is.matrix(mu)) mu = t(mu)
  if(!is.matrix(mu)){
    if(length(mu)==d) mu = matrix(mu,ncol=1)
    if(d==1) mu = matrix(mu,nrow=1)
  }
  if(nrow(mu)!=d) stop("Dimensions of mu and X do not coincide.")
  if(d>5) stop("The exact computation works only in dimension d<=5.")  
  if(d==2){
    theta = seq(-pi,pi,length=eval)
    V = cbind(cos(theta),sin(theta))
  } else {
    V = matrix(rnorm(eval*d),ncol=d)
    V = t(apply(V,1,function(x) x/c(sqrt(crossprod(x)))))
  }
  res = gfunC(V,t(X),mu,n,d,eval)
  nm = res$num
  dn = c(res$den)
  #
  ln = sqrt(c(crossprod(nm)))
  nm = nm/ln
  dn = dn/ln
  of = V%*%nm/dn
  return(list(domain = V, vals = of, numerator = nm))
}
