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
#' 
#' @param alpha Magnitudes \code{alpha} of the Oja quantiles to be calculated.
#' A single value in the interval \code{[0,1]} if \code{u} is a 
#' single vector, or a numerical vector of values inside \code{[0,1]} of length
#' \code{nq} if \code{u} is a matrix with \code{nq} rows. The \code{i}-th 
#' element of \code{alpha} corresponds to the \code{i}-th row of \code{u}.
#' 
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
#' positive integer. By default taken to be \code{B=1e4}.
#' 
#' @param whiten The pre-whitening method of the dataset to achieve better
#' numerical stability. Possible values are \code{"none"}, \code{"cov"}, 
#' \code{"var"}, or \code{"Tyler"}. Default is \code{"cov"}, which means that
#' whitening is performed using the sample
#' covariance matrix. This is the same as calling \code{whiten="var"}. With
#' \code{whiten=="none"} no pre-whitening is performed. For \code{"Tyler"},
#' the data matrix is whitened using the robust Tyler's scatter 
#' estimator, with a center at the spatial median. 
#' 
#' @param batch Size of mini-batches for the stochastic gradient descent 
#' algorithm. In each iteration of the algorithm, \code{batch} of 
#' randomly selected \code{d}-tuples of points from \code{X} are taken, and 
#' the gradients of the summands of the objective function \code{f} associated 
#' with these \code{batch} subsets are used to approximate the gradient 
#' of \code{f}. By default set to \code{NULL}, in which case 
#' \code{batch} is taken as \code{max(1,round(min(100,sqrt(n))))}, where 
#' \code{n} is the sample size.
#' 
#' @param gamma The initial step value for the stochastic gradient descent. By 
#' default set to \code{NULL}, which means that \code{gamma} is evaluated as
#' \code{1/4} of the maximum range of the dataset \code{X} among its \code{d}
#' coordinates.
#' 
#' @param x0 The initial value for the iterative procedure. By default set to 
#' \code{NULL}, meaning that \code{x0} is taken to be the sample mean of 
#' \code{X}.
#' 
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
#' complete objective function of an Oja quantile (see function 
#' \link{OjaQuantileObjective}), 
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
                       B=1e4, 
                       whiten="cov",
                       batch=NULL, 
                       gamma=NULL, x0=NULL,
                       trck=FALSE){
  
  n = nrow(X)
  d = ncol(X)
  method = match.arg(method,c("SGD","optim"))
  averaging = match.arg(averaging,c("all","suffix","none"))
  whiten = match.arg(whiten,c("Tyler","var","cov","none"))
  if(is.matrix(u)) u = t(u)
  if((!is.matrix(u))&(d==1)) u = matrix(u,nrow=1)
  if((!is.matrix(u))&(d>1))  u = matrix(u,ncol=1)
  nqs = ncol(u)
  if(nrow(u)!=d) stop("The dimensions of X and u do not match.")
  if(length(alpha)==1) alpha = rep(alpha,nqs)
  if(length(alpha)!=nqs) 
    stop("The number of rows in u and the length of alpha must equal.")
  
  # whitening transform
  if(d==1){
    whiten="none" # no whitening is needed
  }
  if(whiten=="Tyler"){
    S = ICSNP::tyler.shape(X, location=ICSNP::spatial.median(X))
  }
  if((whiten=="var")|(whiten=="cov")){
    S = var(X)
  }
  if(whiten=="none") S = diag(d)
  #
  eS = eigen(S)
  Shi = eS$vectors%*%diag(eS$values^{-1/2})%*%t(eS$vectors)
  Sh = eS$vectors%*%diag(eS$values^{+1/2})%*%t(eS$vectors)
  X0 = X       # store the original (un-whitenened) X in X0
  X = X0%*%Shi # whitened X
  
  if(method=="SGD"){
    # initialize parameters gamma and x0
    if(is.null(gamma)){
      gamma0 = apply(X,2,range)
      gamma = max(gamma0[2,]-gamma0[1,])/4
    }
    if(is.null(x0)) x0 = colMeans(X)
    if(is.null(batch)) batch = max(1,round(min(100,sqrt(n))))
    x0 = matrix(x0,nrow=d,ncol=nqs)
    
    # set up gamma schemes for final weighting
    if(averaging=="all") gammas = gamma/sqrt(1:B)
    if(averaging=="suffix"){
      gammas = gamma/sqrt(1:B)
      gammas[-(floor(3*B/4):B)] = 0
    }
    if(averaging=="none"){
      gammas = rep(0,B)
      gammas[B] = 1
    }
    
    if(!trck){
      res = OjaSGD(t(X), u, alpha, B, batch, 
                   gamma, gammas, 
                   x0, nrow(X), ncol(X), nqs, trck)
      q = t(res$q)%*%Sh
      return(list(q = q, 
                  qtrck = array(q,dim = c(nqs,ncol(X),1))
      ))
    }
    if(trck){
      res = OjaSGD(t(X), u, alpha, B, batch, 
                   gamma, gammas, 
                   x0, nrow(X), ncol(X), nqs, trck)
      q = t(res$q)%*%Sh
      qtrck = aperm(res$x0trck,c(2,1,3))
      qtrck = apply(qtrck, 3, function(x) x%*%Sh)
      dim(qtrck) = c(nqs,d,B)
      return(list(q = q, 
                  qtrck = qtrck))
    }
  }
  if(method=="optim"){
    d = ncol(X)
    if(d>5) stop("The exact computation works only in dimension d<=5.")
    res = matrix(nrow=nqs, ncol=d)
    for(i in 1:nqs){
      if(d>1) res[i,] = 
          optim(c(colMeans(X)), function(x) 
            OjaQuantileObjective(x,X,u[,i],alpha[i]))$par
      if(d==1) res[i,] = 
          optim(c(colMeans(X)), function(x) 
            OjaQuantileObjective(x,X,u[,i],alpha[i]), 
                method="Brent", lower = min(X)-1/2, upper=max(X)+1/2)$par
    }
    return(list(q = res%*%Sh, 
                qtrck = array(res%*%Sh, dim = c(nqs,ncol(X),1))))
  }
}

#### OjaRank ----
#' Oja Rank
#'
#' Iterative computation of the Oja ranks (that is, Oja signs and Oja 
#' outlyingness) for multivariate data:
#' \itemize{
#' \item A stochastic gradient descent algorithm, and 
#' \item a simple random algorithm based on maximizing the objective function
#' over a large number of randomly chosen directions.
#' }
#' 
#' @param mu A single vector of dimension \code{d}, or a numerical matrix
#' of dimension \code{nq}-times-\code{d} of \code{nq} vectors in dimension 
#' \code{d}. Each row of the matrix corresponds to one point in which
#' the Oja ranks should be calclulated.
#' 
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' 
#' @param method An idicator of the method used to compute the Oja ranks. 
#' Can be either \code{method="SGD"} for stochastic gradient descent (default), 
#' or \code{method="random"} for a naive random optimization method using 
#' the complete objective function evaluated at a large number of randomly
#' selected directions in the unit sphere.
#'  
#' @param B Number of iterations of the stochastic gradient descent method. 
#' Used for searching for the Oja sign vector. A 
#' positive integer. By default taken to be \code{B=1e3}.
#' 
#' @param B2 Number of iterations for evaluating the maximum value of the 
#' objective function. Used for calculating the Oja outlyingness/depth. A 
#' positive integer. By default taken to be \code{B2=1e4}.
#' 
#' @param batch Size of mini-batches for the stochastic gradient descent 
#' algorithm. In each iteration of the algorithm, \code{batch} of 
#' randomly selected \code{d}-tuples of points from \code{X} are taken, and 
#' the gradients of the summands of the objective function \code{g} associated 
#' with these \code{batch} subsets are used to approximate the gradient 
#' of \code{g}. By default set to \code{50}.
#' 
#' @note The algorithm \code{method="random"} should be avoided; it is typically
#' less accurate and slower:
#' \itemize{
#' \item The algorithm using \code{method="SGD"} is very fast, and can be used 
#' also for higher values of \code{n} and \code{d}. 
#' \item Using \code{method="random"} involves the exact evaluation of the 
#' complete objective function of an Oja rank (see function 
#' \link{OjaRankObjective}), 
#' which is of order \code{O(n^d)}. Thus, running with \code{method="random"} 
#' and higher \code{d} or \code{n} is extremely slow. 
#' }
#'
#' @return A list with the following components:
#' \itemize{
#' \item \code{signs}: A matrix of the approximate Oja signs of size 
#' \code{nq}-times-\code{d}, one row per one row of \code{mu}.
#' \item \code{outlyingness}: A numerical vector of the approximate Oja 
#' outlyingness of each row of \code{mu} of length \code{nq}. One entry per 
#' one row of \code{mu}.
#' }
#' The Oja ranks are obtained directly by multiplying \code{signs*outlyingness}.
#'
#' @examples
#' n = 500
#' d = 2
#' X = matrix(rnorm(n*d),ncol=d)
#' 
#' # Oja ranks of all data points
#' res = OjaRank(X, X)
#' 
#' plot(X,pch=16,cex=.25,asp=1,ann=FALSE)
#' for(i in 1:100){ 
#' # display Oja ranks for the first 100 functions as arrows
#'   arrows(X[i,1],X[i,2],
#'     X[i,1]+res$out[i]*res$sign[i,1],
#'     X[i,2]+res$out[i]*res$sign[i,2],col="orange",
#'     length=.1,lwd=2)
#'     }
#'     
#' # summary statistics of the obtained Oja outlyingness
#' summary(res$out)

OjaRank = function(mu, X, method="SGD", 
                   B=1e3, B2=1e4, batch=50){

  n = nrow(X)
  d = ncol(X)
  X = t(X)
  method = match.arg(method,c("SGD","random"))
  if(is.matrix(mu)) mu = t(mu)
  if((!is.matrix(mu))&(d==1)) mu = matrix(mu,nrow=1)
  if((!is.matrix(mu))&(d>1))  mu = matrix(mu,ncol=1)
  nqs = ncol(mu)
  if(nrow(mu)!=d) stop("The dimensions of X and mu do not match.")
  
  if(method=="SGD"){

    # initialization
    v0 = mu - rowMeans(X)
    v0 = apply(v0,2,function(x) x/c(sqrt(sum(x^2))))  
    
    # Oja signs
    resS = OjaRankSGD(mu, 
                      X, B, batch, gamma0=1, gammas=1/sqrt(1:B), 
                      v0 = v0, n, d, nq=nqs, trck=FALSE)
    
    # Oja outlyingness
    if(B2>0){
      resO = gfunCLL(resS$q, X, mu, B=B2, n, d, nv=ncol(resS$q))
      resO = pmin(pmax(resO,0),1)
    }
    if(B2==0) resO = rep(NA,nqs)
    
    return(list(signs = t(resS$q), outlyingness = resO))
  }
  
  if(method=="random"){
    
    q = matrix(nrow=nqs, ncol=d)
    resO = rep(NA,nqs)
    
    for(i in 1:n){
      g = OjaRankObjective(c(mu[,i]), t(X), eval=B)
      q[i,] = g$domain[which.max(g$vals),]
      resO[i] = max(g$vals)
    }
    return(list(signs = q, outlyingness = resO))
  }
}

#### spatialQuantile ----
#' Spatial Quantile
#'
#' Iterative computation of the spatial quantiles for multivariate data: 
#' an exact optimization algorithm using function \link[stats]{optim}.
#' 
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' @param u A single unit vector of dimension \code{d}, or a numerical matrix
#' of dimension \code{nq}-times-\code{d} of \code{nq} unit vectors in dimension 
#' \code{d}. Each row of the matrix corresponds to one unit direction in which
#' the spatial quantile should be calclulated.
#' @param alpha Magnitudes \code{alpha} of the spatial quantiles to be 
#' calculated. A single value in the interval \code{[0,1]} if \code{u} is a 
#' single vector, or a numerical vector of values inside \code{[0,1]} of length
#' \code{nq} if \code{u} is a matrix with \code{nq} rows. The \code{i}-th 
#' element of \code{alpha} corresponds to the \code{i}-th row of \code{u}.
#'
#' @return A matrix of the approximate spatial quantiles of size 
#' \code{nq}-times-\code{d}, one row per one row of \code{u} and an element of 
#' \code{alpha}. 
#'
#' @examples
#' n = 100
#' d = 2
#' X = matrix(rnorm(n*d),ncol=d)
#' 
#' alpha = 0.7
#' u = -c(1,rep(0,d-1))
#' 
#' res = spatialQuantile(X, u, alpha)
#' 
#' plot(X,pch=16,ann=FALSE)
#' points(res[,1],res[,2],col="orange",pch=16,cex=2)
#' 
#' # Multiple quantiles
#' nq = 10
#' alpha = rep(0.7,nq)
#' theta = seq(-pi,pi,length=nq)
#' u = cbind(cos(theta),sin(theta))
#' 
#' res = spatialQuantile(X, u, alpha)
#' 
#' plot(X,pch=16,ann=FALSE)
#' points(res[,1],res[,2],col="orange",pch=16,cex=2)

spatialQuantile = function(X, u, alpha){
  
  n = nrow(X)
  d = ncol(X)
  if(is.matrix(u)) u = t(u)
  if(!is.matrix(u)) u = matrix(u,ncol=1)
  nqs = ncol(u)
  if(length(alpha)==1) alpha = rep(alpha,nqs)
  if(length(alpha)!=nqs) 
    stop("The number of rows in u and the length of alpha must equal.")

  d = ncol(X)
  res = matrix(nrow=nqs, ncol=d)
  for(i in 1:nqs){
    if(d>1) res[i,] = 
        optim(c(colMeans(X)), function(x) objfSpatial(x,X,u[,i],alpha[i]))$par
    if(d==1) res[i,] = 
        optim(c(colMeans(X)), function(x) objfSpatial(x,X,u[,i],alpha[i]), 
              method="Brent", lower = min(X)-1/2, upper=max(X)+1/2)$par
  }
  return(res)
}

#### spatialRank ----
#' Spatial Rank
#'
#' Computation of the spatial ranks (that is, spatial signs and spatial 
#' outlyingness) for multivariate data.
#' 
#' @param mu A single vector of dimension \code{d}, or a numerical matrix
#' of dimension \code{nq}-times-\code{d} of \code{nq} vectors in dimension 
#' \code{d}. Each row of the matrix corresponds to one point in which
#' the spatial ranks should be calclulated.
#' 
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' 
#' @return A list with the following components:
#' \itemize{
#' \item \code{signs}: A matrix of the spatial signs of size 
#' \code{nq}-times-\code{d}, one row per one row of \code{mu}.
#' \item \code{outlyingness}: A numerical vector of the spatial
#' outlyingness of each row of \code{mu} of length \code{nq}. One entry per 
#' one row of \code{mu}.
#' }
#' The spatial ranks are obtained directly by multiplying 
#' \code{signs*outlyingness}.
#'
#' @examples
#' n = 500
#' d = 2
#' X = matrix(rnorm(n*d),ncol=d)
#' 
#' # spatial ranks of all data points
#' res = spatialRank(X, X)
#' 
#' plot(X,pch=16,cex=.25,asp=1,ann=FALSE)
#' for(i in 1:100){ 
#' # display spatial ranks for the first 100 functions as arrows
#'   arrows(X[i,1],X[i,2],
#'     X[i,1]+res$out[i]*res$sign[i,1],
#'     X[i,2]+res$out[i]*res$sign[i,2],col="orange",
#'     length=.1,lwd=2)
#'     }
#'     
#' # summary statistics of the obtained spatial outlyingness
#' summary(res$out)

spatialRank = function(mu, X){
  
  n = nrow(X)
  d = ncol(X)
  X = t(X)
  if(is.matrix(mu)) mu = t(mu)
  if((!is.matrix(mu))&(d==1)) mu = matrix(mu,nrow=1)
  if((!is.matrix(mu))&(d>1))  mu = matrix(mu,ncol=1)
  nqs = ncol(mu)
  if(nrow(mu)!=d) stop("The dimensions of X and mu do not match.")
  
  # spatial ranks
  res = spatialRankC(mu, X, n, d, nmu=nqs)
  
  # spatial outlyingness
  outl = apply(res,2,function(x) sqrt(sum(x^2)))

  # spatial signs
  signs = t(res)/outl
      
  return(list(signs = signs, outlyingness = outl))
}

#### Objective function for Quantiles ----
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
#' OjaQuantileObjective(resSGD$q, X, u, alpha)
#' 
#' neval = 101
#' xgr = seq(min(X[,1])-0.5,max(X[,1])+0.5,length=neval)
#' ygr = seq(min(X[,2])-0.5,max(X[,2])+0.5,length=neval)
#' mu = as.matrix(expand.grid(xgr,ygr))
#' objmat = matrix(OjaQuantileObjective(mu,X,u,alpha),ncol=neval)
#' 
#' contour(xgr,ygr,objmat,ann=FALSE)
#' points(X,pch=16)
#' points(resSGD$q[,1],resSGD$q[,2],col="orange",pch=16,cex=2)

OjaQuantileObjective = function(mu, X, u, alpha){
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

#### Objective function for Ranks ----
#' Objective Function for the Oja Rank/Sign/Depth/Outlyingness
#'
#' The complete objective function to be minimized for computing the Oja 
#' ranks/depht/outlyingness for multivariate data. 
#'
#' @param mu Argument at which the objective function is to be evaluated. 
#' A numerical vector of dimension \code{d}.
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' @param eval Number of points in the unit sphere in the \code{d}-space
#' where the objective function is evaluated. By default set to \code{1001}.
#'
#' @note Since this is an exact evaluation of the objective function, running
#' this function with higher \code{d} (that is, \code{d>2}) or \code{n} 
#' (\code{n>500}) can be extremely slow.
#'
#' @return A list with the following components:
#' \itemize{
#' \item \code{V}: A numerical matrix of dimensions \code{eval}-times-\code{d}
#' of values where the objective function is evaluated. If \code{d==2}, these
#' values are taken equi-distant in the unit circle. If \code{d>2}, the points 
#' are sampled randomly and uniformly on the unit sphere.
#' \item \code{vals}: A numerical vector of length \code{eval} that contains the
#' exact values of the objective function, one entry per a row of \code{V}.
#' \item \code{numerator}: The exact \code{d}-vector whose inner product with
#' \code{v} gives the numerator of the objective function at \code{v}.
#' } 
#'
#' @examples
#' n = 100
#' d = 2
#' X = matrix(rnorm(n*d),ncol=d)
#' 
#' alpha = .7
#' u = c(1,rep(0,d-1))
#' 
#' # sample Oja quantile of order alpha in direction u
#' resSGD = OjaQuantile(X, u, alpha) 
#' 
#' # evaluate the objective function for Oja ranks
#' g = OjaRankObjective(resSGD$q, X, eval=1001)
#'
#' # numerically computed Oja sign
#' (Osign = g$domain[which.max(g$vals),]) 
#' # numerically computed Oja outlyingness and depth 
#' (Ooutl = max(g$vals))
#' 1 - max(g$vals)
#' 
#' # numerically computed Oja rank
#' Ooutl*Osign

OjaRankObjective = function(mu, X, eval=1001){
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

#### Objective function for spatial quantiles ----
#' Objective Function for the spatial quantiles
#'
#' The complete objective function to be minimized for computing the spatial
#' quantiles for multivariate data. 
#'
#' @param mu Arguments at which the objective function is to be evaluated. 
#' A numerical vector of dimension \code{d}, or a matrix of dimension 
#' \code{m}-times-\code{d}, each row per one argument of the objective function.
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' @param u A single unit vector of dimension \code{d}. Corresponds to the unit 
#' direction in which the spatial quantile should be calclulated.
#' @param alpha The magnitude \code{alpha} of the spatial quantile to be 
#' calculated. A single value in the interval \code{[0,1]}.
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
#' res = spatialQuantile(X, u, alpha)
#' objfSpatial(res, X, u, alpha)
#' 
#' neval = 101
#' xgr = seq(min(X[,1])-0.5,max(X[,1])+0.5,length=neval)
#' ygr = seq(min(X[,2])-0.5,max(X[,2])+0.5,length=neval)
#' mu = as.matrix(expand.grid(xgr,ygr))
#' objmat = matrix(objfSpatial(mu,X,u,alpha),ncol=neval)
#' 
#' contour(xgr,ygr,objmat,ann=FALSE)
#' points(X,pch=16)
#' points(res[,1],res[,2],col="orange",pch=16,cex=2)

objfSpatial = function(mu, X, u, alpha){
  n = nrow(X)
  d = ncol(X)
  if(is.matrix(mu)) mu = t(mu)
  if(!is.matrix(mu)){
    if(length(mu)==d) mu = matrix(mu,ncol=1)
    if(d==1) mu = matrix(mu,nrow=1)
  }
  if(nrow(mu)!=d) stop("Dimensions of mu and X do not coincide.")
  return(objfSpatialC(mu, t(X), u, alpha, n, d, ncol(mu))/n)
}

#### rankScatter ----
#' An Oja/Tyler Scatter Estimator 
#'
#' An estimator of the scatter matrix of a multivariate distributions based on 
#' the ideas of Tyler (1987). Can be used in conjunction with the Oja signs 
#' (\link{OjaRank}) giving an Oja scatter estimator, or with the spatial signs
#' (\link{spatialRank}) giving the classical Tyler scatter estimator.
#' 
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' 
#' @param method Possible values are \code{"Oja"} for the Oja ranks, 
#' \code{"Tyler"} or \code{"spatial"} for the Tyler shape matrix constructed
#' using the spatial ranks, or \code{"var"}, or \code{"cov"} for the matrix
#' based on the trival sign vectors given simply by unit directions from the
#' sample mean of \code{X} to all data points.
#' 
#' @param B Number of iterations of the stochastic gradient descent method
#' passed to functions \link{OjaRank} and \link{spatialRank}. 
#' Used for searching for the Oja/spatial sign vector that is used in the
#' construction of the scatter estimator. A positive integer. 
#' By default taken to be \code{B=1e3}.
#' 
#' @param B2 Number of iterations for evaluating the maximum value of the 
#' objective function (Oja/spatial outlyingness) passed to functions 
#' \link{OjaRank} or \link{spatialRank}. Used for calculating the Oja/spatial
#' outlyingness/depth. A non-negative integer. By default taken to be 
#' \code{B2=0}, meaning that the outlyingness is not evaluated.
#' 
#' @param tolerance A small constant that gives the stopping criterion for the
#' iterative procedure. By default \code{1e-2}.
#' 
#' @param echo An indicator of whether a diagnostic message about the 
#' convergence of the algorithm should be printed. By default 
#' \code{TRUE}.
#' 
#' @details
#' The shape matrix is defined as the weighted sample variance matrix of 
#' \code{X}, where the weights attached to each row of \code{X} are 
#' \code{(1-outl)^expn}.
#' The final matrix is possibly scaled so that its determinant 
#' (if \code{scaling="determinant"})
#' of trace (if \code{scaling="trace"}) equals \code{scale}.  
#'
#' @return A list with the following components:
#' \itemize{
#' \item \code{location}: A vector of legnth \code{d} with the corresponding
#' location estimator, given as either the Oja median (if \code{method="Oja"}),
#' the spatial median (if \code{method="Tyler"} or \code{method="spatial"}), or
#' the same mean of \code{X} (if \code{method="cov"} or \code{method="var"}).
#' \item \code{scatter}: A \code{d}-times-\code{d} estimated scatter matrix.
#' \item \code{Xwhiten}: Matrix \code{X} whitened by the resulting scatter
#' estimator. The rows of this matrix correspond to the approximately isotropic
#' affine transform of the original dataset \code{X}.
#' \item \code{ranks}: The ranks of the data \code{X} used for the construction
#' of the scatter estimator.
#' } 
#' 
#' @references Tyler, D. E. (1987). A distribution-free M-estimator of 
#' multivariate scatter. \emph{The Annals of Statistics}, 15, 234-251.
#' 
#' @references Taskinen, S., Frahm, G., Nordhausen, K., and Oja H. (2022). A 
#' review of Tyler's shape matrix and its extensions. \emph{Robust and 
#' Multivariate Statistical Methods}, pp 23-41.
#'
#' @examples
#' n = 50
#' rho = -0.95
#' d = 2
#' Sigma = matrix(c(1,-0.95,-0.95,1),ncol=2)
#' 
#' X = mvtnorm::rmvnorm(n, sigma=Sigma)
#' 
#' OjaRS = rankScatter(X, method="Oja")
#' TylRS = rankScatter(X, method="Tyler")
#' covRS = rankScatter(X, method="cov")
#' 
#' plot(X,pch=16,xlab="",ylab="",cex=.25,asp=1)
#' car::ellipse(OjaRS$loc,OjaRS$scatter,sqrt(qchisq(.95,df=d)), center.cex=1,
#'   col="orange",lty=1,lwd=3)
#' car::ellipse(TylRS$loc,TylRS$scatter,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="pink",lty=1,lwd=3)
#' car::ellipse(covRS$loc,covRS$scatter,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="magenta",lty=1,lwd=3)
#' car::ellipse(c(0,0),Sigma,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="brown",lty=2,lwd=3)

rankScatter = function(X, 
          method=c("Oja", "Tyler", "spatial", "var", "cov"), 
          B = 1e3, B2 = 0, tolerance = 1e-2, echo=TRUE){
  
  n = nrow(X)
  d = ncol(X)
  method = match.arg(method,c("Oja", "Tyler", "spatial", "var", "cov"))  
  
  if(method=="var" | method=="cov"){
    locest = colMeans(X)
    varest = var(X)
    eV = eigen(varest)
    Vih = eV$vectors%*%diag(eV$values^{-1/2})%*%t(eV$vectors)
    Z = X%*%Vih
    rnks = list()
    rnks$signs = t(apply(t(Z) - colMeans(Z),2,function(x) x/c(sqrt(sum(x^2)))))
    rnks$outlyingness = rep(NA,n)
    
    return(list(location = locest, scatter = varest,
                Xwhiten = Z, ranks = rnks))
  }
  
  V = diag(d)
  cont = TRUE
  b = 0
  while(cont){
    b = b + 1
    eV = eigen(V)
    # all.equal(eV$vectors%*%diag(eV$values)%*%t(eV$vectors),V)
    Vih = eV$vectors%*%diag(eV$values^{-1/2})%*%t(eV$vectors)
    Vh = eV$vectors%*%diag(eV$values^{+1/2})%*%t(eV$vectors)
    Z = X%*%Vih
    if(method=="Tyler" | method=="spatial")
      sR = spatialRank(Z, Z)
    if(method=="Oja") sR = OjaRank(Z, Z, B = B, B2 = B2)
    Vn = Vh%*%(d*t(sR$signs)%*%sR$signs/n)%*%Vh
    if((err<-sqrt(sum(c(Vn - V)^2)))<tolerance) cont = FALSE
    V = Vn
  }
  varest = d*V/sum(diag(V))
  if(echo) print(paste0(
    "coverged in ",b,"th iteration, absolute error ",
    round(err,5),"."))
  if(method=="Tyler" | method=="spatial") 
    locest = c(spatialQuantile(X, c(1,rep(0,d-1)), 0))
  if(method=="Oja") locest = c(OjaQuantile(X, c(1,rep(0,d-1)), 0)$q)
  return(list(location = locest, scatter = varest, 
              Xwhiten = Z, ranks = sR))
}

#### weightedShape ----
#' Shape Estimator Based on Weighting
#'
#' An estimator of the shape matrix of multivariate elliptically symmetric
#' distributions. The estimator is a weighted sample covariance matrix, where
#' the data are weighted according to their depth (that is, 1-outlyingness).
#' 
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' 
#' @param outl A numerical vector of outlyingness values of the rows of 
#' \code{X}. This can be any numerical vector, in particular it can be vector
#' of outlyingness values using Oja ranks (\link{OjaRank}) or spatial ranks
#' (\link{spatialRank}).
#' 
#' @param expn The weights given to the rows of \code{X} are of the form
#' \code{(1-outl)^expn}, where \code{expn} is an exponent that can affect the
#' weighting scheme. By default \code{expn=1}.
#' 
#' @param scale A scale factor that sets the determinant or trace of the 
#' resulting estimator to a fixed constant \code{scale}. No scaling is applied
#' by default.
#' 
#' @param scaling An indicator of the scaling method to be used on the resulting
#' matrix. Possible values are \code{"none"} (no scaling, default), 
#' \code{"trace"} for scaling by setting a trace of the resulting matrix, or
#' \code{"determinant"} for scaling by setting the determinant of the resulting
#' matrix.
#' 
#' @details
#' The shape matrix is defined as the weighted sample variance matrix of 
#' \code{X}, where the weights attached to each row of \code{X} are 
#' \code{(1-outl)^expn}.
#' The final matrix is possibly scaled so that its determinant 
#' (if \code{scaling="determinant"})
#' of trace (if \code{scaling="trace"}) equals \code{scale}.  
#'
#' @return A list with two components:
#' \itemize{
#' \item \code{location}: A vector of legnth \code{d} with the corresponding
#' location estimator, given as the weighted sample mean, where the weights
#' are the same as for obtaining the shape matrix estimator.
#' \item \code{shape}: A \code{d}-times-\code{d} estimated shape matrix.
#' } 
#' 
#' @seealso \link{OjaQuantile} for Oja quantiles, and \link{spatialQuantile} for
#' spatial quantiles; \link{trimmedShape} for a similar estimator based on 
#' trimming, and \link{quantileShape} for a similar estimator based on 
#' directional quantiles.
#'
#' @examples
#' n = 50
#' d = 2
#' rho = -0.95
#' Sigma = matrix(c(1,-0.95,-0.95,1),ncol=2)
#' 
#' # scaling by trace of the true matrix
#' scl = sum(diag(Sigma))
#' 
#' X = mvtnorm::rmvnorm(n, sigma=Sigma)
#' 
#' Ojaout = OjaRank(X, X)$outlyingness
#' spatout = spatialRank(X, X)$outlyingness
#' 
#' resOja = weightedShape(X, Ojaout, expn=1, scale=scl, scaling="trace")
#' resspa = weightedShape(X, spatout, expn=1, scale=scl, scaling="trace")
#'   
#' plot(X,pch=16,xlab="",ylab="",cex=.25,asp=1)
#' car::ellipse(resOja$loc,resOja$shape,sqrt(qchisq(.95,df=d)), center.cex=1,
#'   col="orange",lty=1,lwd=3)
#' car::ellipse(resspa$loc,resspa$shape,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="pink",lty=1,lwd=3)
#' car::ellipse(c(0,0),Sigma,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="brown",lty=2,lwd=3)

weightedShape = function(X, outl, expn = 1, scale = NULL,
                           scaling = c("none","trace","determinant")){
  d = ncol(X)
  scaling = match.arg(scaling,c("none","trace","determinant"))
  # if(is.null(scale)) scale = d
  w = c(1 - outl)^expn
  Xw = X*c(w)/sum(w)
  wgt = list()
  wgt$location = colSums(Xw)
  wgt$shape = (t(X) - wgt$location)%*%diag(c(w))%*%
    t(t(X) - wgt$location)/sum(w)
  if(!is.null(scale)){
    if(scaling=="determinant")
      wgt$shape = wgt$shape*((scale/det(wgt$shape))^{1/d})
    if(scaling=="trace")
      wgt$shape = wgt$shape*(scale/sum(diag(wgt$shape)))
  }
  return(wgt)
}

#### trimmedShape ----
#' Shape Estimator Based on Trimming
#'
#' An estimator of the shape matrix of multivariate elliptically symmetric
#' distributions. The estimator is a trimmed sample covariance matrix, where
#' trimming is achieved by discarding observations with high outlyingness.
#' 
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' 
#' @param outl A numerical vector of outlyingness values of the rows of 
#' \code{X}. This can be any numerical vector, in particular it can be vector
#' of outlyingness values using Oja ranks (\link{OjaRank}) or spatial ranks
#' (\link{spatialRank}).
#' 
#' @param alpha A threshold value for trimming the outlyingness in \code{outl}. 
#' A single value in the interval \code{[0,1)}, indicating the proportion of
#' observations with highest outlyingness to discard. By default 
#' \code{alpha=1/2}, meaning that the estimate is computed from 
#' \code{1-alpha = 1/2} of the observations with lowest outlyingness.
#' 
#' @param scale A scale factor that sets the determinant or trace of the 
#' resulting estimator to a fixed constant \code{scale}. No scaling is applied
#' by default.
#' 
#' @param scaling An indicator of the scaling method to be used on the resulting
#' matrix. Possible values are \code{"none"} (no scaling, default), 
#' \code{"trace"} for scaling by setting a trace of the resulting matrix, or
#' \code{"determinant"} for scaling by setting the determinant of the resulting
#' matrix.
#' 
#' @details
#' The shape matrix is defined as the sample variance matrix of 
#' \code{1-alpha}-proporition of rows of \code{X} with the lower outlyingness.
#' The final matrix is possibly scaled so that its determinant 
#' (if \code{scaling="determinant"})
#' of trace (if \code{scaling="trace"}) equals \code{scale}.  
#'
#' @return A list with two components:
#' \itemize{
#' \item \code{location}: A vector of legnth \code{d} with the corresponding
#' location estimator, given as the sample mean of the observations considered
#' in constructing the shape matrix estimator.
#' \item \code{shape}: A \code{d}-times-\code{d} estimated shape matrix.
#' } 
#' 
#' @seealso \link{OjaQuantile} for Oja quantiles, and \link{spatialQuantile} for
#' spatial quantiles; \link{weightedShape} for a similar estimator based on 
#' weighting, and \link{quantileShape} for a similar estimator based on 
#' directional quantiles.
#'
#' @examples
#' n = 50
#' d = 2
#' rho = -0.95
#' Sigma = matrix(c(1,-0.95,-0.95,1),ncol=2)
#' 
#' # scaling by trace of the true matrix
#' scl = sum(diag(Sigma))
#' 
#' X = mvtnorm::rmvnorm(n, sigma=Sigma)
#' 
#' Ojaout = OjaRank(X, X)$outlyingness
#' spatout = spatialRank(X, X)$outlyingness
#' 
#' resOja = trimmedShape(X, Ojaout, alpha=1/2, scale=scl, scaling="trace")
#' resspa = trimmedShape(X, spatout, alpha=1/2, scale=scl, scaling="trace")
#'   
#' plot(X,pch=16,xlab="",ylab="",cex=.25,asp=1)
#' car::ellipse(resOja$loc,resOja$shape,sqrt(qchisq(.95,df=d)), center.cex=1,
#'   col="orange",lty=1,lwd=3)
#' car::ellipse(resspa$loc,resspa$shape,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="pink",lty=1,lwd=3)
#' car::ellipse(c(0,0),Sigma,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="brown",lty=2,lwd=3)  

trimmedShape = function(X, outl, alpha = 1/2, scale = NULL,
                          scaling = c("none","trace","determinant")){
  d = ncol(X)
  scaling = match.arg(scaling,c("none","trace","determinant"))
  # if(is.null(scale)) scale = d
  Xalpha = X[outl<=quantile(outl, 1-alpha),]
  trm = list()
  trm$location = colMeans(Xalpha)
  trm$shape = cov(Xalpha)
  if(!is.null(scale)){
    if(scaling=="determinant")
      trm$shape = trm$shape*((scale/det(trm$shape))^{1/d})
    if(scaling=="trace")
      trm$shape = trm$shape*(scale/sum(diag(trm$shape)))
  }
  return(trm)
}

#### quantileShape ----
#' Shape Estimator Based on Directional Quantiles
#'
#' An estimator of the shape matrix of multivariate elliptically symmetric
#' distributions. The estimator is based on a notion of directional quantiles
#' (spatial quantiles of Oja quantiles).
#' 
#' @param X A numerical matrix with the dataset of dimension 
#' \code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
#' \code{d} is its dimension.
#' 
#' @param alpha A grid of magnitudes \code{alpha} of the spatial/Oja quantiles 
#' to be considered in the estimator. A single value, or a numerical vector
#' of values in the interval \code{(0,1]}. By default we take a single value 
#' \code{alpha=1/2}.
#' 
#' @param method An indicator of which directional quantiles to use. Possible
#' values are \code{"Oja"} for Oja quantiles (default) or \code{"spatial"} for
#' spatial quantiles.
#' 
#' @param qs Number of quantiles to consider at each order of \code{alpha}. By
#' default \code{qs=100}. In dimension \code{d=2}, these quantiles are taken in
#' directions that are equidistant on the unit circle. In dimension \code{d>2},
#' a random sample of directions on the unit sphere is taken.
#'
#' @param scale A scale factor that sets the determinant or trace of the 
#' resulting estimator to a fixed constant \code{scale}. No scaling is applied
#' by default.
#' 
#' @param scaling An indicator of the scaling method to be used on the resulting
#' matrix. Possible values are \code{"none"} (no scaling, default), 
#' \code{"trace"} for scaling by setting a trace of the resulting matrix, or
#' \code{"determinant"} for scaling by setting the determinant of the resulting
#' matrix.
#' 
#' @details
#' The shape matrix is defined as the sample variance matrix of a collection of
#' directional quantiles of \code{X}. The quantiles are taken at all levels in
#' the vector \code{alpha}; at each level, \code{qs} quantiles are considered.
#' In total, \code{length(alpha)*qs} quantiles are considered. The final matrix 
#' is possibly scaled so that its determinant (if \code{scaling="determinant"})
#' of trace (if \code{scaling="trace"}) equals \code{scale}.  
#'
#' @return A list with two components:
#' \itemize{
#' \item \code{location}: A vector of legnth \code{d} with the corresponding
#' (Oja/spatial) directional median.
#' \item \code{shape}: A \code{d}-times-\code{d} estimated shape matrix.
#' } 
#' 
#' @seealso \link{OjaQuantile} for Oja quantiles, and \link{spatialQuantile} for
#' spatial quantiles; \link{trimmedShape} for a similar estimator based on 
#' trimming, and \link{weightedShape} for a similar estimator based on 
#' weighting.
#'
#' @examples
#' n = 50
#' d = 2
#' rho = -0.95
#' Sigma = matrix(c(1,-0.95,-0.95,1),ncol=2)
#' 
#' # scaling by trace of the true matrix
#' scl = sum(diag(Sigma))
#' 
#' X = mvtnorm::rmvnorm(n, sigma=Sigma)
#' 
#' alphag = seq(0,1/2,length=11)[-1]
#' resOja = quantileShape(X, alphag, method="Oja",
#'   scale=scl, scaling="trace")
#' resspa = quantileShape(X, alphag, method="spatial", 
#'   scale=scl, scaling="trace")
#'   
#' plot(X,pch=16,xlab="",ylab="",cex=.25,asp=1)
#' car::ellipse(resOja$loc,resOja$shape,sqrt(qchisq(.95,df=d)), center.cex=1,
#'   col="orange",lty=1,lwd=3)
#' car::ellipse(resspa$loc,resspa$shape,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="pink",lty=1,lwd=3)
#' car::ellipse(c(0,0),Sigma,sqrt(qchisq(.95,df=d)), center.cex=1,
#'  col="brown",lty=2,lwd=3)  

quantileShape = function(X, alpha = 1/2, 
                           method=c("Oja", "spatial"), 
                           qs = 1e2, scale = NULL,
                           scaling = c("none","trace","determinant")){
  d = ncol(X)
  method = match.arg(method,c("Oja", "spatial"))
  scaling = match.arg(scaling,c("none","trace","determinant"))
  # if(is.null(scale)) scale = d
      
  # build set of u's for quantiles
  if(d==2){
    thetas = seq(-pi,pi,length=qs+1)[-1]
    u = cbind(cos(thetas),sin(thetas))
  } else {
    u = matrix(rnorm(qs*d),ncol=d)
    u = t(apply(u, 1, function(x) x/c(sqrt(sum(x^2)))))
  }
  nalpha = length(alpha)
  
  if(method=="Oja"){
    loc = c(OjaQuantile(X, u[1,], 0)$q)
    Qs = matrix(nrow=0, ncol=d)
    for(ialpha in 1:nalpha) 
      Qs = rbind(Qs, OjaQuantile(X, u, rep(alpha[ialpha],qs))$q)
  }
  if(method=="spatial"){
    loc = c(spatialQuantile(X, u[1,], 0))
    Qs = matrix(nrow=0, ncol=d)
    for(ialpha in 1:nalpha) 
      Qs = rbind(Qs, spatialQuantile(X, u, rep(alpha[ialpha],qs)))
  }
  
  res = list()
  res$location = loc
  res$shape = var(Qs)
  if(!is.null(scale)){
    if(scaling=="determinant") 
      res$shape = res$shape*((scale/det(res$shape))^{1/d})
    if(scaling=="trace") 
      res$shape = res$shape*(scale/sum(diag(res$shape)))
  }
  return(res)
}
