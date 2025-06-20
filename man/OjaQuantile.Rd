% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/code.R
\name{OjaQuantile}
\alias{OjaQuantile}
\title{Oja Quantile}
\usage{
OjaQuantile(
  X,
  u,
  alpha,
  method = "SGD",
  averaging = "all",
  B = 10000,
  whiten = "cov",
  batch = NULL,
  gamma = NULL,
  x0 = NULL,
  trck = FALSE
)
}
\arguments{
\item{X}{A numerical matrix with the dataset of dimension 
\code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
\code{d} is its dimension.}

\item{u}{A single unit vector of dimension \code{d}, or a numerical matrix
of dimension \code{nq}-times-\code{d} of \code{nq} unit vectors in dimension 
\code{d}. Each row of the matrix corresponds to one unit direction in which
the Oja quantile should be calclulated.}

\item{alpha}{Magnitudes \code{alpha} of the Oja quantiles to be calculated.
A single value in the interval \code{[0,1]} if \code{u} is a 
single vector, or a numerical vector of values inside \code{[0,1]} of length
\code{nq} if \code{u} is a matrix with \code{nq} rows. The \code{i}-th 
element of \code{alpha} corresponds to the \code{i}-th row of \code{u}.}

\item{method}{An idicator of the method used to compute the Oja quantiles. 
Can be either \code{method="SGD"} for stochastic gradient descent (default), 
or \code{method="optim"} for a numerical optimization method using function
\link[stats]{optim}. 

\emph{All the further arguments to this function apply only in the case} 
\code{method="SGD"}.}

\item{averaging}{The averaging method used for the final estimates using the 
stochastic gradient method. By default set to \code{averaging="all"}, meaning
that the resulting estimate is a weighted average of all the \code{B} 
updates. Other possible values are \code{averaging="suffix"}, where the 
weighted average of only the last \code{B/4} updates is considered, or
\code{averaging="none"}, where only the final \code{B}th update is taken, 
without any averaging.}

\item{B}{Number of iterations of the stochastic gradient descent method. A 
positive integer. By default taken to be \code{B=1e4}.}

\item{whiten}{The pre-whitening method of the dataset to achieve better
numerical stability. Possible values are \code{"none"}, \code{"cov"}, 
\code{"var"}, or \code{"Tyler"}. Default is \code{"cov"}, which means that
whitening is performed using the sample
covariance matrix. This is the same as calling \code{whiten="var"}. With
\code{whiten=="none"} no pre-whitening is performed. For \code{"Tyler"},
the data matrix is whitened using the robust Tyler's scatter 
estimator, with a center at the spatial median.}

\item{batch}{Size of mini-batches for the stochastic gradient descent 
algorithm. In each iteration of the algorithm, \code{batch} of 
randomly selected \code{d}-tuples of points from \code{X} are taken, and 
the gradients of the summands of the objective function \code{f} associated 
with these \code{batch} subsets are used to approximate the gradient 
of \code{f}. By default set to \code{NULL}, in which case 
\code{batch} is taken as \code{max(1,round(min(100,sqrt(n))))}, where 
\code{n} is the sample size.}

\item{gamma}{The initial step value for the stochastic gradient descent. By 
default set to \code{NULL}, which means that \code{gamma} is evaluated as
\code{1/4} of the maximum range of the dataset \code{X} among its \code{d}
coordinates.}

\item{x0}{The initial value for the iterative procedure. By default set to 
\code{NULL}, meaning that \code{x0} is taken to be the sample mean of 
\code{X}.}

\item{trck}{Indicator of whether as an output of the procedure, also the
whole history of \code{B} iterated solutions should be given. 
By default set to \code{FALSE}.}
}
\value{
A list with the following components:
\itemize{
\item \code{q}: A matrix of the approximate Oja quantiles of size 
\code{nq}-times-\code{d}, one row per one row of \code{u} and an element of 
\code{alpha}. 
\item \code{qtrck}: If \code{trck=FALSE}, an array of size 
\code{nq}-times-\code{d}-times-\code{1} which contains the same matrix as in
\code{q}. For \code{method="SGD"} and \code{trck=TRUE}, an array of size 
\code{nq}-times-\code{d}-times-\code{B}, where in the last dimension we have
the whole history of the \code{B} iterations of the approximate procedure for 
the Oja quantiles.
}
}
\description{
Iterative computation of the Oja quantiles for multivariate data:
\itemize{
\item A stochastic gradient descent algorithm, and 
\item an exact optimization algorithm using function \link[stats]{optim}.
}
}
\note{
The algorithm \code{method="optim"} typically gives more accurate 
results. On the other hand, the speed of the two algorithms differs 
drastically:
\itemize{
\item The algorithm using \code{method="SGD"} is very fast, and can be used 
also for higher values of \code{n} and \code{d}. 
\item Using \code{method="optim"} involves the exact evaluation of the 
complete objective function of an Oja quantile (see function 
\link{OjaQuantileObjective}), 
which is of order \code{O(n^d)}. Thus, running with \code{method="optim"} and
higher \code{d} or \code{n} is extremely slow. 
}
}
\examples{
n = 100
d = 2
X = matrix(rnorm(n*d),ncol=d)

alpha = 0.7
u = -c(1,rep(0,d-1))

# Computation using stochastic gradient descent
resSGD = OjaQuantile(X, u, alpha)

# Computation using function optim
resoptim = OjaQuantile(X, u, alpha, method="optim")

plot(X,pch=16,ann=FALSE)
points(resSGD$q[,1],resSGD$q[,2],col="orange",pch=16,cex=2)
points(resoptim$q[,1],resoptim$q[,2],col="violet",pch=1,cex=3,lwd=3)

# Multiple quantiles
nq = 10
alpha = rep(0.7,nq)
theta = seq(-pi,pi,length=nq)
u = cbind(cos(theta),sin(theta))

# Computation using stochastic gradient descent
resSGD = OjaQuantile(X, u, alpha)

# Computation using function optim
resoptim = OjaQuantile(X, u, alpha, method="optim")

plot(X,pch=16,ann=FALSE)
points(resSGD$q[,1],resSGD$q[,2],col="orange",pch=16,cex=2)
points(resoptim$q[,1],resoptim$q[,2],col="violet",pch=1,cex=3,lwd=3)

# Multiple quantiles including tracing the algorithm
resSGD = OjaQuantile(X, u, alpha, B=100, trck=TRUE)
plot(X,pch=16,ann=FALSE)
points(resSGD$q[,1],resSGD$q[,2],col="orange",pch=16,cex=2)
for(i in 1:nq) 
lines(resSGD$qtrck[i,1,],resSGD$qtrck[i,2,],col="orange",lwd=2)
}
