% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/code.R
\name{OjaRank}
\alias{OjaRank}
\title{Oja Rank}
\usage{
OjaRank(mu, X, method = "SGD", B = 1000, B2 = 10000, batch = 50)
}
\arguments{
\item{mu}{A single vector of dimension \code{d}, or a numerical matrix
of dimension \code{nq}-times-\code{d} of \code{nq} vectors in dimension 
\code{d}. Each row of the matrix corresponds to one point in which
the Oja ranks should be calclulated.}

\item{X}{A numerical matrix with the dataset of dimension 
\code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
\code{d} is its dimension.}

\item{method}{An idicator of the method used to compute the Oja ranks. 
Can be either \code{method="SGD"} for stochastic gradient descent (default), 
or \code{method="random"} for a naive random optimization method using 
the complete objective function evaluated at a large number of randomly
selected directions in the unit sphere.}

\item{B}{Number of iterations of the stochastic gradient descent method. 
Used for searching for the Oja sign vector. A 
positive integer. By default taken to be \code{B=1e3}.}

\item{B2}{Number of iterations for evaluating the maximum value of the 
objective function. Used for calculating the Oja outlyingness/depth. A 
positive integer. By default taken to be \code{B2=1e4}.}

\item{batch}{Size of mini-batches for the stochastic gradient descent 
algorithm. In each iteration of the algorithm, \code{batch} of 
randomly selected \code{d}-tuples of points from \code{X} are taken, and 
the gradients of the summands of the objective function \code{g} associated 
with these \code{batch} subsets are used to approximate the gradient 
of \code{g}. By default set to \code{50}.}
}
\value{
A list with the following components:
\itemize{
\item \code{signs}: A matrix of the approximate Oja signs of size 
\code{nq}-times-\code{d}, one row per one row of \code{mu}.
\item \code{outlyingness}: A numerical vector of the approximate Oja 
outlyingness of each row of \code{mu} of length \code{nq}. One entry per 
one row of \code{mu}.
}
The Oja ranks are obtained directly by multiplying \code{signs*outlyingness}.
}
\description{
Iterative computation of the Oja ranks (that is, Oja signs and Oja 
outlyingness) for multivariate data:
\itemize{
\item A stochastic gradient descent algorithm, and 
\item a simple random algorithm based on maximizing the objective function
over a large number of randomly chosen directions.
}
}
\note{
The algorithm \code{method="random"} should be avoided; it is typically
less accurate and slower:
\itemize{
\item The algorithm using \code{method="SGD"} is very fast, and can be used 
also for higher values of \code{n} and \code{d}. 
\item Using \code{method="random"} involves the exact evaluation of the 
complete objective function of an Oja rank (see function 
\link{OjaRankObjective}), 
which is of order \code{O(n^d)}. Thus, running with \code{method="random"} 
and higher \code{d} or \code{n} is extremely slow. 
}
}
\examples{
n = 500
d = 2
X = matrix(rnorm(n*d),ncol=d)

# Oja ranks of all data points
res = OjaRank(X, X)

plot(X,pch=16,cex=.25,asp=1,ann=FALSE)
for(i in 1:100){ 
# display Oja ranks for the first 100 functions as arrows
  arrows(X[i,1],X[i,2],
    X[i,1]+res$out[i]*res$sign[i,1],
    X[i,2]+res$out[i]*res$sign[i,2],col="orange",
    length=.1,lwd=2)
    }
    
# summary statistics of the obtained Oja outlyingness
summary(res$out)
}
