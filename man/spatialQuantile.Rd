% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/code.R
\name{spatialQuantile}
\alias{spatialQuantile}
\title{Spatial Quantile}
\usage{
spatialQuantile(X, u, alpha)
}
\arguments{
\item{X}{A numerical matrix with the dataset of dimension 
\code{n}-times-\code{d}, where \code{n} is the size of the dataset and 
\code{d} is its dimension.}

\item{u}{A single unit vector of dimension \code{d}, or a numerical matrix
of dimension \code{nq}-times-\code{d} of \code{nq} unit vectors in dimension 
\code{d}. Each row of the matrix corresponds to one unit direction in which
the spatial quantile should be calclulated.}

\item{alpha}{Magnitudes \code{alpha} of the spatial quantiles to be 
calculated. A single value in the interval \code{[0,1]} if \code{u} is a 
single vector, or a numerical vector of values inside \code{[0,1]} of length
\code{nq} if \code{u} is a matrix with \code{nq} rows. The \code{i}-th 
element of \code{alpha} corresponds to the \code{i}-th row of \code{u}.}
}
\value{
A matrix of the approximate spatial quantiles of size 
\code{nq}-times-\code{d}, one row per one row of \code{u} and an element of 
\code{alpha}.
}
\description{
Iterative computation of the spatial quantiles for multivariate data: 
an exact optimization algorithm using function \link[stats]{optim}.
}
\examples{
n = 100
d = 2
X = matrix(rnorm(n*d),ncol=d)

alpha = 0.7
u = -c(1,rep(0,d-1))

res = spatialQuantile(X, u, alpha)

plot(X,pch=16,ann=FALSE)
points(res[,1],res[,2],col="orange",pch=16,cex=2)

# Multiple quantiles
nq = 10
alpha = rep(0.7,nq)
theta = seq(-pi,pi,length=nq)
u = cbind(cos(theta),sin(theta))

res = spatialQuantile(X, u, alpha)

plot(X,pch=16,ann=FALSE)
points(res[,1],res[,2],col="orange",pch=16,cex=2)
}
