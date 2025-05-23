# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

signC <- function(x) {
    .Call(`_OjaQuantiles_signC`, x)
}

cvectorC <- function(X1, d) {
    .Call(`_OjaQuantiles_cvectorC`, X1, d)
}

grdC <- function(mu, X1, u, alpha, d) {
    .Call(`_OjaQuantiles_grdC`, mu, X1, u, alpha, d)
}

OjaSGD <- function(X, u, alpha, B, batch, gamma0, gammas, x0, n, d, nq, trck) {
    .Call(`_OjaQuantiles_OjaSGD`, X, u, alpha, B, batch, gamma0, gammas, x0, n, d, nq, trck)
}

OjaSGD2 <- function(X, u, alpha, B, batch, gamma0, x0, n, d, nq, trck) {
    .Call(`_OjaQuantiles_OjaSGD2`, X, u, alpha, B, batch, gamma0, x0, n, d, nq, trck)
}

objfC <- function(mu, X, u, alpha, n, d, nmu) {
    .Call(`_OjaQuantiles_objfC`, mu, X, u, alpha, n, d, nmu)
}

gfunC <- function(v, X, mu, n, d, nv) {
    .Call(`_OjaQuantiles_gfunC`, v, X, mu, n, d, nv)
}

objfSpatialC <- function(mu, X, u, alpha, n, d, nmu) {
    .Call(`_OjaQuantiles_objfSpatialC`, mu, X, u, alpha, n, d, nmu)
}

