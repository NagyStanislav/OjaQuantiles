// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

#include <Rcpp.h>
using namespace Rcpp; // for Rcout to work

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
double signC(double x){
  if(x>0) return(1);
  if(x<0) return(-1);
  return(0);
}

// [[Rcpp::export]]
arma::vec cvectorC(arma::mat X1, int d) {
  // X1 is a d-times-d matrix with observations in columns
  arma::vec cvec(d, arma::fill::zeros);
  arma::mat Y(d,d);
  for(int i=0;i<d;i++){
    Y = X1;
    Y.row(i).ones();
    cvec(i) = pow(-1,i+2+d+1+i)*det(Y);
  }
  return(cvec);
}

// [[Rcpp::export]]
arma::vec grdC(arma::vec mu, arma::mat X1, arma::vec u, double alpha, int d){
  // X1 is a d-times-d matrix, where observations are in columns (!)
  // X1 = t(X1)
  arma::vec cvec = cvectorC(X1, d);
  double dW = pow(-1,d+2)*det(X1) + dot(mu,cvec);
  return(cvec*(signC(dW)-alpha*signC(dot(u,cvec))));
}

// [[Rcpp::export]]
Rcpp::List OjaSGD(arma::mat X, arma::mat u, arma::vec alpha,int B, int batch, 
                  double gamma0, arma::vec gammas, arma::mat x0, int n, int d, int nq, bool trck) {
  // X is a d-times-n matrix (!)
  // u is a d-times-nq matrix, each column for one element in alpha
  // alpha is a nq-long vector
  // x0 is a d-times-nq matrix of initial values
  
  arma::mat X1(d,d);
  arma::vec cvec(d);
  double dW, dW0, gamma;
  arma::mat q(d,nq, arma::fill::zeros); // matrix of final estimates
  double qden; // denominator for q = sum(gammas)
  int Bcube;
  if(trck) Bcube = B; else Bcube = 1;
  
  arma::cube x0trck(d,nq,Bcube, arma::fill::value(arma::datum::nan)); // tracking progress of optimization 
  
  qden = 0;
  
  for(int b=0; b<B; b++){
    arma::imat Bs = arma::randi(batch, d, arma::distr_param(0, n-1));
    // Rcout << Bs << std::endl;
    arma::mat grB(d, nq, arma::fill::zeros);
    for(int ib=0; ib<batch; ib++){
      // setting up the matrix X1
      for(int j=0; j<d; j++) X1.col(j) = X.col(Bs(ib,j));
      // computing the gradient
      cvec = cvectorC(X1, d);                   // vector c
      dW0 = pow(-1,d+2)*det(X1);                // constant c0
        for(int k=0; k<nq; k++){
          dW = dW0 + dot(x0.col(k),cvec);       // det(W)
          grB.col(k) = grB.col(k) + cvec*(signC(dW)-alpha(k)*signC(dot(u.col(k),cvec)));
          // same as 
          // grB = grB + grdC(x0, X1, u, alpha, d);
        }
      }
    grB = grB/batch;
    gamma = gamma0/sqrt(b+1);
    // Rcout << b << x0 << std::endl;
    for(int k=0; k<nq; k++){
      x0.col(k) = x0.col(k) - gamma*grB.col(k);
      q.col(k) = q.col(k) + gammas(b)*x0.col(k);
      }
    qden = qden + gammas(b);
    if(trck){
       for(int k=0; k<nq; k++) (x0trck.slice(b)).col(k) = x0.col(k);
    }
  }
  return(
    Rcpp::List::create(
      Rcpp::Named("qden")=qden,
      Rcpp::Named("q")=q/qden,
      Rcpp::Named("x0")=x0,
      Rcpp::Named("x0trck")=x0trck)
    );  
}

// [[Rcpp::export]]
Rcpp::List OjaSGD2(arma::mat X, arma::mat u, arma::vec alpha,int B, int batch, 
double gamma0, arma::mat x0, int n, int d, int nq, bool trck) {
  // X is a d-times-n matrix (!)
  // u is a d-times-nq matrix, each column for one element in alpha
  // alpha is a nq-long vector
  // x0 is a d-times-nq matrix of initial values

  arma::mat X1(d,d);
  arma::vec cvec(d);
  double dW, dW0, gamma;
  arma::mat q(d,nq); // matrix of final estimates
  int Bcube;
  if(trck) Bcube = B; else Bcube = 1;
  
  arma::cube x0trck(d,nq,Bcube, arma::fill::value(arma::datum::nan)); // tracking progress of optimization 
  
  q = x0;
  
  for(int b=0; b<B; b++){
    arma::imat Bs = arma::randi(batch, d, arma::distr_param(0, n-1));
     //   Rcout << Bs << std::endl;
    arma::mat grB(d, nq, arma::fill::zeros);
    for(int ib=0; ib<batch; ib++){
      // setting up the matrix X1
      for(int j=0; j<d; j++) X1.col(j) = X.col(Bs(ib,j));
      // computing the gradient
      cvec = cvectorC(X1, d);                   // vector c
      dW0 = pow(-1,d+2)*det(X1);                // constant c0
        for(int k=0; k<nq; k++){
          dW = dW0 + dot(x0.col(k),cvec);       // det(W)
          grB.col(k) = grB.col(k) + cvec*(signC(dW)-alpha(k)*signC(dot(u.col(k),cvec)));
          // same as 
          // grB = grB + grdC(x0, X1, u, alpha, d);
        }
      }
    grB = grB/batch;
    gamma = gamma0/sqrt(b+1);
    for(int k=0; k<nq; k++) x0.col(k) = x0.col(k) - gamma*grB.col(k);
    if(trck){
       for(int k=0; k<nq; k++) (x0trck.slice(b)).col(k) = x0.col(k);
    }
  }
  return(
    Rcpp::List::create(
      Rcpp::Named("x0")=x0,
      Rcpp::Named("x0trck")=x0trck)
    );  
}

// [[Rcpp::export]]
arma::vec objfC(arma::mat mu, arma::mat X, arma::vec u, double alpha, int n, int d, int nmu){
  // mu is a d-times-nmu matrix
  // X is a d-times-n matrix (!)
  // u is a d-vector

  arma::vec res(nmu, arma::fill::zeros);
  arma::vec cvec(d);
  double Wd, Wd0;
  arma::mat X1(d,d);

  if(d==1){
    for(int i1 = 0; i1<n; i1++){
      X1.col(0) = X.col(i1); 
      cvec = cvectorC(X1, d);
      Wd0 = pow(-1,d+2)*det(X1);
      for(int k=0; k<nmu; k++){
        Wd = Wd0 + dot(mu.col(k),cvec);
        res(k) = res(k) + abs(Wd)*(1-alpha*signC(dot(u,cvec))*signC(Wd));
        }         
    }
  return(res);
  }
  if(d==2){
    for(int i1 = 0; i1<n-1; i1++){
      for(int i2 = i1+1; i2<n; i2++){
        X1.col(0) = X.col(i1); 
        X1.col(1) = X.col(i2);
        cvec = cvectorC(X1, d);
        //
        Wd0 = pow(-1,d+2)*det(X1);
        for(int k=0; k<nmu; k++){
          Wd = Wd0 + dot(mu.col(k),cvec);
          res(k) = res(k) + abs(Wd)*(1-alpha*signC(dot(u,cvec))*signC(Wd));
          }          
      }
    }
  return(res);
  }
  if(d==3){
    for(int i1 = 0; i1<n-2; i1++){
      for(int i2 = i1+1; i2<n-1; i2++){
        for(int i3 = i2+1; i3<n; i3++){
          X1.col(0) = X.col(i1); 
          X1.col(1) = X.col(i2); 
          X1.col(2) = X.col(i3); 
          cvec = cvectorC(X1, d);
          Wd0 = pow(-1,d+2)*det(X1);
          for(int k=0; k<nmu; k++){
            Wd = Wd0 + dot(mu.col(k),cvec);
            res(k) = res(k) + abs(Wd)*(1-alpha*signC(dot(u,cvec))*signC(Wd));
            }          
        }
      }
    }
  return(res);
  }
  if(d==4){
    for(int i1 = 0; i1<n-3; i1++){
      for(int i2 = i1+1; i2<n-2; i2++){
        for(int i3 = i2+1; i3<n-1; i3++){
          for(int i4 = i3+1; i4<n; i4++){
            X1.col(0) = X.col(i1); 
            X1.col(1) = X.col(i2); 
            X1.col(2) = X.col(i3); 
            X1.col(3) = X.col(i4); 
            cvec = cvectorC(X1, d);
            Wd0 = pow(-1,d+2)*det(X1);
            for(int k=0; k<nmu; k++){
              Wd = Wd0 + dot(mu.col(k),cvec);
              res(k) = res(k) + abs(Wd)*(1-alpha*signC(dot(u,cvec))*signC(Wd));
              }         
          }
        }
      }
    }
  return(res);
  }
  if(d==5){
    for(int i1 = 0; i1<n-4; i1++){
      for(int i2 = i1+1; i2<n-3; i2++){
        for(int i3 = i2+1; i3<n-2; i3++){
          for(int i4 = i3+1; i4<n-1; i4++){
            for(int i5 = i4+1; i5<n; i5++){
              X1.col(0) = X.col(i1); 
              X1.col(1) = X.col(i2); 
              X1.col(2) = X.col(i3); 
              X1.col(3) = X.col(i4); 
              X1.col(4) = X.col(i5); 
              cvec = cvectorC(X1, d);
              Wd0 = pow(-1,d+2)*det(X1);
              for(int k=0; k<nmu; k++){
                Wd = Wd0 + dot(mu.col(k),cvec);
                res(k) = res(k) + abs(Wd)*(1-alpha*signC(dot(u,cvec))*signC(Wd));
                }          
            }
          }
        }
      }
    }
  return(res);
  }
  // works only for d<=5
  return(res);
}

// Oja outlyingness and signs

// [[Rcpp::export]]
Rcpp::List gfunC(arma::mat v, arma::mat X, arma::vec mu, int n, int d, int nv){
  // mu is a d-vector
  // X is a d-times-n matrix (!)
  // v is a nv-times-d matrix (!!)
  
  arma::vec cvec(d), num(d, arma::fill::zeros), den(nv, arma::fill::zeros);
  double c0;
  arma::mat X1(d,d);
  
  if(d==2){
    for(int i1 = 0; i1<n-1; i1++){
      for(int i2 = i1+1; i2<n; i2++){
        X1.col(0) = X.col(i1); 
        X1.col(1) = X.col(i2); 
        cvec = cvectorC(X1, d);
        c0 = det(X1);
        num = num + signC(c0 + dot(mu,cvec))*cvec;
        den = den + (abs(v*cvec));   
      }
    }
  }
  if(d==3){
    for(int i1 = 0; i1<n-2; i1++){
      for(int i2 = i1+1; i2<n-1; i2++){
        for(int i3 = i2+1; i3<n; i3++){
          X1.col(0) = X.col(i1); 
          X1.col(1) = X.col(i2); 
          X1.col(2) = X.col(i3); 
          cvec = cvectorC(X1, d);
          c0 = det(X1);
          num = num + signC(c0 + dot(mu,cvec))*cvec;
          den = den + (abs(v*cvec));           
        }
      }
    }
  } 
  if(d==4){
    for(int i1 = 0; i1<n-3; i1++){
      for(int i2 = i1+1; i2<n-2; i2++){
        for(int i3 = i2+1; i3<n-1; i3++){
          for(int i4 = i3+1; i4<n; i4++){
            X1.col(0) = X.col(i1); 
            X1.col(1) = X.col(i2); 
            X1.col(2) = X.col(i3); 
            X1.col(3) = X.col(i4); 
            cvec = cvectorC(X1, d);
            c0 = det(X1);
            num = num + signC(c0 + dot(mu,cvec))*cvec;
            den = den + (abs(v*cvec)); 
          }
        }
      }
    }
  }
  if(d==5){
    for(int i1 = 0; i1<n-4; i1++){
      for(int i2 = i1+1; i2<n-3; i2++){
        for(int i3 = i2+1; i3<n-2; i3++){
          for(int i4 = i3+1; i4<n-1; i4++){
            for(int i5 = i4+1; i5<n; i5++){
              X1.col(0) = X.col(i1); 
              X1.col(1) = X.col(i2); 
              X1.col(2) = X.col(i3); 
              X1.col(3) = X.col(i4); 
              X1.col(4) = X.col(i5); 
              cvec = cvectorC(X1, d);
              c0 = det(X1);
              num = num + signC(c0 + dot(mu,cvec))*cvec;
              den = den + (abs(v*cvec));         
            }
          }
        }
      }
    }
  } 
  return(Rcpp::List::create(Rcpp::Named("num")=num,Rcpp::Named("den")=den));
}

// Procedures for the computation of the spatial signs

// [[Rcpp::export]]
arma::vec objfSpatialC(arma::mat mu, arma::mat X, arma::vec u, double alpha, int n, int d, int nmu){
  // mu is a d-times-nmu matrix
  // X is a d-times-n matrix (!)
  // u is a d-vector

  arma::vec res(nmu, arma::fill::zeros);
  arma::vec X1(d);

   for(int i = 0; i<n; i++){
      X1 = X.col(i); 
      for(int k=0; k<nmu; k++){
        res(k) = res(k) + arma::norm(X1-mu.col(k),2) + alpha*dot(u,X1-mu.col(k));
        }         
    }
  return(res);
}