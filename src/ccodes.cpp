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
Rcpp::List OjaRankSGD(arma::mat mu, arma::mat X, int B, int batch, 
                  double gamma0, arma::vec gammas, arma::mat v0, int n, int d, int nq, bool trck) {
  // mu is a d-times-nq matrix, each column for one point
  // X is a d-times-n matrix (!)
  // v0 is a d-times-nq matrix of initial values
  
  arma::mat X1(d,d);
  arma::vec cvec(d);
  double dW, dW0, gamma, taux;
  arma::mat q(d, nq, arma::fill::zeros); // matrix of final estimates
  double qden; // denominator for q = sum(gammas)
  int Bcube;
  if(trck) Bcube = B; else Bcube = 1;
  
  arma::cube v0trck(d,nq,Bcube, arma::fill::value(arma::datum::nan)); // tracking progress of optimization
  arma::cube w0trck(d,nq,Bcube, arma::fill::value(arma::datum::nan)); // tracking progress of optimization: weighted  
  qden = 0;
  
  for(int b=0; b<B; b++){
    arma::imat Bs = arma::randi(batch, d, arma::distr_param(0, n-1));
    // Rcout << Bs << std::endl;
    arma::mat grB(d, nq, arma::fill::zeros);
    arma::mat ta(d, nq, arma::fill::zeros); // matrix of sign(c0+mu'c)*c
    arma::vec tb(nq, arma::fill::zeros);    // vector of |v'c|
    arma::vec tc(nq, arma::fill::zeros);    // vector of sign(c0+mu'c)*(v'c)
    arma::mat td(d, nq, arma::fill::zeros); // matrix of sign(v'c)*c
    arma::vec te(nq, arma::fill::zeros);    // vector of |v'c|^2
    for(int ib=0; ib<batch; ib++){       
      // setting up the matrix X1
      for(int j=0; j<d; j++) X1.col(j) = X.col(Bs(ib,j));
      // computing the gradient
      cvec = cvectorC(X1, d);                          // vector c
      dW0 = pow(-1,d+2)*det(X1);                       // constant c0
      for(int k=0; k<nq; k++){
        dW = dW0 + dot(mu.col(k),cvec);                // (c0 + mu'c) 
        taux = dot(v0.col(k), cvec);                   // v'c  
        // Rcout << "taux:" << taux << std::endl;
        ta.col(k) = ta.col(k) + signC(dW)*cvec;        // sign(c0+mu'c)*c
        tb(k) = tb(k) + abs(taux);                     // |v'c|
        tc(k) = tc(k) + signC(dW)*taux;                // sign(c0+mu'c)*(v'c)
        td.col(k) = td.col(k) + signC(taux)*cvec;      // sign(v'c)*c
        te(k) = te(k) + pow(taux,2);                   // |v'c|^2
      }
    }
    tb = tb/batch;
    tc = tc/batch;
    // Rcout << "ta:" << ta << std::endl;
    // Rcout << "tb:" << tb << std::endl;
    // Rcout << "tc:" << tc << std::endl;
    // Rcout << "td:" << td << std::endl;
    // Rcout << "te:" << te << std::endl;
    for(int k=0; k<nq; k++) grB.col(k) = (ta.col(k)*tb(k) - tc(k)*td.col(k))/(te(k)); // batched gradient
    gamma = gamma0/sqrt(b+1);
    // Rcout << b << grB << std::endl;
    for(int k=0; k<nq; k++){
      v0.col(k) = v0.col(k) + gamma*grB.col(k);
      taux = 0;
      for(int k2=0; k2<d; k2++) taux = taux + pow(v0(k2,k),2);
      v0.col(k) = v0.col(k)/sqrt(taux); 
      q.col(k) = q.col(k) + gammas(b)*v0.col(k);
      }
    qden = qden + gammas(b);
    if(trck){
       for(int k=0; k<nq; k++){
        (v0trck.slice(b)).col(k) = v0.col(k);
        (w0trck.slice(b)).col(k) = q.col(k)/qden;
        taux = 0;
        for(int k2=0; k2<d; k2++) taux = taux + pow(w0trck(k2,k,b),2);
        (w0trck.slice(b)).col(k) = (w0trck.slice(b)).col(k)/sqrt(taux); // normalization 
        }
    }
  }
  q = q/qden;
  for(int k=0; k<nq; k++){ // normalization of q
    taux = 0;
    for(int k2=0; k2<d; k2++) taux = taux + pow(q(k2,k),2);
    q.col(k) = q.col(k)/sqrt(taux); 
  }  
  return(
    Rcpp::List::create(
      Rcpp::Named("qden")=qden,
      Rcpp::Named("q")=q,
      Rcpp::Named("v0")=v0,
      Rcpp::Named("v0trck")=v0trck,
      Rcpp::Named("w0trck")=w0trck)
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

// [[Rcpp::export]]
arma::vec gfunCLL(arma::mat v, arma::mat X, arma::mat mu, int B, int n, int d, int nv){
  // approximation of gfunC based on B random replicates of indices from X
  // mu is a d-times-nv matrix
  // X is a d-times-n matrix (!)
  // v is a d-times-nv matrix (!)
  
  arma::vec cvec(d);
  double dW, dW0, taux;
  arma::mat X1(d,d);
  arma::vec ta(nv, arma::fill::zeros), tb(nv, arma::fill::zeros), res(nv, arma::fill::zeros);
  
  arma::imat Bs = arma::randi(B, d, arma::distr_param(0, n-1));
  for(int ib = 0; ib<B; ib++){
    for(int j=0; j<d; j++) X1.col(j) = X.col(Bs(ib,j));
      // computing the objective function
      cvec = cvectorC(X1, d);                          // vector c
      dW0 = pow(-1,d+2)*det(X1);                       // constant c0
      for(int k=0; k<nv; k++){
        dW = dW0 + dot(mu.col(k),cvec);                // (c0 + mu'c)
        taux = dot(v.col(k), cvec);                    // v'c  
        // Rcout << "taux:" << taux << std::endl;
        ta(k) = ta(k) + signC(dW)*taux;                // sign(c0+mu'c)*(v'c)
        tb(k) = tb(k) + abs(taux);                     // |v'c|
    }
  }
  for(int k=0; k<nv; k++) res(k) = ta(k)/tb(k);
  return(res);
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

// [[Rcpp::export]]
double euclNorm(arma::vec x, int d){
// Euclidean norm of a vector
  double res; 
  res = 0;
  for(int i = 0; i<d; i++) res = res + pow(x(i),2);
  res = sqrt(res);
  return(res);
}

// [[Rcpp::export]]
arma::mat spatialRankC(arma::mat mu, arma::mat X, int n, int d, int nmu){
  // mu is a d-times-nmu matrix
  // X is a d-times-n matrix (!)

  arma::mat res(d, nmu, arma::fill::zeros);
  double eN;

  for(int j = 0; j<nmu; j++){
    for(int i = 0; i<n; i++){
      eN = euclNorm(X.col(i) - mu.col(j), d);
      if(eN > arma::datum::eps) res.col(j) = res.col(j) - (X.col(i) - mu.col(j))/eN;       
    }
  }
  res = res/n;
  return(res);
}