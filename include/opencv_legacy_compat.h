// include/opencv_legacy_compat.h
#pragma once
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv_legacy {

// --------- Alias de tipo ---------
using CvMat = cv::Mat;

// --------- Constantes “antiguas” mapeadas ---------
constexpr int CV_RGB2GRAY  = cv::COLOR_RGB2GRAY;
constexpr int CV_BGR2GRAY  = cv::COLOR_BGR2GRAY;
constexpr int CV_RGBA2GRAY = cv::COLOR_RGBA2GRAY;
constexpr int CV_BGRA2GRAY = cv::COLOR_BGRA2GRAY;
constexpr int CV_GRAY2BGR  = cv::COLOR_GRAY2BGR;

// Flags de SVD “fantasma” para mantener firmas antiguas
constexpr int CV_SVD            = 0;
constexpr int CV_SVD_MODIFY_A   = 1;
constexpr int CV_SVD_U_T        = 2;
constexpr int CV_SVD_V_T        = 4;

// --------- Construcción / liberación de matrices estilo C ---------
inline CvMat  cvMat(int rows, int cols, int type, void* data) {
  return CvMat(rows, cols, type, data);
}
inline CvMat* cvCreateMat(int rows, int cols, int type) {
  return new CvMat(rows, cols, type);
}
inline void   cvReleaseMat(CvMat** M) {
  if (M && *M) { delete *M; *M = nullptr; }
}

// --------- Acceso equivalente a cvmGet/cvmSet ---------
inline double cvmGet(const CvMat& M, int r, int c) { return M.at<double>(r,c); }
inline void   cvmSet(CvMat& M, int r, int c, double v) { M.at<double>(r,c)=v; }
inline double cvmGet(const CvMat* M, int r, int c) { assert(M); return M->at<double>(r,c); }
inline void   cvmSet(CvMat* M, int r, int c, double v) { assert(M); M->at<double>(r,c)=v; }

// --------- Solve / Invert (SVD) ---------
inline void cvSolve(const CvMat& A, const CvMat& b, CvMat& x, int /*flags*/=CV_SVD) {
  cv::solve(A,b,x,cv::DECOMP_SVD);
}
inline void cvSolve(const CvMat* A, const CvMat* b, CvMat* x, int /*flags*/=CV_SVD) {
  assert(A && b && x); cv::solve(*A,*b,*x,cv::DECOMP_SVD);
}
inline void cvInvert(const CvMat& A, CvMat& Ai, int /*method*/=CV_SVD) {
  cv::invert(A,Ai,cv::DECOMP_SVD);
}
inline void cvInvert(const CvMat* A, CvMat* Ai, int /*method*/=CV_SVD) {
  assert(A && Ai); cv::invert(*A,*Ai,cv::DECOMP_SVD);
}

// --------- SVD con U^T/V^T (emulación) ---------
inline void cvSVD(const CvMat& A, CvMat& w, CvMat& Ut, CvMat* Vt=nullptr, int /*flags*/=0) {
  cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  w  = svd.w.clone();
  Ut = svd.u.t();              // emula CV_SVD_U_T
  if (Vt) *Vt = svd.vt.clone(); // emula CV_SVD_V_T
}
inline void cvSVD(const CvMat* A, CvMat* w, CvMat* Ut, CvMat* Vt=nullptr, int /*flags*/=0) {
  assert(A && w && Ut);
  cv::SVD svd(*A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
  *w  = svd.w.clone();
  *Ut = svd.u.t();
  if (Vt) *Vt = svd.vt.clone();
}

// --------- Operaciones auxiliares ---------
inline void cvMulTransposed(const CvMat& A, CvMat& Dst, int aTa) {
  cv::mulTransposed(A, Dst, aTa!=0);
}
inline void cvMulTransposed(const CvMat* A, CvMat* Dst, int aTa) {
  assert(A && Dst); cv::mulTransposed(*A, *Dst, aTa!=0);
}
inline void cvSetZero(CvMat& M) { M.setTo(0); }
inline void cvSetZero(CvMat* M) { assert(M); M->setTo(0); }

// Puntero “tipo CvMat->data.db” (dobles) para cambios mecánicos mínimos
inline double* dbptr(CvMat* M) { assert(M); return reinterpret_cast<double*>(M->data); }
inline const double* dbptr(const CvMat* M){ assert(M); return M->ptr<double>(0); }

} // namespace cv_legacy