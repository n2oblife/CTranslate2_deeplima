#include "ctranslate2/primitives/cpu_mkl.h"

#include <vector>

#include <mkl.h>

namespace ctranslate2 {
  namespace primitives {

    template<>
    void copy(const float* x, float* y, size_t size) {
      cblas_scopy(size, x, 1 /* incx */, y, 1 /* incy */);
    }

    template<>
    void add(const float* a, const float* b, float* c, size_t size) {
      vsAdd(size, a, b, c);
    }

    template<>
    void sub(const float* a, const float* b, float* c, size_t size) {
      vsSub(size, a, b, c);
    }

    template<>
    void mul(float a, float* y, size_t size) {
      cblas_sscal(size, a, y, 1 /* incx */);
    }

    template<>
    void mul(const float* a, const float* b, float* c, size_t size) {
      vsMul(size, a, b, c);
    }

    template<>
    void inv(const float* x, float* y, size_t size) {
      vsInv(size, x, y);
    }

    template<>
    void pow(const float* x, float *y, float power, size_t size) {
      vsPowx(size, x, power, y);
    }

    template<>
    void exp(const float* x, float* y, size_t size) {
      vmsExp(size, x, y, VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    }

    template<>
    void tanh(const float* x, float* y, size_t size) {
      vsTanh(size, x, y);
    }

    template<>
    void transpose_2d(const float* a, const size_t* dims, float* b) {
      auto rows = dims[0];
      auto cols = dims[1];
      mkl_somatcopy('R', 'T', rows, cols, 1.0, a, cols, b, rows);
    }

    template<>
    void gemm(const float* a, const float* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              float alpha, float beta,
              float* c) {
      MKL_INT lda = transpose_a ? m : k;
      MKL_INT ldb = transpose_b ? k : n;
      MKL_INT ldc = n;

      MKL_INT m_ = m;
      MKL_INT n_ = n;
      MKL_INT k_ = k;

      CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

      cblas_sgemm(CblasRowMajor,
                  trans_a, trans_b,
                  m_, n_, k_,
                  alpha, a, lda,
                  b, ldb,
                  beta, c, ldc);
    }

    template<>
    void gemm(const int16_t* a, const int16_t* b,
              bool transpose_a, bool transpose_b,
              size_t m, size_t n, size_t k,
              int16_t alpha, int32_t beta,
              int32_t* c) {
      MKL_INT lda = transpose_a ? m : k;
      MKL_INT ldb = transpose_b ? k : n;
      MKL_INT ldc = n;

      MKL_INT m_ = m;
      MKL_INT n_ = n;
      MKL_INT k_ = k;

      CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;
      CBLAS_OFFSET offsetc = CblasFixOffset;

      MKL_INT16 oa = 0;
      MKL_INT16 ob = 0;
      MKL_INT32 oc = 0;

      cblas_gemm_s16s16s32(CblasRowMajor,
                           trans_a, trans_b,
                           offsetc, m_, n_, k_,
                           static_cast<float>(alpha),
                           reinterpret_cast<const MKL_INT16*>(a), lda, oa,
                           reinterpret_cast<const MKL_INT16*>(b), ldb, ob,
                           static_cast<float>(beta),
                           reinterpret_cast<MKL_INT32*>(c), ldc, &oc);
    }

    template<>
    void gemm_batch(const float* a, const float* b,
                    bool transpose_a, bool transpose_b,
                    size_t batch_size,
                    size_t m, size_t n, size_t k,
                    float alpha, float beta,
                    float* c) {
      MKL_INT lda = transpose_a ? m : k;
      MKL_INT ldb = transpose_b ? k : n;
      MKL_INT ldc = n;

      MKL_INT b_ = batch_size;
      MKL_INT m_ = m;
      MKL_INT n_ = n;
      MKL_INT k_ = k;

      CBLAS_TRANSPOSE trans_a = transpose_a ? CblasTrans : CblasNoTrans;
      CBLAS_TRANSPOSE trans_b = transpose_b ? CblasTrans : CblasNoTrans;

      std::vector<const float*> a_array(batch_size);
      std::vector<const float*> b_array(batch_size);
      std::vector<float*> c_array(batch_size);
      for (MKL_INT i = 0; i < b_; ++i) {
        a_array[i] = a + (i * m_ * k_);
        b_array[i] = b + (i * k_ * n_);
        c_array[i] = c + (i * m_ * n_);
      }

      cblas_sgemm_batch(CblasRowMajor,
                        &trans_a, &trans_b,
                        &m_, &n_, &k_,
                        &alpha, a_array.data(), &lda,
                        b_array.data(), &ldb,
                        &beta, c_array.data(), &ldc,
                        1 /* group_count */, &b_);
    }

  }
}