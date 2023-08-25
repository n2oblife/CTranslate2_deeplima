#include "ctranslate2/ops/matmul.h"

#include "dispatch.h"

namespace ctranslate2 {
  namespace ops {

    MatMul::MatMul(bool trans_a, bool trans_b, float alpha)
      : _trans_a(trans_a)
      , _trans_b(trans_b)
      , _alpha(alpha) {
    }

    void MatMul::operator()(const StorageView& a, const StorageView& b, StorageView& c) const {
      PROFILE("MatMul");
      switch (a.dtype()) {
      case DataType::FLOAT32:
        DEVICE_DISPATCH(a.device(), (compute<D, float>(a, b, c)));
        break;
#ifdef CT2_WITH_CUDA
      case DataType::FLOAT16:
        if (a.device() != Device::CUDA)
          throw std::invalid_argument("FP16 MatMul is only supported on CUDA");
        compute<Device::CUDA, float16_t>(a, b, c);
        break;
#endif
      default:
        throw std::invalid_argument("MatMul: unsupported compute type " + dtype_name(a.dtype()));
      }
    }

    template <Device D, typename T>
    void MatMul::compute(const StorageView& a, const StorageView& b, StorageView& c) const {
      dim_t m, k_a;
      if (_trans_a) {
        m = a.dim(-1);
        k_a = a.dim(-2);
      } else {
        m = a.dim(-2);
        k_a = a.dim(-1);
      }

      dim_t k_b, n;
      if (_trans_b) {
        n = b.dim(-2);
        k_b = b.dim(-1);
      } else {
        n = b.dim(-1);
        k_b = b.dim(-2);
      }

      if (k_a != k_b)
        throw std::invalid_argument("MatMul: k dimension of inputs a and b should match");

      const dim_t k = k_a;
      const dim_t a_batch_size = a.size() / (m * k);
      const dim_t b_batch_size = b.size() / (k * n);

      if (a_batch_size != b_batch_size)
        throw std::invalid_argument("MatMul: batch dimension of inputs a and b should match");

      {
        Shape output_shape(a.shape());
        output_shape[output_shape.size() - 1] = n;
        output_shape[output_shape.size() - 2] = m;
        c.resize(std::move(output_shape));
      }

      const dim_t batch_size = a_batch_size;
      const dim_t lda = _trans_a ? m : k;
      const dim_t ldb = _trans_b ? k : n;
      const dim_t ldc = n;
      const float beta = 0;

      if (batch_size > 1) {
        const dim_t stridea = m * k;
        const dim_t strideb = k * n;
        const dim_t stridec = m * n;
        primitives<D>::gemm_batch_strided(_trans_a, _trans_b,
                                          m, n, k,
                                          _alpha,
                                          a.data<T>(), lda, stridea,
                                          b.data<T>(), ldb, strideb,
                                          beta,
                                          c.data<T>(), ldc, stridec,
                                          batch_size);
      } else {
        primitives<D>::gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
                            _trans_a, _trans_b,
                            m, n, k,
                            _alpha,
                            a.data<T>(), lda,
                            b.data<T>(), ldb,
                            beta,
                            c.data<T>(), ldc);
      }
    }

    /**
     * Compute the batch matrix-matrix multiplication between the batches
     * If input is a (b×n×m), b is (b×m×p), out will be (b×n×p).
    */
    void bmm(const StorageView& a, const StorageView& b, StorageView& out) {
      Shape a_shape(a.shape()), b_shape(b.shape());
      if (a_shape.size() != 3 || b_shape.size() != 3){
        throw std::invalid_argument("MatMul: the matrix to be computed must be have 3 dim");
      }
      if (a_shape[0] != b_shape[0]){
        throw std::invalid_argument("MatMul: the matrix to be computed must be have have the same number of batch b");
      }
      if (a_shape[2] != b_shape[1]){
        throw std::invalid_argument("MatMul: the shape of the matrix to be computed is good. (b*n*m) * (b*m*p) -> (b*n*p). ");
      }
      Shape out_shape({a_shape[0], a_shape[1], b_shape[2]});
      out.resize(std::move(out_shape));
      StorageView res(a.dtype(), a.device());
      ops::MatMul mm();
      for (int i=0; i<a_shape[0]; i++){
        mm(, , res); // TODO compute the matmul
        out;
        res.clear();
      }
    }

  }
}
