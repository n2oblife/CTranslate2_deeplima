#include "ctranslate2/ops/argmax.h"

namespace ctranslate2
{
    namespace ops
    {
        /**
         * Returns the indices of the maximum value of all elements in the input
        */
        void Argmax::operator()(const StorageView& input, dim_t index) const 
        {
            if (input.shape().size() != 2){}
            dim_t current_max
            for (int i=0; i < mainputt.shape(); i++){
                
            }
        }
    } // namespace ops
    
} // namespace ctranslate2

