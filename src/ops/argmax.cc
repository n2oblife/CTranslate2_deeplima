#include "ctranslate2/ops/argmax.h"

namespace ctranslate2
{
    namespace ops
    {
        /**
         * Returns the indices of the maximum value of all elements in the input
        */
        void Argmax::operator()(const StorageView& input, dim_t index)
        {
            if (input.shape().size() != 2){}
            dim_t current_max;
            for (size_t i=0; i < input.shape().size(); i++){
                
            }
        }
    } // namespace ops
    
} // namespace ctranslate2

