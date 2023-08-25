#pragma once

#include "op.h"

namespace ctranslate2 {
    namespace ops {
        
        class Argmax : public BinaryOp {
            public :
            void operator()(const StorageView& mat, dim_t index);

            private :
            void max_line(,dim_t max);
        } 
    }
}