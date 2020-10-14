#include <assert.h>
#include <math.h>
#include "uwnet.h"
#include "matrix.h"



void forward_bias(matrix m, matrix b) 
{
    assert(m.cols == b.cols);
    assert(m.rows == b.rows);

    for(int i = 0; i < m.cols * m.rows; i++) {
        b.data[i] += (m.data[i]);
    }
}

matrix forward_net(net m, matrix X) {

    matrix out = matmul(X, m.layers.w);
    forward_bias(out, m.layers.b);
    activate_matrix(out, m.layers->activation);
}
