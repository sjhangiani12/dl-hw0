#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix m: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
void forward_bias(matrix m, matrix b)
{
    assert(b.rows == 1);
    assert(m.cols == b.cols);
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            m.data[i*m.cols + j] += b.data[j];
        }
    }
}

// Calculate bias updates from a delta matrix
// matrix delta: error made by the layer
// matrix db: delta for the biases
void backward_bias(matrix delta, matrix db)
{
    int i, j;
    for(i = 0; i < delta.rows; ++i){
        for(j = 0; j < delta.cols; ++j){
            db.data[j] += -delta.data[i*delta.cols + j];
        }
    }
}

// Run a connected layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer: f(wx + b)
matrix forward_connected_layer(layer l, matrix in)
{
    // implemented forward connected layer functionality
    matrix out = matmul(in, l.w);
    forward_bias(out, l.b);
    activate_matrix(out, l.activation);

    // Saving our input and output and making a new delta matrix to hold errors
    // Probably don't change this
    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a connected layer backward
// layer l: layer to run
// matrix delta: 
void backward_connected_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    // printf("Dimensions of in: %d rows by %d cols\n", in.rows, in.cols);
    // printf("Dimensions of out: %d rows by %d cols\n", out.rows, out.cols);
    // printf("Dimensions of delta: %d rows by %d cols\n", delta.rows, delta.cols);
    // printf("Dimensions of w: %d rows by %d cols\n", l.w.rows, l.w.cols);
    // printf("Dimensions of dw: %d rows by %d cols\n", l.dw.rows, l.dw.cols);
    // printf("Dimensions of b: %d rows by %d cols\n", l.b.rows, l.b.cols);
    // printf("Dimensions of db: %d rows by %d cols\n", l.db.rows, l.db.cols);
    // printf("Dimensions of prev_delta: %d rows by %d cols\n", prev_delta.rows, prev_delta.cols);

    // TODO: 3.2
    // delta is the error made by this layer, dL/dout
    // First modify in place to be dL/d(in*w+b) using the gradient of activation
    gradient_matrix(out, l.activation, delta);
    
    // Calculate the updates for the bias terms using backward_bias
    // The current bias deltas are stored in l.db
    backward_bias(delta, l.db);

    // Then calculate dL/dw. Use axpy to subtract this dL/dw into any previously stored
    // updates for our weights, which are stored in l.dw
    // l.dw = l.dw - dL/dw
    
    matrix dL_dw = matmul(transpose_matrix(in), delta);
    // printf("Dimensions of dL_dw: %d rows by %d cols\n", dL_dw.rows, dL_dw.cols);
    axpy_matrix(-1, dL_dw,l.dw);
    free_matrix(dL_dw);

    if(prev_delta.data){
        // Finally, if there is a previous layer to calculate for,
        // calculate dL/d(in). Again, using axpy, add this into the current
        // value we have for the previous layers delta, prev_delta.
        
        matrix dL_dIn = matmul(delta, transpose_matrix(l.w));
        // printf("Dimensions of dL_dIn: %d rows by %d cols\n", dL_dIn.rows, dL_dIn.cols);
        axpy_matrix(1,dL_dIn, prev_delta);
        free_matrix(dL_dIn);
    }
}

// Update 
void update_connected_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 3.3
    // Currently l.dw and l.db store:
    //l.dw = momentum * l.dw_prev - dL/dw
    //l.db = momentum * l.db_prev - dL/db

    // For our weights we want to include weight decay:
    // l.dw = l.dw - decay * l.w
    axpy_matrix(-1*decay, l.w, l.dw);

    // Then for both weights and biases we want to apply the updates:
    // l.w = l.w + rate*l.dw
    // l.b = l.b + rade*l.db
    axpy_matrix(rate, l.dw, l.w);
    axpy_matrix(rate, l.db, l.b);

    // Finally, we want to scale dw and db by our momentum to prepare them for the next round
    //l.dw *= momentum
    //l.db *= momentum

    for(int i = 0; i < l.dw.cols * l.dw.rows; i++) {
        l.dw.data[i] *= momentum;
    }

    for(int i = 0; i < l.db.cols * l.db.rows; i++) {
        l.db.data[i] *= momentum;
    }
}

layer make_connected_layer(int inputs, int outputs, ACTIVATION activation)
{
    layer l = {0};
    l.w  = random_matrix(inputs, outputs, sqrtf(2.f/inputs));
    l.dw = make_matrix(inputs, outputs);
    l.b  = make_matrix(1, outputs);
    l.db = make_matrix(1, outputs);
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.activation = activation;
    l.forward  = forward_connected_layer;
    l.backward = backward_connected_layer;
    l.update   = update_connected_layer;
    return l;
}

