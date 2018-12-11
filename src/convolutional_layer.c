#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix m: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
void forward_convolutional_bias(matrix m, matrix b)
{
    assert(b.rows == 1);
    assert(m.cols % b.cols == 0);
    int spatial = m.cols / b.cols;
    int i,j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            m.data[i*m.cols + j] += b.data[j/spatial];
        }
    }
}

// Calculate bias updates from a delta matrix
// matrix delta: error made by the layer
// matrix db: delta for the biases
void backward_convolutional_bias(matrix delta, matrix db)
{
    assert(db.rows == 1);
    assert(delta.cols % db.cols == 0);
    int spatial = delta.cols / db.cols;
    int i,j;
    for(i = 0; i < delta.rows; ++i){
        for(j = 0; j < delta.cols; ++j){
            db.data[j/spatial] += delta.data[i*delta.cols + j];
        }
    }
}


float im2col_get_pixel(image im, int x, int y, int c)
{

	if(x >= im.w) return 0;
	if(y >= im.h) return 0;
	if(x < 0) return 0;
	if(y < 0) return 0;

	/*
	   if(x >= im.w) x = im.w - 1;
	   if(y >= im.h) y = im.h - 1;
	   if(x < 0) x = 0;
	   if(y < 0) y = 0;
	 */
	assert(c >= 0);
	assert(c < im.c);
	return im.data[x + im.w*(y + im.h*c)];
}

void col2im_addto_pixel(image im, int x, int y, int c, float val) {
	if(x >= im.w) return ;
	if(y >= im.h) return ;
	if(x < 0) return ;
	if(y < 0) return ;
	im.data[x + im.w*(y + im.h*c)] += val;
}

// Make a column matrix out of an image
// image im: image to process
// int size: kernel size for convolution operation
// int stride: stride for convolution
// returns: column matrix
matrix im2col(image im, int size, int stride)
{
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;
    int cols = outw * outh;
    matrix col = make_matrix(rows, cols);

    // TODO: 5.1 - fill in the column matrix
	// Check if works correctly
	int pad = (size%2)?size/2:size/2-1;
	for (int chan = 0; chan < im.c; ++chan) 
	{
		for (int h = -pad, y = 0; y < outh ; h += stride, ++y) {
			for (int w = -pad, x = 0; x < outw; w += stride, ++x) {
				for (int i = 0; i < size; ++i) {
					for (int j = 0; j < size; ++j) {
						int row = h+i; // row number in image
						int column = w+j; // column number in image
						col.data[cols*(size*size)*chan + cols*((size)*i + j) + y*outw + x] = im2col_get_pixel(im, column, row, chan);
					}
				}
			}
		}
	}

    return col;
}

// The reverse of im2col, add elements back into image
// matrix col: column matrix to put back into image
// int size: kernel size
// int stride: convolution stride
// image im: image to add elements back into
void col2im(matrix col, int size, int stride, image im)
{
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;
    int cols = outw * outh;

    // TODO: 5.2 - add values into image im from the column matrix
	int pad = (size%2)?size/2:size/2-1;
	for (int chan = 0; chan < im.c; ++chan) {
		for (int h = -pad, y = 0; y < outh; h += stride, ++y) {
			for (int w = -pad, x = 0; x < outw; w += stride, ++x) {
				for (int i = 0; i < size; ++i) {
					for (int j = 0; j < size; ++j) {
						int row = h+i; // row number in image
						int column = w+j; // column number in image
						float val = col.data[cols*(size*size)*chan + cols*((size)*i + j) + y*outw + x];
						col2im_addto_pixel(im, column, row, chan, val);
					}
				}
			}
		}
	}	
}

// Run a convolutional layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_convolutional_layer(layer l, matrix in)
{
    int i, j;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.filters);
    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        matrix wx = matmul(l.w, x);
        for(j = 0; j < wx.rows*wx.cols; ++j){
            out.data[i*out.cols + j] = wx.data[j];
        }
        free_matrix(x);
        free_matrix(wx);
    }
	
    // YSS DONE!
    if(l.batchnorm) {
        matrix xnorm = batch_normalize_forward(l, out);
        out = xnorm;
    }

    forward_convolutional_bias(out, l.b);
    activate_matrix(out, l.activation);

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a convolutional layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_convolutional_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    gradient_matrix(out, l.activation, delta);
    backward_convolutional_bias(delta, l.db);

	// YSS DONE!
    if(l.batchnorm){
        matrix dx = batch_normalize_backward(l, delta);
        free_matrix(delta);
        l.delta[0] = delta = dx;
    }

    int i;
    matrix wt = transpose_matrix(l.w);
    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        image dexample = float_to_image(prev_delta.data + i*in.cols, l.width, l.height, l.channels);
        assert(in.cols == l.width*l.height*l.channels);

        delta.rows = l.filters;
        delta.cols = outw*outh;
        delta.data = l.delta[0].data + i*delta.rows*delta.cols;

        matrix x = im2col(example, l.size, l.stride);
        matrix xt = transpose_matrix(x);
        matrix dw = matmul(delta, xt);
        axpy_matrix(1, dw, l.dw);
        
        if(prev_delta.data){
            matrix col = matmul(wt, delta);
            col2im(col, l.size, l.stride, dexample);
            free_matrix(col);
        }
        free_matrix(x);
        free_matrix(xt);
        free_matrix(dw);
    }
    free_matrix(wt);

}

// Update convolutional layer
// layer l: layer to update
// float rate: learning rate
// float momentum: momentum term
// float decay: l2 regularization term
void update_convolutional_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 5.3 Update the weights, similar to the connected layer.
	// update bias
	axpy_matrix(rate, l.db, l.b);
	
	// suppose that before this operation l.dw = -dL/dw + m*delta(prev)
	axpy_matrix(-decay, l.w, l.dw); // lambda*w
	axpy_matrix(rate, l.dw, l.w);
	// now we apply momentum
	scal_matrix(momentum, l.dw);
	scal_matrix(momentum, l.db);
}

// Make a new convolutional layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of convolutional filter to apply
// int stride: stride of operation
layer make_convolutional_layer(int w, int h, int c, int filters, int size, int stride, ACTIVATION activation)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.filters = filters;
    l.size = size;
    l.stride = stride;
    l.w  = random_matrix(filters, size*size*c, sqrtf(2.f/(size*size*c)));
    l.dw = make_matrix(filters, size*size*c);
    l.b  = make_matrix(1, filters);
    l.db = make_matrix(1, filters);
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.activation = activation;
    l.forward  = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update   = update_convolutional_layer;
	
    // YSS DONE!
    l.x = calloc(1, sizeof(matrix));
    l.rolling_mean = make_matrix(1, c);
    l.rolling_variance = make_matrix(1, c);
    
    return l;
}

