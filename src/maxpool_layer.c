#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"

/*
int get_padding(int size) {
	if (size%2) {
		return size/2;
	} else {
		return size/2-1;
	}
}*/

int valid(int row, int column, int rows, int columns) {
	return 0 <= row && row < rows && 0 <= column && column < columns;
}

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);
	

    // TODO: 6.1 - iterate over the input and fill in the output with max values
	int pad = (l.size%2)?l.size/2:l.size/2-1;
	for (int r = 0; r < in.rows; ++r) {
//        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
		for (int chan = 0; chan < l.channels; ++chan) {
			for (int h = -pad, y = 0; y < outh; ++y, h += l.stride) {
				for (int w = -pad, x = 0; x < outw; ++x, w += l.stride) {
					int output_index = r*(out.cols)+chan*(outw*outh)+y*(outw)+x;
					float max_value = -FLT_MAX;
					//int max_index = 0;
					for (int i = 0 ; i < l.size; ++i) {
						for (int j = 0 ; j < l.size; ++j) {
							int row = h+i;
							int column = w+j;
							int input_index = r*(in.cols)+chan*(l.width*l.height)+row*(l.width)+column;
							if (valid(row,column,l.height,l.width)) {
								float val = in.data[input_index];
								if (val > max_value) {
					//				max_index = input_index;
									max_value = val;
								}
							}
						}
					}
					out.data[output_index] = max_value;
				}
			}
		}
	}

    l.in[0] = in;
    free_matrix(l.out[0]);
    l.out[0] = out;
    free_matrix(l.delta[0]);
    l.delta[0] = make_matrix(out.rows, out.cols);
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix prev_delta: error term for the previous layer
void backward_maxpool_layer(layer l, matrix prev_delta)
{
    matrix in    = l.in[0];
    matrix out   = l.out[0];
    matrix delta = l.delta[0];

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;

    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

	int pad = (l.size%2)?l.size/2:l.size/2-1;
	for (int r = 0; r < in.rows; ++r) {
//        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
		for (int chan = 0; chan < l.channels; ++chan) {
			for (int h = -pad, y = 0; y < outh; ++y, h += l.stride) {
				for (int w = -pad, x = 0; x < outw; ++x, w += l.stride) {
					int output_index = r*(out.cols)+chan*(outw*outh)+y*(outw)+x;
					float max_value = -FLT_MAX;
					int max_index = -1;
					int has_valid = 0;
					for (int i = 0 ; i < l.size; ++i) {
						for (int j = 0 ; j < l.size; ++j) {
							int row = h+i;
							int column = w+j;
							int input_index = r*(in.cols)+chan*(l.width*l.height)+row*(l.width)+column;
							if (valid(row,column,l.width,l.height)) {
								has_valid = 1;
								float val = in.data[input_index];
								if (val > max_value) {
									max_index = input_index;
									max_value = val;
								}
							}
						}
					}
					if (max_index == -1) {
						assert(has_valid == 0);
					}
					assert(max_index != -1);
					prev_delta.data[max_index] += delta.data[output_index];
				}
			}
		}
	}
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay)
{
}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.in = calloc(1, sizeof(matrix));
    l.out = calloc(1, sizeof(matrix));
    l.delta = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

