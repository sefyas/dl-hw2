#include <stdio.h>
#include <math.h>
#include <float.h>
#include "matrix.h"

matrix mean(matrix x, int spatial);
matrix variance(matrix x, matrix m, int spatial);
matrix normalize(matrix x, matrix m, matrix v, int spatial);

int main(int argc, char **argv)
{
    int rows = 2; int cols = 12; int spatial = 4;

    matrix x = make_matrix(rows, cols);
    int i, j;
    int val = 1;
    for(i = 0; i < x.rows; i++)
    	for(j = 0; j < x.cols; j++)
    		x.data[i * x.cols + j] = val++;
    print_matrix(x);

    matrix m = mean(x, spatial);
    print_matrix(m);

    matrix v = variance(x, m, spatial);
    print_matrix(v);

    matrix x_norm = normalize(x, m, v, spatial);
    print_matrix(x_norm);

    free_matrix(x);
    free_matrix(m);
    free_matrix(v);
    free_matrix(x_norm);

    return 0;
}

matrix mean(matrix x, int spatial)
{
    matrix m = make_matrix(1, x.cols/spatial);
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/spatial] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / spatial;
    }
    return m;
}

matrix variance(matrix x, matrix m, int spatial)
{
    matrix v = make_matrix(1, x.cols/spatial);
    // TODO: 7.1 - calculate variance
    // YSS DONE!
    int i, j;
    float dist2m;
    for (i = 0; i < x.rows; i++)
        for (j = 0; j < x.cols; j++) {
            dist2m = x.data[i * x.cols + j] - m.data[j / spatial];
            v.data[j / spatial] += dist2m * dist2m;
        }
    for(i = 0; i < v.cols; ++i){
        v.data[i] = v.data[i] / x.rows / spatial;
    }
    return v;
}

matrix normalize(matrix x, matrix m, matrix v, int spatial)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - normalize array, norm = (x - mean) / sqrt(variance + eps)
    // YSS
    int i, j, index;
    float org_val, norm_val;
    for (i = 0; i < x.rows; i++) {
        for (j = 0; j < x.cols; j++) {
            index = j / spatial;
            org_val = x.data[i * x.cols + j];
            norm_val = (org_val - m.data[index]) / sqrt(v.data[index] + FLT_MIN);
            norm.data[i * x.cols + j] = norm_val;
            //printf("(%2d, %2d) = %2.2f, M(%2d) = %2.2f, S(%2d)= %2.2f --> norm = %2.2f\n",
            //	i, j, org_val, index, m.data[index], index, v.data[index], norm_val);
        }
    }
    return norm;
    
}