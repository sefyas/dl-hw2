#include "uwnet.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>


const float eps = 0.000001f;

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
    for (i = 0; i < v.cols; ++i) {
        v.data[i] = v.data[i] / (x.rows * spatial);
    }
    return v;
}

matrix normalize(matrix x, matrix m, matrix v, int spatial)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - normalize array, norm = (x - mean) / sqrt(variance + eps)
    // YSS DONE!
    int i, j, index;
    float org_val, norm_val;
    for (i = 0; i < x.rows; i++) {
        for (j = 0; j < x.cols; j++) {
            index = j / spatial;
            org_val = x.data[i * x.cols + j];
            norm_val = (org_val - m.data[index]) / sqrt(v.data[index] + eps);
            norm.data[i * x.cols + j] = norm_val;
        }
    }
    return norm;
    
}

matrix batch_normalize_forward(layer l, matrix x)
{
    float s = .1;
    int spatial = x.cols / l.rolling_mean.cols;
    if (x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
    }
    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix x_norm = normalize(x, m, v, spatial);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);

    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    free_matrix(l.x[0]);
    l.x[0] = x;

    return x_norm;
}


matrix delta_mean(matrix d, matrix variance, int spatial)
{
    matrix dm = make_matrix(1, variance.cols);
    // TODO: 7.3 - calculate dL/dmean
    // YSS DONE!
    int i, j, index;
    for(i = 0; i < d.rows; i++) {
        for(j = 0; j < d.cols; j++) {
            index = j / spatial;
            dm.data[index] += -1 * d.data[i * d.cols + j] / sqrt(variance.data[index] + eps);
        }
    }
    return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial)
{
    matrix dv = make_matrix(1, variance.cols);
    // TODO: 7.4 - calculate dL/dvariance
    // YSS DONE!
    int i, j, index;
    float dist2m;
    for(i = 0; i < d.rows; i++) {
        for(j = 0; j < d.cols; j++) {
            index = j / spatial;
            dist2m = x.data[i * d.cols + j] - mean.data[index];
            dv.data[index] += -0.5 * d.data[i * d.cols + j] * dist2m 
                  / ((variance.data[index] + eps) * sqrt(variance.data[index] + eps));
        }
    }
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial)
{
    
    matrix dx = make_matrix(d.rows, d.cols);
    // TODO: 7.5 - calculate dL/dx
    // YSS DONE!
    int i, j, index;
    float dist2m;
    for(i = 0; i < d.rows; i++) {
        for(j = 0; j < d.cols; j++) {
            index = j / spatial;
            dist2m = x.data[i * d.cols + j] - mean.data[index];
            dx.data[i * d.cols + j] += d.data[i * d.cols + j] / sqrt(variance.data[index] + eps)
                                       + 2 * dv.data[index] * dist2m / (spatial * d.rows)
                                       + dm.data[index] / (spatial * d.rows);
        }
    }
    return dx;
}

matrix batch_normalize_backward(layer l, matrix d)
{
    int spatial = d.cols / l.rolling_mean.cols;
    matrix x = l.x[0];

    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix dm = delta_mean(d, v, spatial);
    matrix dv = delta_variance(d, x, m, v, spatial);
    matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}
