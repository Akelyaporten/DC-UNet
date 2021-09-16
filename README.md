# DC-UNet

Implementation of DC-UNet model in tensorflow.

URL of paper = https://arxiv.org/ftp/arxiv/papers/2006/2006.00414.pdf

In that implementation ,contrary to original paper, UpSample2D() blocks are replaced with Conv2DTranspose() to parameterize up sampling layer.

Also at the last layer instead of sigmoid function, tanh function is used.
