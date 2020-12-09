
def dim_pool(i, k, s):
    return int((i-k)/s)+1


def dim_trans_conv(i, k, s, p):
    return s*(i - 1) + k - 2 * p


i = 120
k = 2
s = 2
p = 0

p1 = dim_pool(i, k, s)
p2 = dim_pool(p1, k, s)
p3 = dim_pool(p2, k, s)
p4 = dim_trans_conv(p3, k, s, p)
p5 = dim_trans_conv(p4, k, s, p)
p6 = dim_trans_conv(p5, k, s, p)

