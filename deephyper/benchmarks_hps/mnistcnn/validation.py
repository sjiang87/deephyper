import math

def is_valid(params):
    f1 = params["f1_size"]
    f2 = params["f2_size"]
    padding1 = params["padding_c1"]
    padding2 = params["padding_c2"]
    pool_padding1 = params["padding_p1"]
    pool_padding2 = params["padding_p2"]
    p = params['p_size']
    maxpool = params['max_pool']

    i1 = 28

    if padding1 == 'valid':
        out1 = i1 - f1 + 1
        i2 = out1
        out2 = i2 - f1 + 1
    else:
        out1 = i1
        i2 = out1
        out2 = i2

    i3 = out2

    if not maxpool:
        out3 = i3
    else:
        if pool_padding1 == 'valid':
            out3 = math.floor(((i3-p)/p)+1)
        else:
            out3 = math.ceil(((i3-p)/p)+1)

    i4 = out3

    if padding2 == 'valid':
        out4 = i4 - f2 + 1
        i5 = out4
        out5 = i5 - f2 + 1
    else:
        out4 = i4
        i5 = out4
        out5 = i5

    i6 = out5

    if padding2 == 'valid':
        if not maxpool:
            if i1 >= f1 and i2 >=f1 and i4 >= f1 and i5 >= f2:
                return True
            else:
                return False
        else:
            if pool_padding2 == 'valid':
                if i1 >= f1 and i2 >=f1 and i3 >= p and i4 >= f2 and i5 >= f2 and i6 >= p:
                    return True
                else:
                    return False
            else:
                if i1 >= f1 and i2 >=f1 and i3 >= p and i4 >= f2 and i5 >= f2:
                    return True
                else:
                    return False
    else:
        if not maxpool:
            if i1 >= f1 and i2 >=f1 and i4 >= f1:
                return True
            else:
                return False
        else:
            if pool_padding2 == 'valid':
                if i1 >= f1 and i2 >=f1 and i3 >= p and i6 >= p:
                    return True
                else:
                    return False
            else:
                if i1 >= f1 and i2 >=f1 and i3 >= p:
                    return True
                else:
                    return False
