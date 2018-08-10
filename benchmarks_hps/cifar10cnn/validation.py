import math

def is_valid(params):
    f1 = params["f1_size"]
    f2 = params["f2_size"]
    s1 = params['stride1']
    s2 = params['stride2']
    padding_c1 = params["padding_c1"]
    padding_c2 = params["padding_c2"]
    p = params['p_size']
    padding_p1 = params['padding_p1']
    padding_p2 = params['padding_p2']
    
    i1 = 28

    if padding_c1 == 'same':
        out1 = math.floor((i1-1)/s1) + 1
        i2 = out1
        out2 = math.floor((i2-1)/s1) + 1
        i3 = out2
    else:
        out1 = math.floor((i1-f1)/s1) + 1
        i2 = out1
        out2 = math.floor((i2-f1)/s1) + 1
        i3 = out2

    if padding_p1 == 'same':
        out3 = math.ceil((i3-p)/p) + 1
        i4 = out3
    else:
        out3 = math.floor((i3-p)/p) + 1
        i4 = out3

    if padding_c2 == 'same':
        out4 = math.floor((i4-1)/s2) + 1
        i5 = out4
        out5 = math.floor((i5-1)/s2) + 1
        i6 = out5
    else:
        out4 = math.floor((i4-f2)/s2) + 1
        i5 = out4
        out5 = math.floor((i5-f2)/s2) + 1
        i6 = out5

    if padding_c2 == 'valid':
            if padding_p2 == 'valid':
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
            if padding_p2 == 'valid':
                if i1 >= f1 and i2 >=f1 and i3 >= p and i6 >= p:
                    return True
                else:
                    return False
            else:
                if i1 >= f1 and i2 >=f1 and i3 >= p:
                    return True
                else:
                    return False
    


    




