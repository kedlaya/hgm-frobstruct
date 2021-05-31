from sage.matrix.constructor import Matrix

# Compute the formal solution matrix at t=0 for the hypergeometric connection.
cpdef formal_solution_matrix_c(H, R, int tprec):
    cdef int i, j, k, d
    K = R.base_ring()
    alpha, beta = H.alpha_beta()
    d = H.degree()
    M = Matrix(R,d,d) # zero matrix
    for j in range(d):
        b = beta[j]
        tmp = {0: K(1)}
        for i in range(d):
            if alpha[i] > b: tmp[0] *= alpha[i]-b
            if beta[i] > b: tmp[0] /= beta[i]-b
        for k in range(1,tprec):
            tmp1 = K.one()
            tmp2 = K.one()
            for i in range(d):
                tmp1 *= alpha[i]-b+k
                tmp2 *= beta[i]-b+k
            tmp[k] = tmp[k-1] * tmp1 / tmp2
        M[0,j] = R(tmp).O(tprec)
        for i in range(1,d):
            for k in range(tprec):
                tmp[k] *= (1-b+k) # apply (1-b + t*d/dt)
            M[i,j] = R(tmp).O(tprec)
    return M

# Cythonized version of sparse power series multiplication.
cpdef mult_c(a, b, int tprec):
    cdef int l1, l2, e, m
    output = {}
    e1 = a.exponents()
    c1 = a.coefficients()
    e2 = b.exponents()
    c2 = b.coefficients()
    for l1 in range(len(e1)):
        for l2 in range(len(e2)):
            e = e1[l1] + e2[l2]
            if e >= tprec:
                continue
            c = c1[l1] * c2[l2]
            if e in output:
                output[e] += c
            else:
                output[e] = c
    m = min(0, min(output.keys()))
    if m >= 0:
        return a.parent()(output)
    else:
        output2 = {}
        for e in output:
            output2[e-m] = output[e]
        return a.parent()(output2, m)

# Cythonized version of mat_mult_sigma, for efficiency at the bottleneck.
# Sage equivalent: return A * B.apply_map(lambda x: x.verschiebung(p))
cpdef mat_mult_sigma_c(A, B, p, int tprec):
    cdef int i, j, k, d, l1, l2, e, m
    d = A.dimensions()[0]
    output = {(i,k):{} for i in range(d) for k in range(d)}
    for i in range(d):
        for j in range(d):
            e1 = A[i,j].exponents()
            c1 = A[i,j].coefficients()
            for k in range(d):
                e2 = B[j,k].exponents()
                c2 = B[j,k].coefficients()
                for l1 in range(len(e1)):
                    for l2 in range(len(e2)):
                        e = e1[l1] + p*e2[l2]
                        if e >= tprec:
                            continue
                        c = c1[l1] * c2[l2]
                        if e in output[i,k]:
                            output[i,k][e] += c
                        else:
                            output[i,k][e] = c
    R = A.base_ring()
    for (i,k) in output:
        m = 0
        if output[i,k]:
            m = min(m, min(output[i,k].keys()))
        if m >= 0:
            output[i,k] = R(output[i,k])
        else:
            dt = {}
            for e in output[i,k]:
                dt[e-m] = output[i,k][e]
            output[i,k] = R(dt, m)
    return A.parent()(output)
