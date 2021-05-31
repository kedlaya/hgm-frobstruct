## TODO: compute precision cutoffs:
## -- power of t^p-1 needed to clear denominators
## -- t-adic precision needed in Frobenius structure

from sage.modular.hypergeometric_motive import HypergeometricData as HGData
load("frob_matrices.pyx")

# Construct the Frobenius degeneration at t=0, using the p-adic Gamma function as per Dwork.
def initial_Frobenius_matrix(H,R,p,pprec=20):
    alpha, beta = H.alpha_beta()
    K = R.base_ring()
    perm = [beta.index(frac(p*b)) for b in beta]
    m = min(H.zigzag(b) for b in beta)
    c = K(-1) * prod(K(b).gamma() for b in beta) / prod(K(a).gamma() for a in alpha)
    d = H.degree()
    M = Matrix(R,d,d)
    for j in range(H.degree()):
        i = perm[j]
        u = c * K((-1)^H.zigzag(beta[i]))
        u <<= (H.zigzag(beta[j])-m)
        u *= prod(K(frac(a-beta[i])).gamma() for a in alpha)
        u /= prod(K(frac(b-beta[i])).gamma() for b in beta)
        M[i,j] = R(u) << (-p+1+floor(p*beta[j]))
    return M

# Form the Frobenius structure as a power series about t=0.
def frobenius_structure(H, R, p, tprec, pprec):
    U = formal_solution_matrix_c(H,R,tprec+p)
    F0 = initial_Frobenius_matrix(H,R,p,pprec)
    F1 = mat_mult_sigma_c(U, F0, 1, tprec+p)
    Utrunc = U.apply_map(lambda x: x.O(tprec//p+1))
    return mat_mult_sigma_c(F1, ~Utrunc, p, tprec)

def frobenius_structure_as_rational_function(H, p, final_pprec, tprec=None):
    d = H.degree()
    alpha,beta = H.alpha_beta()
    if len(set(beta)) < d:
        if len(set(alpha)) == d:
            return frobenius_structure_as_rational_function(H.swap_alpha_beta(), p, final_pprec, tprec)
        else:
            raise ValueError("Repeated values in beta not allowed")
    if any(x.valuation(p) < 0 for x in alpha+beta):
        raise ValueError("Wild primes not allowed")

    w = H.weight()
    # Next step: control p-adic valuation of F and F^{-1}; conjecture below.
    denom_val = max(sum(valuation(a-b, p) for a in alpha) for b in beta)
    # Then: figure out the powers of t^p-1 and t-1 as per K-Tuitman.
    tppow = max(0, ceil(final_pprec-sum(alpha)-d+sum(beta)))
    tpow = max(tppow, final_pprec+denom_val*2)
    if tprec is None: tprec=p*ceil(1+tpow*(1+1/log(p)))
    pprec=ceil(tpow*(1+4/log(p)))
    print('final_pprec={}; denom_val={}; tppow={}; tpow={}; pprec={}; tprec={}'.format(final_pprec, denom_val, tppow, tpow, pprec, tprec))

    K = Qp(p, pprec)
    R.<t> = LaurentSeriesRing(K, tprec+p, sparse=True)
    F = frobenius_structure(H, R, p, tprec, pprec)
    u = (t-1)^(tpow-tppow) * (t^p-1)^(tppow)
    F1 = F.apply_map(lambda x: mult_c(x, u, tprec))
    u1 = u.laurent_polynomial()
    F2 = F.apply_map(lambda x: x.laurent_polynomial() / u1)
    return F2

def euler_factor_list(H,t0list,p,pprec=None,tprec=None,check_FE=False):
    d = H.degree()
    alpha,beta = H.alpha_beta()
    if len(set(beta)) < d:
        if len(set(alpha)) == d:
            return euler_factor_list(H.swap_alpha_beta(), [~t for t in t0list], p, pprec, tprec, check_FE)
        else:
            raise ValueError("Repeated values in beta not allowed")
    if any(x.valuation(p) < 0 for x in alpha+beta):
        raise ValueError("Wild primes not allowed")
    if any(t0.valuation(p) < 0 or t0%p in (0,1) for t0 in t0list):
        raise ValueError("Tame primes not allowed")

    w = H.weight()
    ## The following bound is not best possible; use WeilPolynomials to optimize.
    final_cp_prec = [ceil(i*w/2 + log(2*binomial(d,i)+1,p)) for i in range(d+1)]
    if check_FE:
        final_pprec = max(final_cp_prec)
    else:
        final_pprec = final_cp_prec[floor(d/2)] 
    # Next step: control p-adic valuation of F and F^{-1}; conjecture below.
    denom_val = max(sum(valuation(a-b, p) for a in alpha) for b in beta)
    # Then: figure out the powers of t^p-1 and t-1 as per K-Tuitman.
    tppow = max(0, ceil(final_pprec-sum(alpha)-d+sum(beta)))
    tpow = max(tppow, final_pprec+denom_val*2)
    if tprec is None: tprec=p*ceil(1+tpow*(1+1/log(p)))
    if pprec is None: pprec=ceil(tpow*(1+4/log(p)))
    print('final_pprec={}; denom_val={}; tppow={}; tpow={}; pprec={}; tprec={}'.format(final_pprec, denom_val, tppow, tpow, pprec, tprec))

    K = Qp(p, pprec)
    R.<t> = LaurentSeriesRing(K, tprec+p, sparse=True)
    F = frobenius_structure(H, R, p, tprec, pprec)
    u = (t-1)^(tpow-tppow) * (t^p-1)^(tppow)
    F1 = F.apply_map(lambda x: mult_c(x, u, tprec))
    ans = []
    for t0 in t0list:
        arg = K.teichmuller(t0)
        u1 = u(arg)
        F2 = F1.apply_map(lambda x: x(arg)/u1)
        cp = F2.charpoly()
        if check_FE:
            l = [IntegerModRing(p^final_cp_prec[i])(cp[d-i]).lift_centered() for i in range(d)]
        else:
            l = [IntegerModRing(p^final_cp_prec[i])(cp[d-i]).lift_centered() for i in range(d//2+1)]
            sg = H.sign(t0, p)
            if d%2==0:
                l += [l[-i-1]*p^(w*i)*sg for i in range(1, d//2+1)]
            else:
                l += [l[-i]*p^(w*(i-1/2))*sg for i in range(1, d//2+2)]
        ans.append(PolynomialRing(ZZ,name='T')(l))
        if not check_FE and not ans[-1].reverse().is_weil_polynomial():
            raise RuntimeError("did not obtain a Weil polynomial ({})".format(ans[-1]))
    return(ans)

def euler_factor(H,t0,p,pprec=None,tprec=None,check_FE=False):
    return euler_factor_list(H,[t0],p,pprec,tprec,check_FE)[0]