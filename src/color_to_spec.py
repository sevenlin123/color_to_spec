import numpy as np
from scipy.optimize import minimize

def normalize_to_band(ref, ref_err, band_idx):
    ref_ref = ref[band_idx]
    ref_ref_err = ref_err[band_idx]
    
    ref_norm = []
    ref_err_norm = []
    for k in range(len(ref)):
        if k == band_idx:
            ref_norm.append(1.0)
            ref_err_norm.append(0.0)
        else:
            val_norm = ref[k] / ref_ref
            # Error propagation for division:
            # sig_y = y * sqrt((sig_a/a)^2 + (sig_b/b)^2)
            if ref[k] > 0 and ref_ref > 0:
                rel_err_sq = (ref_err[k] / ref[k])**2 + (ref_ref_err / ref_ref)**2
                err_norm = val_norm * (rel_err_sq ** 0.5)
            else:
                err_norm = 0.0
            ref_norm.append(val_norm)
            ref_err_norm.append(err_norm)
            
    return tuple(ref_norm), tuple(ref_err_norm)

def color_to_spec(gr, ri, rz, gr_err, ri_err, rz_err, norm_band='i'):
    gr_sun = 0.45
    gr_sun_err = 0.02
    gi_sun = 0.57
    gi_sun_err = 0.02
    gz_sun = 0.61
    gz_sun_err = 0.03
    
    g_flux_sun = 10**(-1/2.5)
    r_flux_sun = 10**(-(1-gr_sun)/2.5)
    i_flux_sun = 10**(-(1-gi_sun)/2.5)
    z_flux_sun = 10**(-(1-gz_sun)/2.5)
    
    r_flux_sun_err = (r_flux_sun**2 * (1/2.5 *np.log(10) * gr_sun_err)**2) ** 0.5
    i_flux_sun_err = (i_flux_sun**2 * (1/2.5 *np.log(10) * gi_sun_err)**2) ** 0.5
    z_flux_sun_err = (z_flux_sun**2 * (1/2.5 *np.log(10) * gz_sun_err)**2) ** 0.5 


    gi = gr + ri
    gz = gr + rz

    gi_err = (gr_err**2 + ri_err**2)**0.5
    gz_err = (gr_err**2 + rz_err**2)**0.5

    g_flux = 10**(-1/2.5)
    r_flux = 10**(-(1-gr)/2.5)
    i_flux = 10**(-(1-gi)/2.5)
    z_flux = 10**(-(1-gz)/2.5) 

    r_flux_err = (r_flux**2 * (1/2.5 *np.log(10) * gr_err)**2) ** 0.5
    i_flux_err = (i_flux**2 * (1/2.5 *np.log(10) * gi_err)**2) ** 0.5
    z_flux_err = (z_flux**2 * (1/2.5 *np.log(10) * gz_err)**2) ** 0.5 

    ref = (g_flux/g_flux_sun, r_flux/r_flux_sun, i_flux/i_flux_sun, z_flux/z_flux_sun)
    ref_err = (0, 
               (r_flux/r_flux_sun * (r_flux_err/r_flux)**2 + (r_flux_sun_err/r_flux_sun)**2)**0.5,
               (i_flux/i_flux_sun * (i_flux_err/i_flux)**2 + (i_flux_sun_err/i_flux_sun)**2)**0.5,
               (z_flux/z_flux_sun * (z_flux_err/z_flux)**2 + (z_flux_sun_err/z_flux_sun)**2)**0.5,
                )
    
    if norm_band == 'i':
        return normalize_to_band(ref, ref_err, 2)
    elif norm_band == 'z':
        return normalize_to_band(ref, ref_err, 3)
    else: # default 'g'
        return ref, ref_err


def color_to_spec_gri(gr, ri, gr_err, ri_err, norm_band='i'):
    gr_sun = 0.45
    gr_sun_err = 0.02
    gi_sun = 0.57
    gi_sun_err = 0.02
    
    g_flux_sun = 10**(-1/2.5)
    r_flux_sun = 10**(-(1-gr_sun)/2.5)
    i_flux_sun = 10**(-(1-gi_sun)/2.5)
    
    
    r_flux_sun_err = (r_flux_sun**2 * (1/2.5 *np.log(10) * gr_sun_err)**2) ** 0.5
    i_flux_sun_err = (i_flux_sun**2 * (1/2.5 *np.log(10) * gi_sun_err)**2) ** 0.5
    


    gi = gr + ri
    

    gi_err = (gr_err**2 + ri_err**2)**0.5
    

    g_flux = 10**(-1/2.5)
    r_flux = 10**(-(1-gr)/2.5)
    i_flux = 10**(-(1-gi)/2.5)
    

    r_flux_err = (r_flux**2 * (1/2.5 *np.log(10) * gr_err)**2) ** 0.5
    i_flux_err = (i_flux**2 * (1/2.5 *np.log(10) * gi_err)**2) ** 0.5
   

    ref = (g_flux/g_flux_sun, r_flux/r_flux_sun, i_flux/i_flux_sun)
    ref_err = (0, 
               (r_flux/r_flux_sun * (r_flux_err/r_flux)**2 + (r_flux_sun_err/r_flux_sun)**2)**0.5,
               (i_flux/i_flux_sun * (i_flux_err/i_flux)**2 + (i_flux_sun_err/i_flux_sun)**2)**0.5,
                )
    
    if norm_band == 'i':
        return normalize_to_band(ref, ref_err, 2)
    else: # default 'g'
        return ref, ref_err


def color_to_spec_grz(gr, rz, gr_err, rz_err, norm_band='z'):
    gr_sun = 0.45
    gr_sun_err = 0.02
    gz_sun = 0.61
    gz_sun_err = 0.03
    
    g_flux_sun = 10**(-1/2.5)
    r_flux_sun = 10**(-(1-gr_sun)/2.5)
    z_flux_sun = 10**(-(1-gz_sun)/2.5)
    
    r_flux_sun_err = (r_flux_sun**2 * (1/2.5 *np.log(10) * gr_sun_err)**2) ** 0.5
    z_flux_sun_err = (z_flux_sun**2 * (1/2.5 *np.log(10) * gz_sun_err)**2) ** 0.5

    gz = gr + rz

    gz_err = (gr_err**2 + rz_err**2)**0.5

    g_flux = 10**(-1/2.5)
    r_flux = 10**(-(1-gr)/2.5)
    z_flux = 10**(-(1-gz)/2.5)

    r_flux_err = (r_flux**2 * (1/2.5 *np.log(10) * gr_err)**2) ** 0.5
    z_flux_err = (z_flux**2 * (1/2.5 *np.log(10) * gz_err)**2) ** 0.5

    ref = (g_flux/g_flux_sun, r_flux/r_flux_sun, z_flux/z_flux_sun)
    ref_err = (0, 
               (r_flux/r_flux_sun * (r_flux_err/r_flux)**2 + (r_flux_sun_err/r_flux_sun)**2)**0.5,
               (z_flux/z_flux_sun * (z_flux_err/z_flux)**2 + (z_flux_sun_err/z_flux_sun)**2)**0.5,
                )
    
    if norm_band == 'z':
        return normalize_to_band(ref, ref_err, 2)
    else: # default 'g'
        return ref, ref_err


def color_to_spec_bvr(gr, ri, gr_err, ri_err, norm_band='i'):
    gr_sun = 0.642
    gr_sun_err = 0.016
    gi_sun = 0.996
    gi_sun_err = 0.018
    
    g_flux_sun = 10**(-1/2.5)
    r_flux_sun = 10**(-(1-gr_sun)/2.5)
    i_flux_sun = 10**(-(1-gi_sun)/2.5)
    
    
    r_flux_sun_err = (r_flux_sun**2 * (1/2.5 *np.log(10) * gr_sun_err)**2) ** 0.5
    i_flux_sun_err = (i_flux_sun**2 * (1/2.5 *np.log(10) * gi_sun_err)**2) ** 0.5
    


    gi = gr + ri
    

    gi_err = (gr_err**2 + ri_err**2)**0.5
    

    g_flux = 10**(-1/2.5)
    r_flux = 10**(-(1-gr)/2.5)
    i_flux = 10**(-(1-gi)/2.5)
    

    r_flux_err = (r_flux**2 * (1/2.5 *np.log(10) * gr_err)**2) ** 0.5
    i_flux_err = (i_flux**2 * (1/2.5 *np.log(10) * gi_err)**2) ** 0.5
   

    ref = (g_flux/g_flux_sun, r_flux/r_flux_sun, i_flux/i_flux_sun)
    ref_err = (0, 
               (r_flux/r_flux_sun * (r_flux_err/r_flux)**2 + (r_flux_sun_err/r_flux_sun)**2)**0.5,
               (i_flux/i_flux_sun * (i_flux_err/i_flux)**2 + (i_flux_sun_err/i_flux_sun)**2)**0.5,
                )
    
    if norm_band == 'i':
        return normalize_to_band(ref, ref_err, 2)
    else: # default 'g'
        return ref, ref_err

def color_to_spec_grizy(gr, ri, iz, zy, gr_err, ri_err, iz_err, zy_err, norm_band='i'):
    gr_sun = 0.45
    gr_sun_err = 0.02
    gi_sun = 0.57
    gi_sun_err = 0.02
    gz_sun = 0.61
    gz_sun_err = 0.03
    gy_sun = 0.63
    gy_sun_err = 0.03
    
    g_flux_sun = 10**(-1/2.5)
    r_flux_sun = 10**(-(1-gr_sun)/2.5)
    i_flux_sun = 10**(-(1-gi_sun)/2.5)
    z_flux_sun = 10**(-(1-gz_sun)/2.5)
    y_flux_sun = 10**(-(1-gy_sun)/2.5)
    
    r_flux_sun_err = (r_flux_sun**2 * (1/2.5 *np.log(10) * gr_sun_err)**2) ** 0.5
    i_flux_sun_err = (i_flux_sun**2 * (1/2.5 *np.log(10) * gi_sun_err)**2) ** 0.5
    z_flux_sun_err = (z_flux_sun**2 * (1/2.5 *np.log(10) * gz_sun_err)**2) ** 0.5
    y_flux_sun_err = (y_flux_sun**2 * (1/2.5 *np.log(10) * gy_sun_err)**2) ** 0.5

    gi = gr + ri
    gz = gr + ri + iz
    gy = gr + ri + iz + zy

    gi_err = (gr_err**2 + ri_err**2)**0.5
    gz_err = (gr_err**2 + ri_err**2 + iz_err**2)**0.5
    gy_err = (gr_err**2 + ri_err**2 + iz_err**2 + zy_err**2)**0.5

    g_flux = 10**(-1/2.5)
    r_flux = 10**(-(1-gr)/2.5)
    i_flux = 10**(-(1-gi)/2.5)
    z_flux = 10**(-(1-gz)/2.5)
    y_flux = 10**(-(1-gy)/2.5)

    r_flux_err = (r_flux**2 * (1/2.5 *np.log(10) * gr_err)**2) ** 0.5
    i_flux_err = (i_flux**2 * (1/2.5 *np.log(10) * gi_err)**2) ** 0.5
    z_flux_err = (z_flux**2 * (1/2.5 *np.log(10) * gz_err)**2) ** 0.5
    y_flux_err = (y_flux**2 * (1/2.5 *np.log(10) * gy_err)**2) ** 0.5

    ref = (g_flux/g_flux_sun, r_flux/r_flux_sun, i_flux/i_flux_sun, z_flux/z_flux_sun, y_flux/y_flux_sun)
    ref_err = (0, 
               (r_flux/r_flux_sun * (r_flux_err/r_flux)**2 + (r_flux_sun_err/r_flux_sun)**2)**0.5,
               (i_flux/i_flux_sun * (i_flux_err/i_flux)**2 + (i_flux_sun_err/i_flux_sun)**2)**0.5,
               (z_flux/z_flux_sun * (z_flux_err/z_flux)**2 + (z_flux_sun_err/z_flux_sun)**2)**0.5,
               (y_flux/y_flux_sun * (y_flux_err/y_flux)**2 + (y_flux_sun_err/y_flux_sun)**2)**0.5,
                )
    
    if norm_band == 'i':
        return normalize_to_band(ref, ref_err, 2)
    elif norm_band == 'z':
        return normalize_to_band(ref, ref_err, 3)
    elif norm_band == 'y':
        return normalize_to_band(ref, ref_err, 4)
    else: # default 'g'
        return ref, ref_err

def spec_to_color(s, sir):
    def res_s(colors, s, sir):
        ref, _ = color_to_spec(colors[0], colors[1], colors[2], 0, 0, 0, norm_band='g')
        slope0 = (ref[1]-ref[0])/(6231-4770)*100000
        slope1 = (ref[2]-ref[1])/(7625-6231)*100000
        #slope = (slope0 + slope1)/2
        slope= slope0
        slope_ir = (ref[3]-ref[1])/(9134-6231)*100000
        var = (slope-s)**2 + (slope_ir - sir)**2
        #print(slope, slope_ir, var)
        return(var)

    res =minimize(res_s, [0.75, 0.16, 0.3], args=(s, sir), bounds=[(0.25,1.25),(0.0,0.6),(-0.1,1.0)])    
    return(res.x)



if __name__ == '__main__':
    gr = .73
    ri = 0.3
    rz = 0.26
    
    # 1. Test original normalization to g
    ref_g, _ = color_to_spec(gr, ri, rz, 0, 0, 0, norm_band='g')
    print("g-normalized reflectance:", ref_g)
    
    # 2. Test model normalization to i (0.75 um)
    ref_i, ref_err_i = color_to_spec(gr, ri, rz, 0.05, 0.05, 0.05, norm_band='i')
    print("i-normalized reflectance:", ref_i)
    print("i-normalized errors:", ref_err_i)
    
    slope0 = (ref_g[1]-ref_g[0])/(6231-4770)*100000
    slope1 = (ref_g[2]-ref_g[1])/(7625-6231)*100000
    slope = slope0
    slope_ir = (ref_g[3]-ref_g[1])/(9134-6231)*100000
    print("g-normalized slopes:", slope, slope_ir)
    
    color = spec_to_color(20, 4)
    print("spec_to_color:", color)
    
    # 3. Test model normalization to z (0.9 um) for grz
    ref_grz, ref_err_grz = color_to_spec_grz(gr, rz, 0.05, 0.05, norm_band='z')
    print("z-normalized grz reflectance:", ref_grz)
    print("z-normalized grz errors:", ref_err_grz)
