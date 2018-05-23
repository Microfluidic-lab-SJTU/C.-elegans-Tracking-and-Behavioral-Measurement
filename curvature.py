import numpy as np

def _curvature_fun(x_d, y_d, x_dd, y_dd):
    return (x_d*y_dd - y_d*x_dd)/(x_d*x_d + y_d*y_d)**1.5

def _gradient_windowed(X, points_window, axis):
    '''
    Calculate the gradient using an arbitrary window. The larger window make 
    this procedure less noisy that the numpy native gradient.
    '''
    w_s = 2*points_window
    
    #I use slices to deal with arbritary dimenssions 
    #https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    n_axis_ini = max(0, axis)
    n_axis_fin = max(0, X.ndim-axis-1)
    right_slice = [slice(None, None, None)]*n_axis_ini + [slice(None, -w_s, None)]
    left_slice = [slice(None, None, None)]*n_axis_ini + [slice(w_s, None, None)]
    
    right_pad = [(0,0)]*n_axis_ini + [(w_s, 0)] + [(0,0)]*n_axis_fin
    left_pad = [(0,0)]*n_axis_ini + [(0, w_s)] + [(0,0)]*n_axis_fin
    
    right_side = np.pad(X[right_slice], right_pad, 'edge')
    left_side = np.pad(X[left_slice], left_pad, 'edge')
    
    ramp = np.full(X.shape[axis]-2*w_s, w_s*2)
    
    ramp = np.pad(ramp,  pad_width = (w_s, w_s),  mode='linear_ramp', end_values = w_s)
    #ramp = np.pad(ramp,  pad_width = (w_s, w_s),  mode='constant', constant_values = np.nan)
    ramp_slice = [None]*n_axis_ini + [slice(None, None, None)] + [None]*n_axis_fin
                 
    grad = (left_side - right_side) / ramp[ramp_slice] #divide it by the time window
    
    return grad

def curvature_grad(curve, points_window=None, axis=1, is_nan_border=True):
    '''
    Calculate the curvature using the gradient using differences similar to numpy grad
    
    x1, x2, x3
    
    grad(x2) = (x3-x1)/2
    
    '''
    
    #The last element must be the coordinates
    assert curve.shape[-1] == 2
    assert axis != curve.ndim - 1    
    
    if points_window is None:
        points_window = 1
    
    if curve.shape[0] <= points_window*4:
        return np.full((curve.shape[0], curve.shape[1]), np.nan)
    
    d = _gradient_windowed(curve, points_window, axis=axis)
    dd = _gradient_windowed(d, points_window, axis=axis)
    print(d)
    print(dd)
    gx = d[..., 0]
    gy = d[..., 1]
    ggx = dd[..., 0]
    ggy = dd[..., 1]
    
    curvature_r =  _curvature_fun(gx, gy, ggx, ggy)
    if is_nan_border:
        #I cannot really trust in the border gradient
        w_s = 4*points_window
        n_axis_ini = max(0, axis)
        right_slice = [slice(None, None, None)]*n_axis_ini + [slice(None, w_s, None)]
        left_slice = [slice(None, None, None)]*n_axis_ini + [slice(-w_s, None, None)]
        curvature_r[right_slice] = np.nan
        curvature_r[left_slice] = np.nan
    
    return curvature_r