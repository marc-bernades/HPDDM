#!/usr/bin/python3

import sys
import os
import glob
import numpy as np
import h5py
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc,rcParams
plt.rc( 'text', usetex = True )
plt.rc( 'font', size = 18 )
plt.rc( 'text.latex', preamble = r'\usepackage{amsmath} \usepackage{amssymb}')
#import pint
#ureg = pint.UnitRegistry()
########## PARULA COLORMAP ##########
from matplotlib.colors import LinearSegmentedColormap
import dask.array as da
from scipy.fft import fft, ifft, fftn, ifftn
from matplotlib.ticker import LogFormatterMathtext
from scipy.ndimage import gaussian_filter
import matplotlib.ticker as mticker


cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)


def get_Metrics(d_dir, dataset_name, delta):
    ########## OPEN DATA FILES ##########
    data_file = h5py.File( f"../data_resolvent/{dataset_name}.h5", 'r' )
    #print(list( data_file.keys() ))
    rho_data        = data_file['rho'][:,:,:]
    u_data          = data_file['u'][:,:,:]
    v_data          = data_file['v'][:,:,:]
    w_data          = data_file['w'][:,:,:]
    T_data          = data_file['T'][:,:,:]
    x_data          = data_file['x'][:,:,:]
    y_data          = data_file['y'][:,:,:]
    z_data          = data_file['z'][:,:,:]
    mu_data         = data_file['mu'][:,:,:]
    avg_u_data      = data_file['u'][:,:,:]
    avg_v_data      = data_file['v'][:,:,:]
    avg_w_data      = data_file['w'][:,:,:]
    avg_T_data      = data_file['T'][:,:,:]
    avg_rho_data    = data_file['rho'][:,:,:]
    avg_mu_data     = data_file['mu'][:,:,:]
    avg_P_data      = data_file['P'][:,:,:]
    avg_kappa_data  = data_file['kappa'][:,:,:]
    avg_c_p_data    = data_file['c_p'][:,:,:]
    x_data          = data_file['x'][:,:,:]
    y_data          = data_file['y'][:,:,:]
    z_data          = data_file['z'][:,:,:]
    num_points_x    = x_data[0,0,:].size
    num_points_y    = y_data[0,:,0].size
    num_points_z    = z_data[:,0,0].size
    num_points_xz   = num_points_x*num_points_z

    ########## CALCULATE BULK VALUES ##########
    rho_b = 0.0
    mu_b  = 0.0
    u_b   = 0.0
    P_b   = 0.0
    T_b   = 0.0
    total_volume = 0.0
    for i in range( 1, num_points_x - 2):
        for j in range( 1, num_points_y - 2):
            for k in range( 1, num_points_z - 2 ):
                # Calculate volume
                delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
                delta_y = 0.5*( y_data[k,j+1,i] - y_data[k,j-1,i] )
                delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
                volume  = delta_x*delta_y*delta_z
                total_volume += volume
                # Update bulk parameters
                rho_b += avg_rho_data[k,j,i]*volume
                mu_b  += avg_mu_data[k,j,i]*volume
                u_b   += avg_u_data[k,j,i]*volume
                P_b   += avg_P_data[k,j,i]*volume
                T_b   += avg_T_data[k,j,i]*volume
    
    rho_b *= 1.0/total_volume
    mu_b  *= 1.0/total_volume
    u_b   *= 1.0/total_volume
    P_b   *= 1.0/total_volume
    T_b   *= 1.0/total_volume
    Re_b   = u_b*delta*rho_b/mu_b
    print( "\nBulk values:" )
    print( "Re_b = {:.3f}".format( Re_b ), ", rho_b = {:.3f}".format( rho_b ), "[kg/m3], u_b = {:.3f}".format( u_b ), "[m/s], mu_b = {:.4g}".format( mu_b ), "[Pa s], P_b = {:.3f}".format( P_b ), "[Pa], T_b = {:.3f}".format( T_b ), "[K]" )



    ########## CALCULATE FRICTION VALUES ##########
    total_area_bw = 0.0
    rho_bw        = 0.0
    mu_bw         = 0.0
    T_bw          = 0.0
    kappa_bw      = 0.0
    c_p_bw        = 0.0
    u_boundary_bw = 0.0
    u_inner_bw    = 0.0
    T_boundary_bw = 0.0
    T_inner_bw    = 0.0
    total_area_tw = 0.0
    rho_tw        = 0.0
    mu_tw         = 0.0
    T_tw          = 0.0
    kappa_tw      = 0.0
    c_p_tw        = 0.0
    u_boundary_tw = 0.0
    u_inner_tw    = 0.0
    T_boundary_tw = 0.0
    T_inner_tw    = 0.0
    for i in range( 1, num_points_x-1 ):
        for k in range( 1, num_points_z-1 ):
            ## Bottom wall
            j = 0
            # Compute area bottom wall
            delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
            delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
            area_bw = delta_x*delta_z
            total_area_bw +=area_bw
            # Compute rho, mu, rho, T, kappa and c_p at bottom wall
            rho_bw   += area_bw*0.5*( avg_rho_data[k,j,i] + avg_rho_data[k,j+1,i] )
            mu_bw    += area_bw*0.5*( avg_mu_data[k,j,i] + avg_mu_data[k,j+1,i] )
            T_bw     += area_bw*0.5*( avg_T_data[k,j,i] + avg_T_data[k,j+1,i] )  
            kappa_bw += area_bw*0.5*( avg_kappa_data[k,j,i] + avg_kappa_data[k,j+1,i] )      
            c_p_bw   += area_bw*0.5*( avg_c_p_data[k,j,i] + avg_c_p_data[k,j+1,i])                                
            # Streamwise velocity and temperature boundary and inner points
            u_boundary_bw += area_bw*avg_u_data[k,j,i]
            u_inner_bw    += area_bw*avg_u_data[k,j+1,i]
            T_boundary_bw += area_bw*avg_T_data[k,j,i]
            T_inner_bw    += area_bw*avg_T_data[k,j+1,i]
            ## Top wall
            j = num_points_y - 1
            # Compute area top wall
            delta_x = 0.5*( x_data[k,j,i+1] - x_data[k,j,i-1] )
            delta_z = 0.5*( z_data[k+1,j,i] - z_data[k-1,j,i] )
            area_tw = delta_x*delta_z
            total_area_tw += area_tw
            # Compute rho, mu, rho, T, kappa and c_p at top wall
            rho_tw   += area_tw*0.5*( avg_rho_data[k,j,i] + avg_rho_data[k,j-1,i] )
            mu_tw    += area_tw*0.5*( avg_mu_data[k,j,i] + avg_mu_data[k,j-1,i] )
            T_tw     += area_tw*0.5*( avg_T_data[k,j,i] + avg_T_data[k,j-1,i] )   
            kappa_tw += area_tw*0.5*( avg_kappa_data[k,j,i] + avg_kappa_data[k,j-1,i] )  
            c_p_tw   += area_tw*0.5*( avg_c_p_data[k,j,i] + avg_c_p_data[k,j-1,i] )  
            # Streamwise velocity and temperature boundary wall and inner wall
            u_boundary_tw += area_tw*avg_u_data[k,j,i]
            u_inner_tw    += area_tw*avg_u_data[k,j-1,i]
            T_boundary_tw += area_tw*avg_T_data[k,j,i]
            T_inner_tw    += area_tw*avg_T_data[k,j-1,i]
    rho_bw   *= 1.0/total_area_bw
    mu_bw    *= 1.0/total_area_bw
    T_bw     *= 1.0/total_area_bw
    kappa_bw *= 1.0/total_area_bw
    c_p_bw   *= 1.0/total_area_bw
    u_boundary_bw *= 1.0/total_area_bw
    u_inner_bw    *= 1.0/total_area_bw
    T_boundary_bw *= 1.0/total_area_bw
    T_inner_bw    *= 1.0/total_area_bw
    rho_tw   *= 1.0/total_area_tw
    mu_tw    *= 1.0/total_area_tw
    T_tw     *= 1.0/total_area_tw
    kappa_tw *= 1.0/total_area_tw
    c_p_tw   *= 1.0/total_area_tw
    u_boundary_tw *= 1.0/total_area_tw
    u_inner_tw    *= 1.0/total_area_tw
    T_boundary_tw *= 1.0/total_area_tw
    T_inner_tw    *= 1.0/total_area_tw
     


    ### WALL VALUES
    delta_y_bw = y_data[1,1,1] - y_data[1,0,1]
    delta_y_tw = y_data[1,num_points_y-1,1] - y_data[1,num_points_y-2,1]

    # tau wall
    tau_bw = mu_bw*( u_inner_bw - u_boundary_bw )/delta_y_bw
    tau_tw = mu_tw*( u_inner_tw - u_boundary_tw )/delta_y_tw

    # u_tau
    u_tau_bw = np.sqrt( tau_bw/rho_bw )
    u_tau_tw = np.sqrt( tau_tw/rho_tw )

    # T_tau wall
    T_tau_bw = kappa_bw*( ( T_inner_bw - T_boundary_bw )/delta_y_bw )/( rho_bw*c_p_bw*u_tau_bw )
    T_tau_tw = kappa_tw*( ( T_boundary_tw - T_inner_tw )/delta_y_tw )/( rho_tw*c_p_tw*u_tau_tw )

    # nu wall
    nu_bw = mu_bw/rho_bw
    nu_tw = mu_tw/rho_tw

    # alpha wall
    alpha_bw = kappa_bw/( rho_bw*c_p_bw )
    alpha_tw = kappa_tw/( rho_tw*c_p_tw )

    # Skin friction coefficient
    Cf_bw = tau_bw/( 0.5*rho_b*u_b*u_b )
    Cf_tw = tau_tw/( 0.5*rho_b*u_b*u_b )

    # Reynolds tau
    Re_tau_bw = rho_bw*u_tau_bw*delta/mu_bw
    Re_tau_tw = rho_tw*u_tau_tw*delta/mu_tw

    # Prandtl number at walls
    Pr_bw = c_p_bw*mu_bw/kappa_bw
    Pr_tw = c_p_tw*mu_tw/kappa_tw

    # Nusselt number at walls
    Nu_bw = delta*( ( T_boundary_bw - T_inner_bw )/delta_y_bw )/( T_bw - T_b )
    Nu_tw = delta*( ( T_boundary_tw - T_inner_tw )/delta_y_tw )/( T_tw - T_b )

    # Stanton number at walls
    St_bw = Nu_bw/( Re_tau_bw*Pr_bw )
    St_tw = Nu_tw/( Re_tau_tw*Pr_tw )

    print( "\nWall values:" )
    print( "Bottom wall: rho_bw = {:.3f}".format( rho_bw ), "[kg/m3], u_tau_bw = {:.3f}".format( u_tau_bw ), "[m/s], mu_bw = {:.4g}".format( mu_bw ), "[Pa s], tau_bw = {:.3f}".format( tau_bw ), "[N/m2], T_bw = {:.3f}".format( T_bw ), "[K], T_tau_bw = {:.3f}".format( T_tau_bw ), "[K], nu_bw = {:.4g}".format( nu_bw ), "[m2/s], alpha_bw = {:.4g}".format( alpha_bw ), "[m2/s], Cf_bw = {:.3f}".format( Cf_bw ), ", Re_tau_bw = {:.3f}".format( Re_tau_bw ), ", Pr_bw = {:.3f}".format( Pr_bw ), ", Nu_bw = {:.3f}".format( Nu_bw ), ", St_bw = {:.3f}".format( St_bw ) )
    print( "Top wall: rho_tw = {:.3f}".format( rho_tw ), "[kg/m3], u_tau_tw = {:.3f}".format( u_tau_tw ), "[m/s], mu_tw = {:.4g}".format( mu_tw ), "[Pa s], tau_tw = {:.3f}".format( tau_tw ), "[N/m2], T_tw = {:.3f}".format( T_tw ), "[K], T_tau_tw = {:.3f}".format( T_tau_tw ), "[K], nu_tw = {:.4g}".format( nu_tw ), "[m2/s], alpha_tw = {:.4g}".format( alpha_tw ), "[m2/s], Cf_tw = {:.3f}".format( Cf_tw ), ", Re_tau_tw = {:.3f}".format( Re_tau_tw ), ", Pr_tw = {:.3f}".format( Pr_tw ), ", Nu_tw = {:.3f}".format( Nu_tw ), ", St_tw = {:.3f}".format( St_tw ) )

    
    ##############################
    ### Average variables in space
    ##############################
    avg_u                 = np.zeros( int( num_points_y ) )
    avg_rho               = np.zeros( int( num_points_y ) )
    avg_mu                = np.zeros( int( num_points_y ) )
    y_data_tw          = np.zeros( int( num_points_y ) )
    y_data_bw          = np.zeros( int( num_points_y ) )
    avg_y_plus_bw      = np.zeros( int( num_points_y ) )
    avg_y_plus_tw      = np.zeros( int( num_points_y ) )
    avg_u_plus_bw      = np.zeros( int( num_points_y ) )
    avg_u_plus_tw      = np.zeros( int( num_points_y ) )
    avg_v_plus_bw      = np.zeros( int( num_points_y ) )
    avg_v_plus_tw      = np.zeros( int( num_points_y ) )
    avg_w_plus_bw      = np.zeros( int( num_points_y ) )
    avg_w_plus_tw      = np.zeros( int( num_points_y ) )
    avg_T_plus_bw      = np.zeros( int( num_points_y ) )
    avg_T_plus_tw      = np.zeros( int( num_points_y ) )

    for j in range( 1, num_points_y - 1):
        for i in range( 1, num_points_x - 1 ):
            for k in range( 1, num_points_z - 1 ):
                aux_j = j
                # average velocity across y
                avg_u[aux_j]              += ( 1.0/num_points_xz )*avg_u_data[k,j,i]
                avg_rho[aux_j]            += ( 1.0/num_points_xz )*avg_rho_data[k,j,i]
                avg_mu[aux_j]             += ( 1.0/num_points_xz )*avg_mu_data[k,j,i]

            
                # Compute top wall
                aux_j = num_points_y - j - 1
                # y_data top wall
                y_data_tw[aux_j]       += ( 1.0/num_points_xz )*y_data[k,aux_j,i]
                # Time average wall units
                avg_y_plus_tw[aux_j]   += ( 1.0/num_points_xz )*y_data[k,aux_j,i]*( u_tau_tw/(mu_tw/rho_tw) )
                avg_u_plus_tw[aux_j]   += ( 1.0/num_points_xz )*avg_u_data[k,j,i]*( 1.0/u_tau_tw )
                avg_v_plus_tw[aux_j]   += ( 1.0/num_points_xz )*avg_v_data[k,j,i]*( 1.0/u_tau_tw )
                avg_w_plus_tw[aux_j]   += ( 1.0/num_points_xz )*avg_w_data[k,j,i]*( 1.0/u_tau_tw )
                avg_T_plus_tw[aux_j]   += -1.0*( 1.0/num_points_xz )*((avg_T_data[k,j,i] - T_tw)/T_tau_tw)  

                # Compute bottom wall
                # y_data bottom wall
                y_data_bw[j]      += ( 1.0/num_points_xz )*y_data[k,j,i]
                # Time average wall units
                avg_y_plus_bw[j]  += ( 1.0/num_points_xz )*y_data[k,j,i]*( u_tau_bw/(mu_bw/rho_bw) )
                avg_u_plus_bw[j]  += ( 1.0/num_points_xz )*avg_u_data[k,j,i]*( 1.0/u_tau_bw )
                avg_v_plus_bw[j]  += ( 1.0/num_points_xz )*avg_v_data[k,j,i]*( 1.0/u_tau_bw )
                avg_w_plus_bw[j]  += ( 1.0/num_points_xz )*avg_w_data[k,j,i]*( 1.0/u_tau_bw )
                avg_T_plus_bw[j]  += ( 1.0/num_points_xz )*((avg_T_data[k,j,i] - T_bw)/T_tau_bw)
                

    ########## FIND MAXIMUM TIME-AVERAGED VELOCITY Y-POSITION ##########

    ### Bottom wall
    max_y_index_bw = -1
    max_value      = -1.0
    for p in range( 0, len( avg_u ) ):
        value = avg_u[p]
        if( value > max_value ):
            max_y_index_bw = p
            max_value = value
        else:
            break 
    ### Top wall
    max_y_index_tw = len(avg_u) - max_y_index_bw    
    print(f"max_index_y_bw = {max_y_index_bw}, at y_bw = {y_data_bw[max_y_index_bw]/delta}, y_plus_bw = {avg_y_plus_bw[max_y_index_bw]}")
    print(f"max_index_y_tw = {max_y_index_tw}, at y_tw = {y_data_tw[max_y_index_tw]/delta}, y_plus_tw = {avg_y_plus_tw[max_y_index_tw]}")


    return u_b, u_tau_bw, u_tau_tw, rho_bw, rho_tw, mu_bw, mu_tw, T_tau_bw, T_tau_tw, avg_u, avg_rho, avg_mu, avg_y_plus_bw, avg_u_plus_bw, avg_y_plus_tw, avg_u_plus_tw, max_y_index_bw, max_y_index_tw


def save_DNS_grid(d_dir,dataset_name):

    ########## OPEN DATA FILES ##########
    print("Saving DNS grid...")
    data_file = h5py.File( f"../data_resolvent/{dataset_name}.h5", 'r' )
    x_data          = data_file['x'][:,:,:]
    y_data          = data_file['y'][:,:,:]
    z_data          = data_file['z'][:,:,:]

    num_points_x    = x_data[0,0,:].size
    num_points_y    = y_data[0,:,0].size
    num_points_z    = z_data[:,0,0].size

    n_vars = 3
    grid   = np.zeros((int(n_vars),int( num_points_z - 2 ),int( num_points_y - 2),int( num_points_x - 2) ))

    grid[0,:,:,:] = x_data[1:-1,1:-1,1:-1]
    grid[1,:,:,:] = y_data[1:-1,1:-1,1:-1]
    grid[2,:,:,:] = z_data[1:-1,1:-1,1:-1]
    # Change to X,Y,Z
    grid = np.einsum("ijkl->ilkj",grid)

    np.save( f"{d_dir}/grid.npy", grid)



class POD_process:
    def __init__(self, save_dir, cases, dt, delta, L_x, L_z,
             u_tau_bw, mu_bw, rho_bw, u_tau_tw, mu_tw, rho_tw):
        self.save_dir   = save_dir
        self.cases      = cases
        self.dt         = dt
        self.delta_h    = delta
        self.L_x        = L_x
        self.L_z        = L_z
        self.u_tau_bw   = u_tau_bw
        self.mu_bw      = mu_bw
        self.rho_bw     = rho_bw
        self.u_tau_tw   = u_tau_tw
        self.mu_tw      = mu_tw
        self.rho_tw     = rho_tw
        
        ## Obtain snapshots
        if not(os.path.exists(f"{self.save_dir}/Snapshots_{len(cases)}.npy")):
            print("Snapshots need to be computed in 5 x n_x/2 x n_t x n_z/2, 2000 (ref DMD_3d)...")
            #self.snapshots  = self.preprocess_snapshots()
        else:
            print("Snapshots from file...")
            self.snapshots  = np.load(f"{self.save_dir}/Snapshots_{len(cases)}.npy")
            self.n_snapsots = self.snapshots.shape[1]
            self.n_dim      = 5
            self.n_x        = 128/2
            self.n_y        = 128
            self.n_z        = 128/2

        ## Obtain pseudoboiling
        self.pseudoboiling  = np.load(f"../FFT_time/data/pseudoboiling.npy")

        ## Obtain bulk values
        print("Computing metrics and wall values...")
        (self.ub, self.u_tau_bw, self.u_tau_tw, self.rho_bw, self.rho_tw, self.mu_bw, self.mu_tw, self.T_tau_bw, self.T_tau_tw, 
        self.avg_u, self.avg_rho, self.avg_mu, 
        self.avg_y_plus_bw, self.avg_u_plus_bw, self.avg_y_plus_tw, self.avg_u_plus_tw,
        self.max_y_index_bw, self.max_y_index_tw) = get_Metrics(self.save_dir,f"{cases[0]}", self.delta_h)
        
        ## Wall units
        self.y_plus_bw, self.y_plus_tw = self.preprocess_wall_units()

        ## Obtain delta normalized (dx*dy*dz,1)
        self.Delta_volume = np.load(f"{self.save_dir}/Delta_{len(self.cases)}.npy")
       
        # Obtain weighting matrix
        self.m = np.load(f"{self.save_dir}/Weights_Enorm_{len(self.cases)}.npy") # np.array([m_rho, m_u, m_u, m_u, m_T])

    def compute_POD(self):
        X_unweighted = self.snapshots.copy()
        Q            = self.Delta_volume
        len_delta    = Q.shape[0]

        X_weighted   = X_unweighted.copy()
        # --- Apply Energy-based weighting ---
        for var in range(0, 5):
            for n_T in range(X_weighted.shape[1]):
                ini = var * len_delta
                fin = (var + 1) * len_delta
                U = X_unweighted[ini:fin, n_T]
                F = np.sqrt(self.m[var] * Q[:, 0])
                X_weighted[ini:fin, n_T] = F * U

        print("\nComputing POD via method of snapshots...")


        # --- Method of Snapshots: SVD on temporal correlation matrix ---
        C = np.dot(X_weighted.T, X_weighted)  # Temporal covariance (N x N)
        eigvals, eigvecs = np.linalg.eigh(C)  # Symmetric matrix

        # --- Sort and truncate ---
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # --- Compute spatial POD modes ---
        Phi = np.dot(X_weighted, eigvecs)     # Shape (M x r)
        Phi = Phi / np.sqrt(eigvals) #np.linalg.norm(Phi, axis=0)  # Normalize

        # Validate Modal Orthonormality
        print(np.allclose(Phi.T @ Phi, np.eye(X_weighted.shape[1]), atol=1e-6))


        print("POD completed.")

        # Save results
        np.save(f"{self.save_dir}/POD_eigvals.npy", eigvals)
        np.save(f"{self.save_dir}/POD_eigvecs.npy", eigvecs)
        np.save(f"{self.save_dir}/POD_modes.npy", Phi)
        np.save(f"{self.save_dir}/POD_X_weighted.npy",X_weighted)
        np.save(f"{self.save_dir}/POD_X_unweighted.npy",X_unweighted)
        print(f"Saved POD modes, shape: {Phi.shape}")
        print(f"Saved POD eigenvalues, shape: {eigvals.shape}")

         # Now truncate based on each r
        #POD_list = []
        #for r in rs:
            #Phi_r = Phi_full[:, :r]
            #eigvals_r = eigvals_full[:r]
            #POD_list.append((Phi_r, eigvals_r))

        #return POD_list

    def reconstruct_POD_modes(self, eigvals, eigvecs, Phi, r):
        """
        Reconstruct snapshot data from r POD modes.

        Parameters:
            eigvals: POD eigenvalues
            eigvecs: POD temporal modes
            Phi: POD spatial modes (computed from weighted data)
            r: number of modes to use
            X_unweighted: original unweighted snapshot matrix (M x N)

        Returns:
            X_rec: reconstructed (unweighted) snapshot matrix
        """

        Q = self.Delta_volume           # (len_delta, 1)
        m = self.m                      # [rho, rho, rho, 1/(gamma-1), 1]
        len_delta   = Q.shape[0]
        n_snapshots = self.snapshots.shape[1]
        
        # --- Step 1: Reconstruct in weighted space using r modes ---
        Phi_r = Phi[:, :r]                           # Shape: (M, r)
        A_r = eigvecs[:, :r].T * np.sqrt(eigvals[:r])[:, np.newaxis]  # Shape: (r, N)
        X_rec_weighted = Phi_r @ A_r                 # Shape: (M, N)

        # --- Step 1: Undo energy-based weighting to physical space ---
        X_rec_unweighted = X_rec_weighted.copy() #X_weighted = X.copy()       
        for var in range(5):  # Loop over variables
            ini = var * len_delta
            fin = (var + 1) * len_delta
            F = np.sqrt(m[var] * Q[:, 0])  # shape: (len_delta,)
            for n_T in range(n_snapshots):
                X_rec_unweighted[ini:fin, n_T] /= F

        # --- Step 2: Truncate modes and reconstruct in weighted space ---
        # Truncate modes and eigenvectors
        #Phi_r = Phi[:, :r]                    # Shape: (M, r)

        # Compute time coefficients (modal amplitudes)
        #A_r = Phi_r.T @ X_weighted            # Shape: (r, N)

        # Reconstruct fluctuation field
        #X_rec = Phi_r @ A_r                        # Shape: (M, N)

        # UNDO the weighting applied before POD
        #len_delta = Q.shape[0]
        #for var in range(5):
        #    ini = var * len_delta
        #    fin = (var + 1) * len_delta
        #    F = np.sqrt(m[var] * Q[:, 0])
        #    for n_T in range(X.shape[1]):
        #        X_rec[ini:fin, n_T] /= F

        #error_rel = np.linalg.norm(X_weighted - X_rec_weighted, 'fro') / np.linalg.norm(X_weighted, 'fro')
        #print(f"Relative reconstruction error: {error_rel:.4e}")
        
        return X_rec_unweighted

    def plot_reconstructed_POD(self,X_rec,snapshot_idx,var_idx):
    
        
        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        x_data = grid[0,:,:,:]/self.delta_h
        y_data = grid[1,:,:,:]/self.delta_h
        z_data = grid[2,:,:,:]/self.delta_h

        # Example: 2D slice at y = center, t = 100
        X = self.snapshots
        # Extract and reshape
        Nx = int(self.n_x)
        Ny = int(self.n_y)
        Nz = int(self.n_z)
        z_slice = int(Nz // 2)  # centerline
        
        Q = self.Delta_volume
        len_delta = Q.shape[0]
        ini = var_idx * len_delta
        fin = (var_idx + 1) * len_delta
        orig_field = X[ini:fin, snapshot_idx].reshape(Nx, Ny, Nz)
        recon_field = X_rec[ini:fin, snapshot_idx].reshape(Nx, Ny, Nz)
        diff = orig_field - recon_field
        # Compare slice at y = center
        plt.clf()
        #plt.subplot(3, 1, 1)
        #plt.imshow(orig_field[:, :, z_slice], cmap='jet')
        #plt.title("Original")
        #plt.colorbar()

        #plt.subplot(3, 1, 2)
        #plt.imshow(recon_field[:, :, z_slice], cmap='jet')
        #plt.title("Reconstructed")
        #plt.colorbar()

        #plt.subplot(3, 1, 3)
        #plt.imshow(diff[:, :, z_slice], cmap='bwr')
        #plt.title("Diff")
        #plt.colorbar()

        #plt.suptitle(f"Snapshot {snapshot_idx}, y={y_slice}, variable u")
        #plt.savefig(f'figures/Reconstruct_POD.png', dpi=300)
        
        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        x_data = grid[0,:,:,z_slice]/self.delta_h
        y_data = grid[1,:,:,z_slice]/self.delta_h
        z_data = grid[2,:,:,z_slice]/self.delta_h
        y_data_norm      = np.asarray( y_data.flatten() )
        x_data_norm      = np.asarray( x_data.flatten() )
        

        ## ORIGINAL DATA
        u_data           = orig_field[:, :, z_slice]
        u_data_norm      = np.asarray( u_data.flatten() )
        u_min = np.min(u_data_norm)
        u_max = np.max(u_data_norm)
        print(u_min)
        print(u_max)
        u_min = -0.4
        u_max = 0.4

        # Clear plot
        plt.clf()
        pi = math.pi

        # Plot data
        #my_cmap = parula_map
        my_norm = colors.Normalize( vmin = u_min, vmax = u_max )
        cs = plt.tricontourf( x_data_norm, y_data_norm, u_data_norm, cmap = "bwr", norm = my_norm, levels = np.arange( u_min, u_max + 1e-6, 1.0e-3 ) )

        # Colorbar
        cbar = plt.colorbar( cs, shrink = 0.14, pad = 0.02, ticks = [u_min,0,u_max] , aspect=5)
        cbar.ax.tick_params( labelsize = 9 )
        plt.xlim(0.0, 4*pi)
        plt.xticks([0.0, pi, 2*pi, 3*pi, 4*pi],[ r'${0.0}$',  r'${\pi}$',  r'${2 \pi}$',  r'${3 \pi}$',  r'${4 \pi}$'])
        plt.tick_params( axis = 'x', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.ylim( 0.0, 2)
        plt.yticks( np.arange( 0.0, 2.01, 1.0 ) )
        plt.tick_params( axis = 'y', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.gca().set_aspect( 'equal', adjustable = 'box' )
        ax = plt.gca()
        ax.tick_params( axis = 'both', pad = 7.5 )
        plt.text( 12.9, 2.05, r'${u^{\prime}}^{+}$', fontsize = 9 )

        plt.xlabel( r'${x/\delta}$', size = 9)
        plt.ylabel( r'${y/\delta}$', size = 9 )

        plt.savefig(f'figures/XY_POD_original_u_{snapshot_idx}.png', format = 'png', bbox_inches = 'tight', dpi=600 )


        ## RECONSTRUCTED
        u_data         = recon_field[:, :, z_slice]
        u_data_norm    = np.asarray( u_data.flatten() )
        u_min = np.min(u_data_norm)
        u_max = np.max(u_data_norm)
        u_min = -0.4
        u_max = 0.4

        # Clear plot
        plt.clf()
        
        # Plot data
        #my_cmap = parula_map
        my_norm = colors.Normalize( vmin = u_min, vmax = u_max )
        cs = plt.tricontourf( x_data_norm, y_data_norm, u_data_norm, cmap = "bwr", norm = my_norm, levels = np.arange( u_min, u_max + 1e-6, 1.0e-3 ) )

        # Colorbar
        cbar = plt.colorbar( cs, shrink = 0.14, pad = 0.02, ticks = [u_min,0,u_max], aspect=5)
        cbar.ax.tick_params( labelsize = 9 )
        plt.xlim(0.0, 4*pi)
        plt.xticks([0.0, pi, 2*pi, 3*pi, 4*pi],[ r'${0.0}$',  r'${\pi}$',  r'${2 \pi}$',  r'${3 \pi}$',  r'${4 \pi}$'])
        plt.tick_params( axis = 'x', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.ylim( 0.0, 2)
        plt.yticks( np.arange( 0.0, 2.01, 1.0 ) )
        plt.tick_params( axis = 'y', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.gca().set_aspect( 'equal', adjustable = 'box' )
        ax = plt.gca()
        ax.tick_params( axis = 'both', pad = 7.5 )
        plt.text( 12.9, 2.05, r'${u^{\prime}}^{+}$', fontsize = 9 )

        plt.xlabel( r'${x/\delta}$', size = 9)
        plt.ylabel( r'${y/\delta}$', size = 9 )

        plt.savefig(f'figures/XY_POD_reconstructed_u_{snapshot_idx}.png', format = 'png', bbox_inches = 'tight', dpi=600 )

    def plot_spectrum_POD(self,eigvals):
        plt.clf()
        plt.semilogy(np.cumsum(eigvals) / np.sum(eigvals))
        plt.xlabel("Number of POD Modes")
        plt.ylabel("$E(r)$")
        plt.savefig(f'figures/Spectrum_POD.png', dpi=300)
    
    def plot_POD_mode_structure(self, Phi, mode_idx, var_idx, label, y_pb, slice_axis='z'):
        """
        Plot the spatial structure of a POD mode for a specific variable.

        Parameters:
            Phi: POD spatial modes matrix (M x N)
            mode_idx: POD mode index to visualize (default: 0)
            var_idx: variable index (0=rho, 1=u, 2=v, 3=w, 4=T)
            slice_axis: 'x', 'y', or 'z' — axis to slice at center
        """
        nx, ny, nz = int(self.n_x), int(self.n_y), int(self.n_z)
        Q = self.Delta_volume
        len_delta = Q.shape[0]

        # Extract the relevant portion of the POD mode
        ini = int(var_idx * len_delta)
        fin = int((var_idx + 1) * len_delta)
        mode_data = Phi[ini:fin, mode_idx].reshape((nx, ny, nz))

        # Choose a slice
        if slice_axis == 'z':
            slice_idx = int(nz // 2)
            mode_slice = mode_data[:, :, slice_idx]
            axis_label = 'Z'
        elif slice_axis == 'y':
            slice_idx = int(ny // 2)
            mode_slice = mode_data[:, slice_idx, :]
            axis_label = 'Y'
        elif slice_axis == 'x':
            slice_idx = int(nx // 2)
            mode_slice = mode_data[slice_idx, :, :]
            axis_label = 'X'
        else:
            raise ValueError("slice_axis must be 'x', 'y', or 'z'")

        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        x_data = grid[0,:,:,slice_idx]/self.delta_h
        y_data = grid[1,:,:,slice_idx]/self.delta_h
        z_data = grid[2,:,:,slice_idx]/self.delta_h
        y_data_norm      = np.asarray( y_data.flatten() )
        x_data_norm      = np.asarray( x_data.flatten() )


        ## ORIGINAL DATA
        u_data           = mode_slice
        # Normalize to O(1)
        u_data = mode_slice.copy()
        u_data /= np.max(np.abs(u_data))  # Max absolute value = 1
        u_data_norm = np.asarray(u_data.flatten())
        vmin, vmax = -1.0, 1.0

        # Clear plot
        plt.clf()
        pi = math.pi

        # Plot data
        #my_cmap = parula_map
        my_norm = colors.Normalize( vmin = vmin, vmax = vmax )
        cs = plt.tricontourf( x_data_norm, y_data_norm, u_data_norm, cmap = "bwr", norm = my_norm, levels = np.linspace(vmin, vmax, 100) )
        
        # Pseudoboiling line
        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
        y_pb_smooth = smooth(y_pb,2)
        y_pb_smooth[0:1] = y_pb[0:1]
        y_pb_smooth[-2:] = y_pb[-2:]
        # Contours
        if mode_idx == 0 or mode_idx == 1:
            contour_levels = [-0.5, 0.0, 0.5]
            linestyles = ['dotted', 'dashed', 'dashdot']  # different linestyles for each level
            u_data_norm  = gaussian_filter(u_data_norm, sigma=1)
            for level, ls in zip(contour_levels, linestyles):
                contours = plt.tricontour(x_data_norm, y_data_norm, u_data_norm,levels=[level], colors='black',linestyles=ls,linewidths=0.25)
            y_pb_plot = y_pb_smooth[1:-1]
            plt.plot(x_data[:,0],y_pb_plot[::2]/self.delta_h,linestyle = ':', linewidth = 0.5, color = 'darkviolet', zorder = 1)
        # Colorbar
        cbar = plt.colorbar( cs, shrink = 0.14, pad = 0.02, ticks = [vmin,0,vmax], aspect=5 )
        cbar.ax.tick_params( labelsize = 9 )
        plt.xlim(0.0, 4*pi)
        plt.xticks([0.0, pi, 2*pi, 3*pi, 4*pi],[ r'${0.0}$',  r'${\pi}$',  r'${2 \pi}$',  r'${3 \pi}$',  r'${4 \pi}$'])
        plt.tick_params( axis = 'x', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.ylim( 0.0, 2)
        plt.yticks( np.arange( 0.0, 2.01, 1.0 ) )
        plt.tick_params( axis = 'y', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.gca().set_aspect( 'equal', adjustable = 'box' )
        ax = plt.gca()
        ax.tick_params( axis = 'both', pad = 7.5 )
        #plt.text( 12.9, 2.05, r'${u^{\prime}}^{+}$', fontsize = 9 )
        plt.text( 12.9, 2.05, label, fontsize = 9 )

        plt.xlabel( r'${x/\delta}$', size = 9)
        plt.ylabel( r'${y/\delta}$', size = 9 )

        plt.savefig(f'figures/POD_Mode_{mode_idx}_var_{var_idx}.png', format = 'png', bbox_inches = 'tight', dpi=600 )

        # Plot the POD mode structure
        #plt.clf()
        #plt.figure(figsize=(6, 5))e
        #plt.imshow(mode_slice.T, origin='lower', cmap='bwr')
        #plt.colorbar()
        #plt.title(f"POD Mode #{mode_idx}, Variable #{var_idx}, {axis_label}={slice_idx}")
        #plt.xlabel("X-axis")
        #plt.ylabel("Y-axis")
        #plt.tight_layout()
        #plt.show()
        #plt.savefig(f'figures/POD_Mode_{mode_idx}_var_{var_idx}.png', dpi=300)
        #plt.savefig(f'figures/POD_Mode_{mode_idx}.png', dpi=300)

    def plot_mode_energy_map(self, Phi, mode_index, slice_axis='z', slice_index=None, cmap=parula_map):
        """
        Plot spatial energy map (sum of squares over variables) for a given POD mode.

        Parameters:
            Phi : np.ndarray
                POD modes matrix of shape (5 * nx * ny * nz, num_modes)
            mode_index : int
                Index of the mode to visualize
            nx, ny, nz : int
                Grid dimensions
            slice_axis : str
                One of 'x', 'y', or 'z' to specify slice plane
            slice_index : int or None
                Index of the slice in that direction; if None, defaults to mid-plane
            cmap : str
                Colormap for plotting
        """

        nx, ny, nz = int(self.n_x), int(self.n_y), int(self.n_z)
        n_var = 5
        len_delta = nx * ny * nz

        # Extract mode vector and compute energy contribution per grid point
        mode = Phi[:, mode_index]
        energy_map = np.zeros(len_delta)

        for v in range(n_var):
            ini = int(v * len_delta)
            fin = int((v + 1) * len_delta)
            energy_map += mode[ini:fin] ** 2

        # Reshape to 3D grid
        energy_map = energy_map.reshape((nx, ny, nz))

        # Default slice index
        if slice_index is None:
            if slice_axis == 'x':
                slice_index = int(nx // 2)
            elif slice_axis == 'y':
                slice_index = int(ny // 2)
            elif slice_axis == 'z':
                slice_index = int(nz // 2)

        # Extract 2D slice
        if slice_axis == 'x':
            slice_data = energy_map[slice_index, :, :]
            xlabel, ylabel = 'y', 'z'
        elif slice_axis == 'y':
            slice_data = energy_map[:, slice_index, :]
            xlabel, ylabel = 'x', 'z'
        elif slice_axis == 'z':
            slice_data = energy_map[:, :, slice_index]
            xlabel, ylabel = 'x', 'y'
        else:
            raise ValueError("slice_axis must be one of 'x', 'y', 'z'")

        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        x_data = grid[0,:,:,slice_index]/self.delta_h
        y_data = grid[1,:,:,slice_index]/self.delta_h
        z_data = grid[2,:,:,slice_index]/self.delta_h
        y_data_norm      = np.asarray( y_data.flatten() )
        x_data_norm      = np.asarray( x_data.flatten() )


        ## ORIGINAL DATA
        u_data           = slice_data
        u_data_flat      = np.asarray( u_data.flatten() )
        max_energy       = np.max(u_data_flat)
        if max_energy == 0:
            print("Warning: zero energy slice, skipping plot.")
            return
        u_min, u_max = 0, max_energy  # energy is positive, so range is 0 to max

        # Ticks and labels
        power = int(np.floor(np.log10(max_energy)))
        scale = 10**power

        # Scale ticks to O(1) range
        #ticks = [0, 0.75*scale, max_energy]
        #ticks_scaled = [t/scale for t in ticks]
        ticks = [u_min, 0.5*u_max, u_max]
        #ticks_scaled = [tick / scale for tick in ticks]
        
        def fmt(x, pos):
            return f"{x / scale:.1f}"

        # Format labels to 1 decimal place
        #ticks_labels = [f"{tick:.1f}" for tick in ticks]
        # Round ticks_scaled to 1 decimal place
        #ticks_labels = [f"{tick:.1f}" for tick in ticks_scaled]
        # Clear plot
        plt.clf()
        pi = math.pi

        # Plot data
        #my_cmap = parula_map
        my_norm = colors.Normalize( vmin = u_min, vmax = u_max )
        cs = plt.tricontourf( x_data_norm, y_data_norm, u_data_flat, cmap = "bwr", norm = my_norm, levels=256)

        # Colorbar
        cbar = plt.colorbar( cs, shrink = 0.14, pad = 0.02, ticks = ticks, aspect=5 )
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))
        #cbar.ax.set_yticklabels(ticks_labels)
        cbar.ax.tick_params( labelsize = 9 )
        #cbar.ax.set_ylabel(r"$|E| \times 10^{%d}$" % power, fontsize=12)
        # Remove the default ylabel to avoid confusion
        cbar.ax.set_ylabel('')
        # Add label text on top, centered horizontally above the colorbar
        #cbar.ax.text(0.5, 1.05, r"$|E| \times 10^{%d}$" % power,
                     #ha='center', va='bottom', fontsize=12, transform=cbar.ax.transAxes)
        plt.text( 12.9, 2.05, r"$|E| \times 10^{%d}$" % power, fontsize = 9 )
        
        # Plot lims
        plt.xlim(0.0, 4*pi)
        plt.xticks([0.0, pi, 2*pi, 3*pi, 4*pi],[ r'${0.0}$',  r'${\pi}$',  r'${2 \pi}$',  r'${3 \pi}$',  r'${4 \pi}$'])
        plt.tick_params( axis = 'x', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.ylim( 0.0, 2)
        plt.yticks( np.arange( 0.0, 2.01, 1.0 ) )
        plt.tick_params( axis = 'y', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
        plt.gca().set_aspect( 'equal', adjustable = 'box' )
        ax = plt.gca()
        ax.tick_params( axis = 'both', pad = 7.5 )
        #plt.text( 12.9, 2.05, r'${u^{\prime}}^{+}$', fontsize = 9 )

        plt.xlabel( r'${x/\delta}$', size = 9)
        plt.ylabel( r'${y/\delta}$', size = 9 )
        # Plot
        #plt.clf()
        #plt.figure(figsize=(6, 5))
        #im = plt.imshow(slice_data.T, origin='lower', cmap=cmap, aspect='auto')
        #plt.title(f"Energy Map of POD Mode {mode_index} (slice {slice_axis}={slice_index})")
        #plt.xlabel(xlabel)
        #plt.ylabel(ylabel)
        #plt.colorbar(im, label='Energy')
        #plt.tight_layout()
        #plt.show()
        #plt.savefig(f'figures/POD_Energy_Mode_{mode_idx}_Map.png', dpi=300)
        plt.savefig(f'figures/POD_Energy_Mode_{mode_idx}_Map.png', format = 'png', bbox_inches = 'tight', dpi=600 )


    def compute_energy_convergence(self, eigvals):

        # Sort descending, in case it wasn't saved already sorted
        eigvals = np.sort(eigvals)[::-1]

        # Compute cumulative energy
        cumulative_energy = np.cumsum(eigvals)
        total_energy = cumulative_energy[-1]
        energy_fraction = cumulative_energy / total_energy

        # Plot
        plt.clf()
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1, len(eigvals)+1), energy_fraction, color = 'royalblue', label='Cumulative Energy')
        
        thresholds=[0.9, 0.95, 0.99]
        for t in thresholds:
            r = np.searchsorted(energy_fraction, t) + 1
            plt.axhline(t, color='gray', linestyle=':')
            plt.axvline(r, color='firebrick', linestyle='--')
            plt.plot(r, energy_fraction[r - 1], 'o', color='forestgreen', markersize=6, label=f'{int(t*100)}% marker')
            print(f"Modes for threshold {t} = {r}")
            #plt.text(r + 10, t - 0.05, f"{int(t*100)}% @ {r} modes", color='black')


        plt.xlabel("Number of modes $r$")
        plt.ylabel("$E(r)$")
        #plt.title("POD Energy Convergence")
        #plt.grid(True)
        #plt.legend()
        plt.xlim(0.0, len(eigvals))
        plt.tight_layout()

        plt.savefig(f'figures/Energy_convergence.png', dpi=300)
        print(f"Saved energy convergence plot to figures")


        # Spectrum
        # Eigenvalue spectrum: lambda_i / lambda_max vs index
        normalized_eigvals = eigvals / eigvals[0]
        
        # Improved moving average with edge padding
        def smooth_with_padding(x, window_size=11):
            pad_size = window_size // 2
            padded = np.pad(x, pad_size, mode='edge')  # replicate edges
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(padded, kernel, mode='valid')  # ensures same length as input
            return smoothed

        # Apply smoothing
        window_size = 11  # Odd number; balance between smoothing and retaining details
        smoothed_eigvals = smooth_with_padding(normalized_eigvals, window_size=window_size)

        # Plot
        plt.clf()
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1, len(eigvals)+1), smoothed_eigvals, color = 'royalblue', label='Cumulative Energy')

        plt.yscale('log')  # Optional: log scale helps show decay more clearly
        plt.xscale('log')  # Optional: log scale helps show decay more clearly
        plt.xlabel("Eigenvalue index $i$")
        plt.ylabel(r"$\lambda_i / \lambda_{\max}$")
        #plt.title("Eigenvalue Spectrum")
        plt.xlim(1.0, len(eigvals))
        plt.ylim(1E-8,1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        plt.savefig('figures/Eigenvalue_spectrum.png', dpi=300)
        print("Saved eigenvalue spectrum plot to figures")

        return energy_fraction


    def preprocess_wall_units(self):

        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        y_data_bw = grid[1,0,:,0]                    # BW
        y_data_tw = (2*self.delta_h - grid[1,0,:,0]) # TW
                                   
        y_plus_bw = y_data_bw*(self.u_tau_bw/(self.mu_bw/self.rho_bw))
        y_plus_tw = y_data_tw*(self.u_tau_tw/(self.mu_tw/self.rho_tw))

        return y_plus_bw, y_plus_tw


if __name__ == "__main__":

    ### DEFINED INPUTS ###
    # Define domain length
    delta     = 100*1e-6           # Fixed channel half height
    L_x       = 4*math.pi*delta
    L_z       = 4/3*math.pi*delta
    save_dir  = f"../data_resolvent/data_3d" ##f"../FFT_time/data"            # Store DMD outputs
    delta     = 100*1e-6           # Fixed channel half height
    nit       = 2500              # Snapshot each 50000 iterations 
    time_step = 8*1e-10           # Fixed DNS time step 
    dt        = nit*time_step     # Snapshots sample rate
    u_tau_bw  = 0.19              # Bottom Wall friction velocity
    mu_bw     = 1.5312*1E-05    
    rho_bw    = 148.04
    u_tau_tw  = 0.19               # Top Wall friction velocity
    mu_tw     = 1.5312*1E-05    
    rho_tw    = 148.04

    # Snapshots (on file *.h5)
    cases = []
    for file in glob.glob("../data_resolvent/*.h5"):
        cases.append(file[18:-3])
    cases.sort()
    #cases = cases[0:2]
    #print(cases)

    ### DEFINE FFT CLASS ###
    POD_class = POD_process(save_dir,cases,dt, delta, L_x, L_z, 
            u_tau_bw, mu_bw, rho_bw, u_tau_tw, mu_tw, rho_tw)
    

    ## Compute POD
    print("POD computation...")
    if not(os.path.exists(f"{save_dir}/Snapshots_{len(cases)}.npy")):
        print("Computing POD...")
        POD_class.compute_POD()
    print("POD from file...")
    eigvals = np.load(f"{save_dir}/POD_eigvals.npy", mmap_mode='r')
    eigvecs = np.load(f"{save_dir}/POD_eigvecs.npy", mmap_mode='r')
    Phi     = np.load(f"{save_dir}/POD_modes.npy", mmap_mode='r')
    #X_weighted    = np.load(f"{save_dir}/POD_X_weighted.npy", mmap_mode='r') 
    X_unweighted  = np.load(f"{save_dir}/POD_X_unweighted.npy", mmap_mode='r') 
    
    POD_class.compute_energy_convergence(eigvals)
    
    # Reduced order reconstruction
    print("POD data loaded...")
    r     = 450
    X_rec_unweighted = POD_class.reconstruct_POD_modes(eigvals, eigvecs, Phi, r)
    error_rel = np.linalg.norm(X_unweighted - X_rec_unweighted, 'fro') / np.linalg.norm(X_unweighted, 'fro')
    print(f"Relative reconstruction error: {error_rel:.4e}")
    var_reconstruction = 1 # Streamwise (0,1,2,3,4)
    idx_snap           = 0 # First snapshot
    POD_class.plot_reconstructed_POD(X_rec_unweighted, idx_snap, var_reconstruction)
    #POD_class.plot_spectrum_POD(eigvals)
    print("Computing POD mode structures and energy maps...")
    var_names = [r'$\rho^{\prime}$', r'$u^{\prime}$', r'$v^{\prime}$', r'$w^{\prime}$', r'$T^{\prime}$']
    for mode_idx in range(0,10):
        print(f"Mode idx = {mode_idx}")
        for var_idx in range(0,5):
            POD_class.plot_POD_mode_structure(Phi,mode_idx, var_idx, var_names[var_idx], POD_class.pseudoboiling[int(idx_snap),:])
        POD_class.plot_mode_energy_map(Phi,mode_idx)

