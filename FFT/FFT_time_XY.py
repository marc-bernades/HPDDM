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

def get_DNS(d_dir, dataset_name):
    ########## OPEN DATA FILES ##########
    data_file = h5py.File( f"../../data_resolvent/{dataset_name}.h5", 'r' )
    rho_data        = data_file['rho'][:,:,:]
    u_data          = data_file['u'][:,:,:]
    v_data          = data_file['v'][:,:,:]
    w_data          = data_file['w'][:,:,:]
    T_data          = data_file['T'][:,:,:]
    x_data          = data_file['x'][:,:,:]
    y_data          = data_file['y'][:,:,:]
    z_data          = data_file['z'][:,:,:]
    num_points_x    = x_data[0,0,:].size
    num_points_y    = y_data[0,:,0].size
    num_points_z    = z_data[:,0,0].size
   
    # Define container
    n_vars     = 5
    output_var = np.zeros((int(n_vars),int( num_points_z - 2 ),int( num_points_y - 2),int( num_points_x - 2) ))
    output_var[0,:,:,:] = rho_data[1:-1,1:-1,1:-1]
    output_var[1,:,:,:] = u_data[1:-1,1:-1,1:-1]
    output_var[2,:,:,:] = v_data[1:-1,1:-1,1:-1]
    output_var[3,:,:,:] = w_data[1:-1,1:-1,1:-1]
    output_var[4,:,:,:] = T_data[1:-1,1:-1,1:-1]
    
    # Convert to x, z, y (for FFT space on periodic dimensions kx, kz, y)
    for idx in range(0,n_vars):
        output_var[idx,:,:,:] = np.einsum("ijk->kij",output_var[idx,:,:,:])

    os.system(f"mkdir -p {d_dir}")
    np.save( f"{d_dir}/{dataset_name}.npy", output_var)
    
    return output_var


def get_Metrics(d_dir, dataset_name, delta):
    ########## OPEN DATA FILES ##########
    data_file = h5py.File( f"../../data_resolvent/{dataset_name}.h5", 'r' )
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

    ### PLOT
    # Clear plot
    plt.clf()

    plt.scatter( y_data[0,:,0]/delta, avg_u/u_b,  marker = '+', s = 25, color = 'royalblue', zorder = 1, label = r'$\textrm{Base}$')

    # Configure plot
    plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = True, labeltop = False, direction = 'in' )
    plt.xlim( 0.0, 2.0 )
    plt.xticks( np.arange( 0.0, 2.1, 0.5 ) )
    #plt.xscale( 'log' )
    plt.xlabel( r'$y/\delta$' )
    plt.tick_params( axis = 'y', left = True, right = True, labelleft = True, labelright = False, direction = 'in' )
    plt.ylim( 0.0, 1.2 )
    plt.yticks( np.arange( 0.0, 1.21, 0.4 ) )
    #plt.yscale( 'log' )
    plt.ylabel( r'$u/u_b$' )
    legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left',fontsize=15 )
    #plt.tick_params( axis = 'both', pad = 7.5 )
    plt.savefig( f'figures/Cond_phase_speed_XY/u_vs_y.pdf', format = 'pdf', bbox_inches = 'tight',dpi=300 )

    ### PLOT
    # Clear plot
    plt.clf()
    
    #plt.plot( avg_y_plus_bw[:max_y_index_bw], avg_u_plus_bw[:max_y_index_bw], linestyle = '-', marker = 'o', markersize = 5, color = 'royalblue', zorder = 1, label = r'${bw}$' )
    #plt.plot( avg_y_plus_tw[:max_y_index_tw], avg_u_plus_tw[:max_y_index_tw], linestyle = '-', marker = 'x', markersize = 5, color = 'firebrick', zorder = 1, label = r'${tw}$' )
    plt.scatter( avg_y_plus_bw[:max_y_index_bw], avg_u_plus_bw[:max_y_index_bw], marker = 'o', s = 25, color = 'royalblue', zorder = 1, label = r'${bw}$' )
    plt.scatter( avg_y_plus_tw[:max_y_index_tw], avg_u_plus_tw[:max_y_index_tw], marker = 'x', s = 25, color = 'firebrick', zorder = 1, label = r'${tw}$' )

    # Configure plot
    plt.tick_params( axis = 'x', bottom = True, top = True, labelbottom = True, labeltop = False, direction = 'in' )
    plt.xlim( 0.1, 200.0 )
    #plt.xticks( np.arange( 0.0, 2.1, 0.5 ) )
    plt.xscale( 'log' )
    plt.xlabel( r'$y^+$' )
    plt.tick_params( axis = 'y', left = True, right = True, labelleft = True, labelright = False, direction = 'in' )
    plt.ylim( 0.0, 20.0 )
    plt.yticks( np.arange( 0.0, 20.1, 5.0) )
    #plt.yscale( 'log' )
    plt.ylabel( r'$u^+$' )
    legend = plt.legend( shadow = False, fancybox = False, frameon = False, loc='upper left',fontsize=15 )
    #plt.tick_params( axis = 'both', pad = 7.5 )
    plt.savefig( f'figures/Cond_phase_speed_XY/u_plus_vs_y_plus.pdf', format = 'pdf', bbox_inches = 'tight',dpi=300 )


    return u_b, u_tau_bw, u_tau_tw, rho_bw, rho_tw, mu_bw, mu_tw, T_tau_bw, T_tau_tw, avg_u, avg_rho, avg_mu, avg_y_plus_bw, avg_u_plus_bw, avg_y_plus_tw, avg_u_plus_tw, max_y_index_bw, max_y_index_tw


def save_DNS_grid(d_dir,dataset_name):

    ########## OPEN DATA FILES ##########
    print("Saving DNS grid...")
    data_file = h5py.File( f"../../data_resolvent/{dataset_name}.h5", 'r' )
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

def fft_space(output_var, L_x):
    """
    Read a specific time snapshot from the data and convert it from physical
    to spectral space. dimensions are (nx, nz, ny).

    Parameters:
    u (np.ndarray): The x-velocity field dimensioned (nx, nz, ny).
    v (np.ndarray): The y-velocity field dimensioned (nx, nz, ny).
    w (np.ndarray): The z-velocity field dimensioned (nx, nz, ny).
    """
    #rho = output_var[0,:,:,:]
    u   = output_var
    #v   = output_var[2,:,:,:]
    #w   = output_var[3,:,:,:]
    #T   = output_var[4,:,:,:]

    # Number of points
    nk_x = u.shape[0]
    # grid spacing
    dx = L_x/nk_x

    kx = np.fft.fftfreq(nk_x, d=dx)*2*math.pi 
    #print(kx)
    #sys.exit()
    #kx = kx[kx>0]
    #kz = kz[kz>0]
    
    #rho_hat = np.fft.fftn(rho, axes=(0, 1)) / nk_x / nk_z
    #u_hat   = np.fft.fftn(u, axes=(0,))  #/ nk_x 
    u_hat   = fftn(u, axes=(0,), workers = -1)  #/ nk_x 
    #v_hat   = np.fft.fftn(v, axes=(0, 1))   / nk_x / nk_z
    #w_hat   = np.fft.fftn(w, axes=(0, 1))   / nk_x / nk_z
    #T_hat   = np.fft.fftn(T, axes=(0, 1))   / nk_x / nk_z


    # Store in container
    #output_hat = np.zeros((output_var.shape), dtype=complex)
    #output_hat[0,:,:,:] = rho_hat
    output_hat = u_hat
    #output_hat[2,:,:,:] = v_hat
    #output_hat[3,:,:,:] = w_hat
    #output_hat[4,:,:,:] = T_hat
    
    return output_hat, kx

def fft_time(output_hat_space, kx, dt, n_snapshots):



    freq   = np.fft.fftfreq(n_snapshots, d=dt)
    #freq   = freq[freq > 0]
    omega  = 2*math.pi*freq
    #omega   = np.fft.fftfreq(n_snapshots, d=dt)
    #omega   = omega[omega > 0]

    #rho = output_hat_space[0,:,:,:,:]
    u   = output_hat_space[:,:,:]
    #v   = output_hat_space[2,:,:,:,:]
    #w   = output_hat_space[3,:,:,:,:]
    #T   = output_hat_space[4,:,:,:,:]
    
    #rho_hat = np.fft.fftn(rho, axes=(3,)) / n_snapshots
    #u_hat   = np.fft.fftn(u,   axes=(2,)) #/ n_snapshots
    u_hat   = fftn(u,   axes=(2,), workers = -1) #/ n_snapshots
    #v_hat   = np.fft.fftn(v,   axes=(3,)) / n_snapshots
    #w_hat   = np.fft.fftn(w,   axes=(3,)) / n_snapshots
    #T_hat   = np.fft.fftn(T,   axes=(3,)) / n_snapshots

    # Store in container
    output_hat = np.zeros((output_hat_space.shape), dtype=complex)
    #output_hat[0,:,:,:,:] = rho_hat
    output_hat[:,:,:] = u_hat
    #output_hat[2,:,:,:,:] = v_hat
    #output_hat[3,:,:,:,:] = w_hat
    #output_hat[4,:,:,:,:] = T_hat

    return output_hat, omega


def inv_fft_space(kx, output_hat, L_x):
    
    #rho_hat = output_hat[0,:,:,:]
    u_hat   = output_hat[:,:,:]
    #v_hat   = output_hat[2,:,:,:]
    #w_hat   = output_hat[3,:,:,:]
    #T_hat   = output_hat[4,:,:,:]

    # Number of points
    nk_x = u_hat.shape[0]
    # grid spacing
    dx = L_x/nk_x
    #print(np.fft.ifftfreq(nk_x, d=dx))
    #print(np.fft.ifftfreq(nk_z, d=dz))   
    
    #rho_inv_fft = np.fft.ifftn(rho_hat, axes=(0, 1)) * nk_x * nk_z
    #u_inv_fft   = np.fft.ifftn(u_hat, axes=(0,))  #* nk_x
    u_inv_fft   = ifftn(u_hat, axes=(0,), workers=-1)  #* nk_x
    #v_inv_fft   = np.fft.ifftn(v_hat, axes=(0, 1))   * nk_x * nk_z
    #w_inv_fft   = np.fft.ifftn(w_hat, axes=(0, 1))   * nk_x * nk_z
    #T_inv_fft   = np.fft.ifftn(T_hat, axes=(0, 1))   * nk_x * nk_z

    # Store in container
    output_inv_fft = np.zeros((output_hat.shape), dtype=complex)
    #output_inv_fft[0,:,:,:] = np.real(rho_inv_fft)
    output_inv_fft[:,:,:] = np.real(u_inv_fft)
    #output_inv_fft[2,:,:,:] = np.real(v_inv_fft)
    #output_inv_fft[3,:,:,:] = np.real(w_inv_fft)
    #output_inv_fft[4,:,:,:] = np.real(T_inv_fft)
   
    return output_inv_fft

def inv_fft_time(output_hat, omega, dt):
    
    #rho_hat = output_hat[0,:,:,:,:]
    u_hat   = output_hat[:,:,:,]
    #v_hat   = output_hat[2,:,:,:,:]
    #w_hat   = output_hat[3,:,:,:,:]
    #T_hat   = output_hat[4,:,:,:,:]

    # Number of points
    n_snapshots = u_hat.shape[2]
    
    #rho_inv_fft = np.fft.ifftn(rho_hat, axes=(3,)) * n_snapshots
    #u_inv_fft   = np.fft.ifftn(u_hat,   axes=(2,)) #* n_snapshots
    u_inv_fft   = ifftn(u_hat,   axes=(2,), workers=-1) #* n_snapshots
    #v_inv_fft   = np.fft.ifftn(v_hat,   axes=(3,)) * n_snapshots
    #w_inv_fft   = np.fft.ifftn(w_hat,   axes=(3,)) * n_snapshots
    #T_inv_fft   = np.fft.ifftn(T_hat,   axes=(3,)) * n_snapshots

    # Store in container
    output_inv_fft = np.zeros((output_hat.shape), dtype=complex)
    #output_inv_fft[0,:,:,:,:] = (rho_inv_fft)
    output_inv_fft[:,:,:] = (u_inv_fft)
    #output_inv_fft[2,:,:,:,:] = (v_inv_fft)
    #output_inv_fft[3,:,:,:,:] = (w_inv_fft)
    #output_inv_fft[4,:,:,:,:] = (T_inv_fft)
   
    return output_inv_fft


def turbulent_kinetic_energy(y_plus_TKE_bw, y_plus_TKE_tw, kx, kz, output_hat, save_dir,dataset, y_plus_bw, y_plus_tw):
        
        
    # Half-numbers
    nx_max = int(0.5*len(kx)-1)
    nz_max = int(0.5*len(kz)-1)

    ### Bottom wall
    # Find idx y
    idx = np.zeros((y_plus_TKE_bw.size), dtype = int)
    idx[0] = int(np.argmin((np.abs(y_plus_bw - y_plus_TKE_bw[0]))))
    idx[1] = int(np.argmin((np.abs(y_plus_bw - y_plus_TKE_bw[1]))))
        
    E_hat_0 = np.dot(output_hat[1,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[1,0:nx_max,0:nz_max,idx[0]])) + \
              np.dot(output_hat[2,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[2,0:nx_max,0:nz_max,idx[0]])) + \
              np.dot(output_hat[3,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[3,0:nx_max,0:nz_max,idx[0]]))

    E_hat_1 = np.dot(output_hat[1,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[1,0:nx_max,0:nz_max,idx[1]])) + \
              np.dot(output_hat[2,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[2,0:nx_max,0:nz_max,idx[1]])) + \
              np.dot(output_hat[3,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[3,0:nx_max,0:nz_max,idx[1]]))
        
    [KX, KZ] = np.meshgrid(kx[0:nx_max],kz[0:nz_max])
    fig, axs = plt.subplots(2,2, figsize=(5.4, 4), sharex=True, sharey='row')
    #axs[0, 0].set_title("$y^+ = " + str(idx[0]) + "}$", fontsize=9)
    #axs[0, 1].set_title("$y^+ = " + str(idx[1]) + "}$", fontsize=9)

    fig.subplots_adjust(wspace=0.15, hspace=0.1)
    fig.text(0.05, 0.5, r"$k_z$", va='center', rotation='vertical')
    fig.text(0.51, 0.01, r"$k_x$", ha='center')
        
    axs[0, 0].contourf(KX, KZ, E_hat_0, cmap='RdBu') # levels=np.linspace(min_val, max_val,100))     

    axs[0, 1].contourf(KX, KZ, E_hat_1, cmap='RdBu') # levels=np.linspace(min_val, max_val,100)) 

    ### Top wall
    # Find idx y
    idx = np.zeros((y_plus_TKE_tw.size), dtype = int)
    idx[0] = int(np.argmin((np.abs(y_plus_tw - y_plus_TKE_tw[0]))))
    idx[1] = int(np.argmin((np.abs(y_plus_tw - y_plus_TKE_tw[1]))))
        
    E_hat_0 = np.dot(output_hat[1,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[1,0:nx_max,0:nz_max,idx[0]])) + \
              np.dot(output_hat[2,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[2,0:nx_max,0:nz_max,idx[0]])) + \
              np.dot(output_hat[3,0:nx_max,0:nz_max,idx[0]],np.conj(output_hat[3,0:nx_max,0:nz_max,idx[0]]))

    E_hat_1 = np.dot(output_hat[1,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[1,0:nx_max,0:nz_max,idx[1]])) + \
              np.dot(output_hat[2,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[2,0:nx_max,0:nz_max,idx[1]])) + \
              np.dot(output_hat[3,0:nx_max,0:nz_max,idx[1]],np.conj(output_hat[3,0:nx_max,0:nz_max,idx[1]]))
        
    [KX, KZ] = np.meshgrid(kx[0:nx_max],kz[0:nz_max])
    fig, axs = plt.subplots(2,2, figsize=(5.4, 4), sharex=True, sharey='row')
    #axs[0, 0].set_title("$y^+ = " + str(idx[0]) + "}$", fontsize=9)
    #axs[0, 1].set_title("$y^+ = " + str(idx[1]) + "}$", fontsize=9)

    fig.subplots_adjust(wspace=0.15, hspace=0.1)
    fig.text(0.05, 0.5, r"$k_z$", va='center', rotation='vertical')
    fig.text(0.51, 0.01, r"$k_x$", ha='center')
        
    axs[1, 0].contourf(KX, KZ, E_hat_0, cmap='RdBu') # levels=np.linspace(min_val, max_val,100))     

    axs[1, 1].contourf(KX, KZ, E_hat_1, cmap='RdBu') # levels=np.linspace(min_val, max_val,100))  



    plt.savefig(f'figures/TKE/TKE_{dataset}.png', dpi=300)
    plt.close()




class FFT_process:
    def __init__(self, save_dir, cases, dt, delta, L_x, L_z,
             u_tau_bw, mu_bw, rho_bw, y_plus_TKE_bw,
             u_tau_tw, mu_tw, rho_tw, y_plus_TKE_tw,var_save):
        self.save_dir   = save_dir
        self.cases      = cases
        self.dt         = dt
        self.delta_h    = delta
        self.L_x        = L_x
        self.L_z        = L_z
        self.u_tau_bw   = u_tau_bw
        self.mu_bw      = mu_bw
        self.rho_bw     = rho_bw
        self.y_plus_TKE_bw = y_plus_TKE_bw
        self.u_tau_tw   = u_tau_tw
        self.mu_tw      = mu_tw
        self.rho_tw     = rho_tw
        self.y_plus_TKE_tw = y_plus_TKE_tw
        
        ## Obtain snapshots
        if not(os.path.exists(f"{self.save_dir}/snapshots_{var_save}.npy")):
            self.snapshots  = self.preprocess_snapshots(var_save)
        else:
            self.snapshots  = np.load(f"{self.save_dir}/snapshots_{var_save}.npy")
        ## Obtain pseudo-boiling
        if not(os.path.exists(f"{self.save_dir}/pseudoboiling.npy")):
            print("Computing pseudo boiling...")
            self.pseudoboiling  = self.preprocess_pseudoboiling()
        else:
            self.pseudoboiling  = np.load(f"{self.save_dir}/pseudoboiling.npy")


        ## Obtain bulk values
        print("Computing metrics and wall values...")
        (self.ub, self.u_tau_bw, self.u_tau_tw, self.rho_bw, self.rho_tw, self.mu_bw, self.mu_tw, self.T_tau_bw, self.T_tau_tw, 
        self.avg_u, self.avg_rho, self.avg_mu, 
        self.avg_y_plus_bw, self.avg_u_plus_bw, self.avg_y_plus_tw, self.avg_u_plus_tw,
        self.max_y_index_bw, self.max_y_index_tw) = get_Metrics(self.save_dir,f"{cases[0]}", self.delta_h)
        self.y_plus_bw, self.y_plus_tw        = self.preprocess_wall_units()
        ## Obtan DNS grid
        print("Loading DNS grid...")
        if not(os.path.exists(f"{self.save_dir}/grid.npy")):
            save_DNS_grid(self.save_dir,f"{cases[0]}")
        ## FFT Time and Space
        print("FFT in space and time: ")
        if not(os.path.exists(f"{self.save_dir}/output_fft_space_XY_{var_save}.npy")):
            print("Computing FFT...")
            self.output_fft_space, self.kx      = self.preprocess_fft_space(var_save)
            self.output_fft_time,  self.omega   = self.preprocess_fft_time(var_save)
        else:
            print("Taking FFT from file...")
            self.output_fft_space = np.load( f"{self.save_dir}/output_fft_space_XY_{var_save}.npy")
            self.output_fft_time  = np.load( f"{self.save_dir}/output_fft_time_XY_{var_save}.npy")
            self.kx    = np.load( f"{self.save_dir}/kx_XY.npy")
            self.omega = np.load( f"{self.save_dir}/omega_XY.npy")

        #self.plots_FFT_check()
        #self.plot_invFFT_recovery()

    def plot_invFFT_recovery(self):

        # Find omegas corresponding to velocity
        output_target_hat = self.output_fft_time

        # Inverse FFT in time
        inv_output_time_hat = inv_fft_time(output_target_hat, self.omega, self.dt)

        # Inverse FFT in space
        output_target = np.real(inv_fft_space(self.kx, self.kz, inv_output_time_hat, self.L_x, self.L_z))

        # Velocity first snapshot
        u_recovery = output_target[:,:,:,0]

        self.plot_c_target_XY(output_target[:,:,:,0], "recovery_invFFT")
        self.plot_c_target_XY(self.snapshots[:,:,:,0], "raw")




    def plots_FFT_check(self):

        # Plot the magnitude of the FFT of the first frame in spatial frequency domain
        plt.figure(figsize=(6, 6))
        plt.imshow(np.abs(np.fft.fftshift(self.output_fft_space[1,:,:,60,2])), cmap='jet', extent=(-len(self.kx)//2, len(self.kx)//2, -len(self.kz)//2, len(self.kz)//2))
        plt.colorbar()
        plt.title("FFT Magnitude (First Frame, Spatial Domain)")
        #plt.show()
        plt.savefig(f'figures/Test_space.png', format = 'png', bbox_inches = 'tight', dpi=600 )

        # Plot the magnitude of the FFT along the time axis for the first spatial frequency component
        plt.figure(figsize=(6, 6))
        plt.plot(np.abs(self.output_fft_time[1,len(self.kx)//2, len(self.kx)//2, 60, :]))  # Check the center spatial frequency component
        plt.title("FFT Magnitude Along Time Axis (Center Spatial Frequency Component)")
        plt.xlabel("Time Frames")
        plt.ylabel("Magnitude")
        #plt.show()
        plt.savefig(f'figures/Test_time.png', format = 'png', bbox_inches = 'tight', dpi=600 )


    def preprocess_wall_units(self):

        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        y_data_bw = grid[1,0,:,0]                    # BW
        y_data_tw = (2*self.delta_h - grid[1,0,:,0]) # TW
                                   
        y_plus_bw = y_data_bw*(self.u_tau_bw/(self.mu_bw/self.rho_bw))
        y_plus_tw = y_data_tw*(self.u_tau_tw/(self.mu_tw/self.rho_tw))

        return y_plus_bw, y_plus_tw

    def preprocess_pseudoboiling(self):
        
        nx, ny, n_snapshots = self.snapshots.shape 
        pseudoboiling  = np.zeros((n_snapshots,nx+2 ))  # y-position for each x

        n_t = 0
        for id, case in tqdm(enumerate(self.cases),total=len(self.cases)):

            ########## OPEN DATA FILES ##########
            data_file = h5py.File( f"../../data_resolvent/{case}.h5", 'r' )
            c_p_data        = data_file['c_p'][:,:,:]
            y_data          = data_file['y'][:,:,:]
            num_points_x    = c_p_data[0,0,:].size
            num_points_y    = c_p_data[0,:,0].size
            num_points_z    = c_p_data[:,0,0].size
        
            y_pb = np.zeros(num_points_x)
            for k in range( 0, num_points_x - 1 ):
                idx_pb   = np.argmax(np.abs(c_p_data[int(num_points_z/2),:,k]))
                y_pb[k] = y_data[0,idx_pb,0]

            # Fill data container
            pseudoboiling[n_t,:] = y_pb
            n_t += 1
        
        np.save( f"{self.save_dir}/pseudoboiling.npy", pseudoboiling)
    
        return pseudoboiling

    def preprocess_snapshots(self, var_save):
        # Initialize data container
        n_t   = 0
        n_snapshots = len(self.cases)
        print("Preprocessing snapshots...")

        for id, case in tqdm(enumerate(self.cases),total=len(self.cases)):
            ## Abstract DNS    
            if not(os.path.exists(f"{self.save_dir}/{case}.npy")):
                get_DNS(self.save_dir,f"{case}")
            output_DNS = np.load(f"{self.save_dir}/{case}.npy")
            n_dim, n_x, n_z, n_y = output_DNS.shape
            # Obtain delta_x, delta_y, delta_z
            if n_t == 0:
                snapshots   = np.zeros((n_dim, n_x, n_z, n_y, int(n_snapshots))) # Initialize data container
                mean_DNS    = np.zeros((n_y,n_dim))
                
            # Ensemble-averaged periodic dimensions
            for var in range (0,n_dim):
                mean_DNS[:,var] += np.mean(output_DNS[var,:,:,:], axis = (0,1))

            # Save snapshots
            snapshots[:,:,:,:,n_t]  = output_DNS
            n_t += 1

        # Subtract mean flow in rho, u and T
        mean_DNS *= 1.0/(n_t)

        for n_snapshots in range (0, n_t):
            for idx_x in range (0,n_x):
                for idx_z in range (0,n_z):
                    snapshots[0,idx_x,idx_z,:,n_snapshots] -= mean_DNS[:,0] 
                    snapshots[1,idx_x,idx_z,:,n_snapshots] -= mean_DNS[:,1] 
                    snapshots[4,idx_x,idx_z,:,n_snapshots] -= mean_DNS[:,4]

        #print("Snapshots preprocess completed...")

        #Export XY data at desires plane of streamwise velocity
        z_plane   = int(n_z/2)
        
         
        #var       = 1 
        snapshots = snapshots[var_save,:,z_plane,:,:]
        
        d_dir = self.save_dir
        np.save( f"{d_dir}/snapshots_{var_save}.npy", snapshots)


        return snapshots


            
    def preprocess_fft_space(self, var_save):
        n_t   = 0
        #output_fft_space = np.zeros((self.snapshots.shape), dtype=complex)
        #output_fft_space = sp.dok_matrix((self.snapshots.shape), dtype=complex)
        shape  = self.snapshots.shape
        #chunks = (32,32,64,1)
        output_fft_space = np.zeros(shape, dtype=complex)
        print("Preprocessing FFT space...")
        # Spatial FFT
        #for id, case in enumerate(self.cases):
        for id, case in tqdm(enumerate(self.cases), total=len(self.cases)):
            #print("Computing FFT Space: ", case)
            # FFT spatial
            fft_hat_space, kx = fft_space(self.snapshots[:,:,n_t], self.L_x)
            # Turbulent kinetic energy
            #y_plus_bw, y_plus_tw  = self.preprocess_wall_units()
            #turbulent_kinetic_energy(self.y_plus_TKE_bw, self.y_plus_TKE_tw, 
                    #kx, kz, fft_hat_space, self.save_dir, case, y_plus_bw, y_plus_tw)
            # Check recovery Inv FFT spatial
            # output_inv_fft = inv_fft_space(kx, kz, fft_hat_space, self.L_x, self.L_z)
            # Fill container
            output_fft_space[:,:,n_t] = fft_hat_space
            n_t += 1

        #print("Computing FFT Space chunks...")
        #output_fft_space = output_fft_space.compute()
        d_dir = self.save_dir
        np.save( f"{d_dir}/output_fft_space_XY_{var_save}.npy", output_fft_space)
        np.save( f"{d_dir}/kx_XY.npy", kx)
        #np.save( f"{d_dir}/kz.npy", kz)
        print("FFT Space completed...")
       
        return output_fft_space, kx

    def preprocess_fft_time(self, var_save):

        print("Computing FFT Time ...")
        # For each realization FFT in time
        output_fft_time, omega   = fft_time(self.output_fft_space, self.kx, self.dt, len(self.cases))
         
        d_dir = self.save_dir
        np.save( f"{d_dir}/output_fft_time_XY_{var_save}.npy", output_fft_time)
        np.save( f"{d_dir}/omega_XY.npy", omega)

        print("FFT Time completed...")

        return output_fft_time, omega


    def obtain_spectra_target(self,c_plus_target, y_plus_target,wall_str):

        print("Fourier condition to phase speed target...")

        # Phase speed in outer scales
        if wall_str == "bw":
            c_target = c_plus_target*self.u_tau_bw
        else:
            c_target = c_plus_target*self.u_tau_tw

        # c at target y+ position
        #c_pos2    = np.argmin(np.abs(y_plus_target - self.y_plus_tw))
        #print(y_plus_target)
        #print(f"y_plus = {self.y_plus_tw[c_pos2]}")
        #c_target2 = self.snapshots[0,c_pos2,0]
        #print(f"c_pos2 : {c_pos2}")
        #print(f"c_target2 : {c_target2}")       
        #print(f"c_target : {c_target}")
        # Define gate bandwidth
        #gate_bandwidth = 2*math.pi*10 # rad/s, the range around the desired frequency
        #gate_bandwidth = c_target*0.25 

        # Output target desired phase speed
        output_target_hat_gated = self.output_fft_time.copy() # np.zeros((self.output_fft_time.shape), dtype=complex)

        # Build c_map
        kx_inv = np.zeros_like(self.kx)
        kx_inv[1:] = 1 / self.kx[1:]
        kx_inv[0]  = 0
        c_map = self.omega[:,None]@kx_inv[None,:]
        # Reference value as c_map[:,2]
        idx_kx_ref = 2 # Equivalent to kx = 1 normalized (kx = 10000 *delta)
        #idx_omega_plus = int(c_map.shape[0]/2)
        c_ammend   = self.omega[(np.abs(c_map[:,idx_kx_ref] - c_target)).argmin()]*kx_inv[idx_kx_ref]
        #band_mask  = np.abs(c_map[:,idx_kx_ref] - c_target) < gate_bandwidth
        #pos_mask   = c_map[:,idx_kx_ref] > 0
        #if np.any(band_mask & pos_mask):
        #    c_ammend   = c_map[:,idx_kx_ref][band_mask & pos_mask]
        #else:
        #    c_ammend  = np.array([])
        #print(self.omega[(np.abs(c_map[:,idx_kx_ref] - c_target)).argmin()]*kx_inv[idx_kx_ref])
        print(f"Chosen c: {c_ammend}")

        if c_ammend.size > 0:
            c_idx = np.isin(c_map, c_ammend)
            idx_target = np.argwhere(~c_idx)
        else:
            idx_target = np.empty((0, 2), dtype=int)

        #c_idx      = c_map == c_ammend
        #idx_target = np.argwhere(~c_idx)

        # Set indexes to zero
        for omega_idx, kx_idx in idx_target:
            output_target_hat_gated[kx_idx,:,omega_idx] = 0.0

        # Find omegas corresponding to phase speed for each y
        #for idx_y in range (0,self.output_fft_time.shape[1]):
            #for idx_omega in range (0, len(self.omega)):
                # Find positions that match omega
                #idx_target = np.argmin(np.abs(self.kx/self.omega[idx_omega] - c_target))
                #idx_target = np.abs(self.omega[idx_omega] - self.kx/c_target) < gate_bandwidth
                    
                #for var in range (0,self.output_fft_time.shape[0]):
                   #output_target_hat[var,idx_target,idx_kz,idx_y,idx_omega] = self.output_fft_time[var,idx_target,idx_kz,idx_y,idx_omega]
                #output_target_hat_gated[~idx_target,idx_y,idx_omega] = 0

                
        # Inverse FFT in time
        inv_output_time_hat = inv_fft_time(output_target_hat_gated, self.omega, self.dt)

        # Inverse FFT in space
        output_target = np.zeros((self.output_fft_time.shape))
        output_target = np.real(inv_fft_space(self.kx, inv_output_time_hat, self.L_x))

        return output_target

    def plot_c_target_XY(self,output_target,c_plus_target,wall_str,n_snapshot,scaling, y_pb, var_save, var_sel):

        print("Plotting contour phase speed target in physical space...")
        # Norm wall units
        if wall_str == "bw":
            wall_metrics = [self.rho_bw, self.u_tau_bw, self.u_tau_bw, self.u_tau_bw, self.T_tau_bw]
            u_norm = wall_metrics[var_save] #self.u_tau_bw
            y_norm = self.u_tau_bw/(self.mu_bw/iself.rho_bw)
            max_y_index = self.max_y_index_bw
        else:
            wall_metrics = [self.rho_tw, self.u_tau_bw, self.u_tau_tw, self.u_tau_tw, self.T_tau_tw]
            u_norm = wall_metrics[var_save] #self.u_tau_tw
            y_norm = self.u_tau_tw/(self.mu_tw/self.rho_tw)
            max_y_index = self.max_y_index_tw
        
        if scaling != "wall_units":
            y_norm = 1/self.delta_h

        # Load grid
        grid = np.load(f"{self.save_dir}/grid.npy")

        x_data = grid[0,:,:,:]
        y_data = grid[1,:,:,:]
        z_data = grid[2,:,:,:]

        num_points_x    = x_data[:,0,0].size
        num_points_y    = y_data[0,:,0].size
        num_points_z    = z_data[0,0,:].size
        num_points_xz   = num_points_x*num_points_z

        # Normalize grid
        if scaling == "wall_units":
            y_data_norm = y_data[:,:max_y_index,int(num_points_z/2)]*y_norm
            x_data_norm = x_data[:,:max_y_index,int(num_points_z/2)]*y_norm
            u_data      = output_target[:,:max_y_index]
        else:
            y_data_norm = y_data[:,:,int(num_points_z/2)]*y_norm
            x_data_norm = x_data[:,:,int(num_points_z/2)]*y_norm
            u_data      = output_target

        # Format data
        y_data_norm      = np.asarray( y_data_norm.flatten() )
        x_data_norm      = np.asarray( x_data_norm.flatten() )
        u_data_norm      = np.asarray( u_data.flatten() )/u_norm

        if var_save == 1:
            if c_plus_target == 1:
                u_min = -1.0
                u_max = 1.0
            elif c_plus_target == 5 or c_plus_target == 10:
                u_min = -0.5
                u_max  = 0.5
            elif c_plus_target == 15:
                u_min = -0.25
                u_max = 0.25
            elif c_plus_target == 20:
                u_min = -0.1
                u_max = 0.1
        else:
            u_min = -0.1 #np.min(u_data_norm) #0
            u_max = 0.1 #np.max(u_data_norm) # c_plus_target
        
        #u_min = np.min(u_data_norm) #0
        #u_max = np.max(u_data_norm) # c_plus_target
        # Clip data
        #print(np.min(u_data_norm))
        #print(np.max(u_data_norm))
        u_data_norm[u_data_norm < u_min ] = u_min
        u_data_norm[u_data_norm > u_max ] = u_max

        ### STREAMWISE VELOCITY

        # Clear plot
        plt.clf()
        pi = math.pi

        # Plot data
        #my_cmap = parula_map
        my_norm = colors.Normalize( vmin = u_min, vmax = u_max )
        cs = plt.tricontourf( x_data_norm, y_data_norm, u_data_norm, cmap = "bwr", norm = my_norm, levels = np.arange( u_min, u_max + 1e-6, 1.0e-3 ) )

        # Colorbar
        cbar = plt.colorbar( cs, shrink = 0.14, pad = 0.02, ticks = [-u_max,0,u_max],aspect=5 )
        cbar.ax.tick_params( labelsize = 9 ) 
        plt.clim( u_min, u_max )
        #Postprocess Pseudoboiling line with rolling average
        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
        y_pb_smooth = smooth(y_pb,2)
        y_pb_smooth[0:1] = y_pb[0:1]
        y_pb_smooth[-2:] = y_pb[-2:]
        
        ## Configure plot
        if scaling == "wall_units":
            #plt.xlim( 0.0, 12.0 )
            #plt.xticks( np.arange( 0.0, 12.1, 2.0 ) )
            plt.xlim(0.0, 2500)
            plt.xticks([0.0, 500, 1000, 1500, 2000, 2500],[ r'${0.0}$',  r'${500}$',  r'${1000}$',  r'${1500}$',  r'${2000}$', r'${2500}$'])
            plt.tick_params( axis = 'x', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
            plt.yscale( 'log' )
            plt.ylim( 0.3, 200)
            plt.yticks( np.arange( 1, 10, 100 ) )
            plt.yticks([1, 10, 100], [r'$10^0$', r'$10^1$', r'$10^2$'])  # Fix here
            plt.tick_params( axis = 'y', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
            # Compute log-scale y range
            x_range = 4*pi
            y_range = 0.1  # base-10 log range
            # Set manual aspect to match linear proportions
            ax = plt.gca()
            aspect = x_range / y_range
            ax.set_aspect(aspect, adjustable='box')
            ax.tick_params( axis = 'both', pad = 7.5 )
            if var_save == 0:
                plt.text( 2560, 220, r'${\rho^{\prime}}^{+}$', fontsize = 9 )
            elif var_save == 1:
                plt.text( 2560, 220, r'${u^{\prime}}^{+}$', fontsize = 9 )
            elif var_save == 4:
                plt.text( 2560, 220, r'${T^{\prime}}^{+}$', fontsize = 9 )
            plt.xlabel( r'${x^+}$', size = 9)
            plt.ylabel( r'${y^+}$', size = 9 )
            
            # Plot Pb
            #plt.plot(x_data[:,0,0]*y_norm,(2.0*self.delta_h - y_pb_smooth[1:-1])*y_norm,linestyle = ':', linewidth = 0.5, color = 'darkviolet', zorder = 1)
            
            # Saving figure
            label_save    = c_plus_target
            fig_save_path = f'figures/Cond_phase_speed_XY_plus/{var_sel}/c_{int(c_plus_target)}'
            os.system(f"mkdir -p {fig_save_path}")
            plt.savefig(f'{fig_save_path}/XY_u_c_plus_target_{label_save}_{wall_str}_{n_snapshot}_wider.png', format = 'png', bbox_inches = 'tight', dpi=600 ) 

        else:
            #plt.xlim( 0.0, 12.0 )
            #plt.xticks( np.arange( 0.0, 12.1, 2.0 ) )
            plt.xlim(0.0, 4*pi)
            plt.xticks([0.0, pi, 2*pi, 3*pi, 4*pi],[ r'${0.0}$',  r'${\pi}$',  r'${2 \pi}$',  r'${3 \pi}$',  r'${4 \pi}$'])
            plt.tick_params( axis = 'x', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
            plt.ylim( 0.0, 2)
            plt.yticks( np.arange( 0.0, 2.01, 1.0 ) )
            plt.tick_params( axis = 'y', left = True, right = True, top = True, bottom = True, direction = 'inout', labelsize = 9 )
            plt.gca().set_aspect( 'equal', adjustable = 'box' )
            ax = plt.gca()
            ax.tick_params( axis = 'both', pad = 7.5 )
            if var_save == 0:
                plt.text( 12.9, 2.05, r'${\rho^{\prime}}^{+}$', fontsize = 9 )
            elif var_save == 1:
                plt.text( 12.9, 2.05, r'${u^{\prime}}^{+}$', fontsize = 9 )

            plt.xlabel( r'${x/\delta}$', size = 9)
            plt.ylabel( r'${y/\delta}$', size = 9 )
            
            # Plot Pb
            plt.plot(x_data[:,0,0]*y_norm,(y_pb_smooth[1:-1])*y_norm,linestyle = ':', linewidth = 0.5, color = 'darkviolet', zorder = 1)
            
            # Save fig
            label_save = c_plus_target
            fig_save_path = f'figures/Cond_phase_speed_XY/{var_sel}/c_{int(c_plus_target)}'
            os.system(f"mkdir -p {fig_save_path}")
            plt.savefig(f'{fig_save_path}/XY_u_c_plus_target_{label_save}_{wall_str}_{n_snapshot}.png', format = 'png', bbox_inches = 'tight', dpi=600 ) 
            #plt.savefig(f'figures/Cond_phase_speed_XY/c_{int(c_plus_target)}/XY_u_c_plus_target_{label_save}_{wall_str}_{n_snapshot}.png', format = 'png', bbox_inches = 'tight', dpi=600 )
  

    
if __name__ == "__main__":

    ### DEFINED INPUTS ###
    # Define domain length
    delta     = 100*1e-6           # Fixed channel half height
    L_x       = 4*math.pi*delta
    L_z       = 4/3*math.pi*delta
    save_dir  = f"../data"            # Store DMD outputs
    delta     = 100*1e-6           # Fixed channel half height
    nit       = 2500              # Snapshot each 50000 iterations 
    time_step = 8*1e-10           # Fixed DNS time step 
    dt        = nit*time_step     # Snapshots sample rate
    u_tau_bw  = 0.19              # Bottom Wall friction velocity
    mu_bw     = 1.5312*1E-05    
    rho_bw    = 148.04
    y_plus_TKE_bw = np.array([1,10])
    u_tau_tw  = 0.19               # Top Wall friction velocity
    mu_tw     = 1.5312*1E-05    
    rho_tw    = 148.04
    y_plus_TKE_tw = np.array([1,10,25,50,100])


    # Snapshots (on file *.h5)
    cases = []
    for file in glob.glob("../../data_resolvent/*.h5"):
        cases.append(file[21:-3])
    cases.sort()
    #cases = cases[0:40]
    #print(cases)
    #cases = ["3d_high_pressure_turbulent_channel_flow_28500000", 
    #        "3d_high_pressure_turbulent_channel_flow_30000000"]
    #print(cases)
    
    ### DEFINE FFT CLASS ###
    ## FFT Proces
    # Select field 
    var_names  =  ["rho","u","v","w","T"]
    var_sel    = "u"
    var_save   = var_names.index(var_sel)
    FFT_class = FFT_process(save_dir,cases,dt, delta, L_x, L_z, 
            u_tau_bw, mu_bw, rho_bw, y_plus_TKE_bw,
            u_tau_tw, mu_tw, rho_tw, y_plus_TKE_tw,
            var_save)

    # Obtain desired omega fields
    #y_plus_target = np.array([1, 5, 10, 15, 20])
    c_plus_target = np.array([1,5,10,15,20]) #np.array([1, 5, 10, 15, 20])
    #n_snapshots   = np.linspace(0,len(cases),int(len(cases)+1)) #np.array([0,2,4])
    scaling = "outer_units" #"wall_units" # or outer_units
    wall    = "tw" # or tw
    print("Conditioning phase speed data...")
    for idx, n_snap in tqdm(enumerate(cases), total=len(cases)):
        for cc in range(0,len(c_plus_target)):
            
            if idx == 0: #idx % 2 == 0:
                output_target = FFT_class.obtain_spectra_target(-c_plus_target[cc],y_plus_TKE_tw[cc],wall)
                FFT_class.plot_c_target_XY(output_target[:,:,int(idx)], c_plus_target[cc],wall, int(idx), scaling, FFT_class.pseudoboiling[int(idx),:],var_save, var_sel)
                # Physical sanity check as snapshot
                #FFT_class.plot_c_target_XY(FFT_class.snapshots[:,:,int(idx)], c_plus_target[cc],"tw", int(idx)) #snapshots[var,:,z_plane,:,:]
