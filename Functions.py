from copy import deepcopy
import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import eigsh


def get_lower_triangular_indices(n_sites):
    
    assert(n_sites>2), "Wrong number of sites, must be > 2" 
    L = [[2,1]]
    
    for N in range(3,n_sites+1):
        
        C = deepcopy(L)
        
        for x in C:
            L.append([x[0]+2**(N-1),x[1]+2**(N-1)])
            
        for j in range(2**(N-2)):
            L.append([2**(N-1)+j,2**(N-2)+j])
         
    return L       


def convert_to_symmetric_indices(lower_triangular_indices):
    
    L = lower_triangular_indices
    C = deepcopy(L)
    
    for x in C:
        L.append([x[1],x[0]])
        
    return L    
    
    
    
def get_diagonal_sign(n_sites):
    
    S = np.zeros((2**n_sites))
    
    for i in range(2**(n_sites)):
        suma = 0
        for j in range(n_sites):
            suma += (-1)**np.floor(i/(2**j))

        S[i] = suma
        
    return S

    
def get_rows_columns(n_sites):
    
    Indices_list = convert_to_symmetric_indices(get_lower_triangular_indices(n_sites))
    
    rows = [x[0] for x in Indices_list]
    columns = [x[1] for x in Indices_list]
    
    return np.array(rows),np.array(columns)


def get_sparse_pure_hamiltonian(n_sites,J,h):
    
    sh = 2**n_sites 
    
    row_xy, column_xy = get_rows_columns(n_sites)
    data_xy = np.full((1,len(row_xy)),2.0*J).flatten()
    H_xy = csc_array((data_xy, (row_xy,column_xy)), shape=(sh, sh))

    data_z = -h*get_diagonal_sign(n_sites)
    row_z = np.arange(sh)
    H_z = csc_array((data_z, (row_z,row_z)), shape=(sh, sh))

    return H_xy + H_z  



def get_ground_state_energy(H):
    return eigsh(H,k=1,which='SA',return_eigenvectors=False)



def get_impure_diagonal(h,δh,distance,n_sites):
    
    if n_sites%2==0:
        assert distance%2==0, 'Configuration is not symmetrical'
        imps_locations = [int((n_sites-distance-1)/2),int((n_sites+distance)/2)]
    else: 
        assert distance%2==1, 'Configuration is not symmetrical' 
        imps_locations = [int((n_sites-distance-2)/2),int((n_sites+distance)/2)]
    
    S = np.zeros((2**n_sites))
    
    for i in range(2**(n_sites)):
        suma = 0
        for j in range(n_sites):
            if j in imps_locations:
                suma += (h+δh)*(-1)**np.floor(i/(2**j))
            else: 
                suma += h*(-1)**np.floor(i/(2**j))

        S[i] = suma
        
    return S



def get_sparse_impure_hamiltonian(n_sites,J,h,δh,distance):
    
    sh = 2**n_sites
    
    row_xy, column_xy = get_rows_columns(n_sites)
    data_xy = np.full((1,len(row_xy)),2.0*J).flatten()
    H_xy = csc_array((data_xy, (row_xy,column_xy)), shape=(sh, sh))

    data_z = -get_impure_diagonal(h,δh,distance,n_sites)
    row_z = np.arange(sh)
    H_z = csc_array((data_z, (row_z,row_z)), shape=(sh, sh))

    return H_xy + H_z  