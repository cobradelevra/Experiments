# ps2
import os
import numpy as np
import cv2


# +
# FUNCTION`S CREATED FROM   'INTRO'
def middle_section(arr):
    """
    for plt.plot. Can be colored or not.
    """
    mid_row=int((arr.shape)[0]/2)
    if len(arr.shape) == 3 :
        return (arr[mid_row,:,0:1]).flatten()
    else:
        return((arr[mid_row,:]).flatten())
def convolve_2d(arr,flt):
    """
    filter designed to be 3x3
    """
    conv=np.empty(((arr.shape)[0]-2,(arr.shape)[1]-2))
    for i in (range(arr.shape[0]))[1:-1]:
        in1=np.convolve(arr[i-0,:],flt[0,:],'valid')
        in2=np.convolve(arr[i,:],flt[1,:],'valid')
        in3=np.convolve(arr[i+1,:],flt[2,:],'valid')
        conv[i-1,:]=(in1+in2+in3)
    return conv

def im_asrankn(arr,n):
    """
    pull the first n rank-1 part of the array. with SVD.
    designed for NON-colored image
    """
    U,E,V=np.linalg.svd(arr)
    ans=np.zeros(arr.shape)
    for i in range(n):
        ans+=(U[:,i:i+1]@V[i:i+1,:])*E[i]
    return ans
def gaussian_filter2d(piv,mean=0,sigma=1,normal=False):
    loc=np.square((np.arange(-(piv-1)/2,((piv-1)/2)+1)).reshape((piv,1)))
    temp=np.concatenate(piv*[loc],axis=1)
    x=temp+temp.T
    z=(1/(2*np.pi*sigma**2))*np.exp((-1*np.square(x-mean))/(2*np.square(sigma)))
    if normal:
        return (z-z.mean())
    return z



# -

import scipy.fft as fft

import matplotlib.pyplot as plt
import scipy as sci

## 1-a
# Read images
L = cv2.imread(os.path.join( 'pair1-L.png'), 0) 
R = cv2.imread(os.path.join( 'pair1-R.png'), 0) 
ans=cv2.imread(os.path.join("pair1-D_R.png"),0)

plt.imshow(ans)

plt.imshow(R)

plt.imshow(L)

L_normalized = cv2.normalize(L, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
R_normalized = cv2.normalize(R, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

L_smooth=cv2.GaussianBlur(L,(5,5),10,10)
R_smooth=cv2.GaussianBlur(R,(5,5),10,10)

L.std(),L.mean()


# +
def desparity_cv(L,R,window):
    L=(L-L.mean())/L.std()
    R=(R-R.mean())/R.std()
    expand=(window-1)//2
    R_exp=cv2.copyMakeBorder(R,expand,expand,expand,expand,cv2.BORDER_REFLECT)
    L_exp=cv2.copyMakeBorder(L,expand,expand,expand,expand,cv2.BORDER_REFLECT)
    row_size=L.shape[0]
    col_size=L.shape[1]
    desparity_map=np.zeros(L.shape)
    for i in range(row_size):
        temp_L=L_exp[i+expand:i+expand+window,:]
        temp_R=R_exp[i+expand:i+expand+window,:]
        for j in range(col_size):
            flt=temp_L[:,j:j+window]
            x=cv2.filter2D(temp_R,-1,flt)[expand:expand+1,expand:-expand]
            x_hat=np.argmax(x)
            desparity_map[i,j]=(x_hat-j)
    return desparity_map
        
        
    
# -

# %%time
deneme3=desparity_cv(L,R,7)

plt.imshow(deneme3)



# +
def desparity_cv(L,R,window):
    L=(L-L.mean())/L.std()
    R=(R-R.mean())/R.std()
    expand=(window-1)//2
    R_exp=cv2.copyMakeBorder(R,expand,expand,expand,expand,cv2.BORDER_REFLECT)
    L_exp=cv2.copyMakeBorder(L,expand,expand,expand,expand,cv2.BORDER_REFLECT)
    row_size=L.shape[0]
    col_size=L.shape[1]
    desparity_map=np.zeros(L.shape)
    for i in range(row_size):
        temp_L=L_exp[i+expand:i+expand+window,:]
        temp_R=R_exp[i+expand:i+expand+window,:]
        def conv__(tem_L,temp_R,desparity_map,i=i,j=j):
            if np.any(desparity_map[i,:]!=0):
                sec1=temp_L[:,j:j+window]
                sec2=temp_L[:,j-1:j-1+window]
                path1=sec1*temp_R[:,j:j+window]
                path2=sec1*temp_R[:,j+1:j+1+window]
                path3=sec2*temp_R[:,j:j+window]
                mx=np.argmax((path1,path2,path3))
                if mx==0:
                    desparity_map[]
                    
                
        for j in range(col_size):
            
            flt=temp_L[:,j:j+window]
            x=cv2.filter2D(temp_R,-1,flt)[expand:expand+1,expand:-expand]
            x_hat=np.argmax(x)
            desparity_map[i,j]=(x_hat-j)
    return desparity_map
        
        
    
# -



# Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
from disparity_ssd import disparity_ssd
D_L = disparity_ssd(L, R)
D_R = disparity_ssd(R, L)

# TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
# Note: They may need to be scaled/shifted before saving to show results properly

# TODO: Rest of your code here
