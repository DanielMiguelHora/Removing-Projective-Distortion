
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import linalg
from scipy.interpolate import griddata

def make_ground_truth_points(xi):
    points_xi = np.array(xi)
    zzd = np.zeros((5,3))
    for ii in range(len(points_xi)):
        zzd[ii][0] = points_xi[ii][0]
        zzd[ii][1] = points_xi[ii][1]
        zzd[ii][2] = 1
    
    jj = 0
    aa = [0,0,1,0,1,3,0,3]
    zz = np.zeros((5,3))
    for ii in range(len(zzd)-1):
        zz[ii,0] = zzd[aa[jj],0]
        zz[ii,1] = zzd[aa[jj+1],1]
        zz[ii,2] = 1
        jj = jj+2
    zz[4,:] = zz[0,:]
    return zz[0:4,:],zzd[0:4,:]



def get_new_image(nrows,ncols,imm,bounds,transf_prec,nsamples):
    '''Apply transformation H into the image to get the corrected image'''
    # No clue how this works. Definitely need to brush up python image processing
    # Copied directly
    xx = np.linspace(1,ncols, ncols)
    yy = np.linspace(1,nrows,nrows)
    [xi,yi] = np.meshgrid(xx,yy)
    a0 = np.reshape(xi,-1,order = "F")+bounds[2]
    a1 = np.reshape(yi,-1,order = "F")+bounds[0]
    a2 = np.ones((ncols*nrows))
    uv = np.vstack((a0.T, a1.T, a2.T))
    new_transf =np.dot(transf_prec,uv)
    val_normalization =  np.vstack((new_transf[2,:],new_transf[2,:],new_transf[2,:]))

    newT = new_transf/val_normalization

    ## Review
    xi = np.reshape(newT[0,:],(nrows,ncols),order = "F")
    yi = np.reshape(newT[1,:],(nrows,ncols),order = "F")
    cols = imm.shape[1]
    rows = imm.shape[0]
    xxq = np.linspace(1,rows,rows).astype(np.int64)
    yyq = np.linspace(1,cols,cols).astype(np.int64)
    [x,y] = np.meshgrid(yyq,xxq)
    x = (x-1).astype(np.int64)
    y = (y-1).astype(np.int64)

    ix = np.random.randint(im.shape[1],size = nsamples)
    iy = np.random.randint(im.shape[0],size = nsamples)
    samples = im[iy,ix]
    int_im = griddata((iy,ix),samples,(yi,xi))

    #plotting
    fig = plt.figure(figsize = (8,8))
    columns = 2
    rows = 1
    fig.add_subplot(rows,columns,1)
    plt.imshow(im)

    fig.add_subplot(rows,columns,2)
    plt.imshow(int_im.astype(np.uint8))
    plt.show()
if __name__ == "__main__":
    # Load image
    cwd = os.getcwd()
    ddir = os.listdir(cwd)
    if not "images" in ddir:
        print("images folder not found")
    else:
        image_to_load = os.path.join(cwd,"images","book.jpg")
        image = Image.open(image_to_load)
        im = np.array(image)
    

    # Image shape
    h, w, chan = im.shape


    # Select 4 points
    plt.imshow(image)
    print("Select 4 points to crop")
    points_raw = plt.ginput(4)
    
    zz, zzd = make_ground_truth_points(points_raw)

    #print( img_c.shape ) # h, w, channel
    # Compute A
    A = np.zeros((8,9))
    jj = 0
    for ii in range(len(zzd)):
        a = np.zeros((1,3))[0]
        b = -zzd[ii]
        c = zz[ii][1] * zzd[ii]
        d = zzd[ii]
        e = -zz[ii][0] * zzd[ii]
        row1 = np.concatenate((a,b,c),axis=None)
        row2 = np.concatenate((d,a,e),axis=None)
        A[jj] = row1
        A[jj+1] = row2
        jj = jj + 2
    #print(A)

    # Solve for the null_space of A
    null_space_A = -linalg.null_space(A)
    H = np.reshape(null_space_A,(3,3))
    #print(H)


    

