import os
import numpy as np
from PIL import Image
from scipy import linalg
from matplotlib import pyplot as plt

def create_coordinate_array(h,w):
    coords_array = np.zeros((h*w,4))

    # Create coordinates {x,y | x = (-w//2, w//2 - 1); y = (h//2,-h//2-1)}
    x_values = np.arange(-w//2,(w//2),1)
    y_values = np.arange(h//2,(-h//2),-1)

    #print(x_values.size)
    #print(y_values.size)

    [X,Y] = np.meshgrid(x_values,y_values)
    x_coords = np.reshape(X,(X.size,1))
    y_coords = np.reshape(Y,(Y.size,1))

    #print(coords_array)

    # Map indices into the coords_array
    rows = np.arange(0,h,1,dtype=int)
    cols = np.arange(0,w,1,dtype=int)
    [C, R] = np.meshgrid(cols,rows)
    rowi = np.reshape(R,(R.size,1))
    coli = np.reshape(C,(C.size,1))

    coords_array[:,0:1] = x_coords
    coords_array[:,1:2] = y_coords
    coords_array[:,2:3] = coli
    coords_array[:,3:4] = rowi

    #print(coords_array)
    return coords_array

def generate_GT(points_raw):
    # Transform selected points into our cartesian grid
    points_xi = np.zeros((4,3))
    for ii in range(len(points_raw)):
        points_xi[ii][0] = -w//2 + points_raw[ii][0]
        points_xi[ii][1] = h//2 - points_raw[ii][1]
        points_xi[ii][2] = 1
    #print(points_xi)
    
    jj = 0
    aa = [0,0,1,0,1,3,0,3]
    zz = np.zeros((4,3))
    for ii in range(len(points_xi)):
        zz[ii][0] = points_xi[aa[jj],0]
        zz[ii][1] = points_xi[aa[jj+1],1]
        zz[ii][2] = 1
        jj = jj + 2
    
    #print(points_xi.shape)
    #print(zz.shape)
    return points_xi, zz


def normalize_points(zz):
    '''center and scale points'''
    uu = zz.T
    ff_xx = np.ones(uu.shape)
    indices, = np.where(abs(uu[2,:])>10**-12)
    ff_xx[0:2,indices] = uu[0:2,indices]/uu[2,indices]
    ff_xx[2,indices] = 1
    mu = np.mean(ff_xx[0:2,:],axis=1)
    mu_r = np.zeros((mu.shape[0],ff_xx.shape[1]))
    for ii in range(ff_xx.shape[1]):
        mu_r[:,ii] = mu
    mu_dist = np.mean((np.sum((ff_xx[0:2]-mu_r)**2,axis=0))**0.5)
    scale = (2**0.5/mu_dist)
    s0 = -scale*mu[0]
    s1 = -scale*mu[1]
    S = np.array([[scale,0,s0],[0,scale,s1],[0,0,1]])
    normalized_zz = S @ ff_xx
    return normalized_zz, S

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
    print("Width: " + str(w) + " " + " Height: " + str(h))
    
    plt.figure(1)
    plt.imshow(image)
    points_raw = plt.ginput(4)
    plt.close(1)
    #print(points_raw)

    #points_raw = np.array([(1244.6298701298704, 1090.4090909090905), (2705.6688311688313, 635.863636363636), (2884.2402597402606, 2291.707792207792), (1277.0974025974028, 2113.136363636363)])
    # Transfer the image array into a cartesian coordinate plane?
    # rowi -> row index; coli -> col index in the original image
    
    coords_array = create_coordinate_array(h,w) # coords_array = [xi, yi, coli, rowi]
    points_xi, zz = generate_GT(points_raw) # Generate ground truth rectangle

    #print(coords_array)


    points_xi_normalized, normalizing_matrix = normalize_points(points_xi)
    zz_norm, normalizing_matrix_zz = normalize_points(zz)

    # Compute Homography
    # Compute for matrix A
    A = np.zeros((2*(zz_norm.shape[0]+1),9))
    j = 0
    for i in range(zz_norm.shape[0]+1):
        a = np.zeros((1,3))[0]
        b = (-zz_norm[2,i] * points_xi_normalized[:,i])
        c = (zz_norm[1,i] * points_xi_normalized[:,i])
        d = ( zz_norm[2,i]  * points_xi_normalized[:,i])
        e = ( -zz_norm[0,i]  * points_xi_normalized[:,i])
        row1 = np.concatenate((a,b,c),axis = None)
        row2 = np.concatenate((d,a,e),axis = None)

        # Stack Ai's
        A[j,:] = row1
        A[j+1,:] = row2
        j = j + 2

    null_space_of_A = -linalg.null_space(A)
    hh = np.reshape(null_space_of_A, (3,3))
    denormalized_H = np.dot(np.linalg.inv(normalizing_matrix_zz),np.dot(hh,normalizing_matrix))
    
    #print(denormalized_H)

    # Test H
    # point_raw = 738.13636364  888.48701299
    # points_after = 738.13636364  401.47402597




    # Apply H to the four bounds of the image, get new bounds
    img_bounds = np.array([[-w//2, (w//2)-1, (w//2)-1, -w//2],[h//2,h//2,(-h//2)+1,(-h//2)+1],[1,1,1,1]])
    new_image_bounds = np.zeros(img_bounds.shape)
    for ii in range(4):
        new_image_bounds[:,ii] = denormalized_H @ img_bounds[:,ii]
    xmin = np.amin(new_image_bounds[0])
    ymax = np.amax(new_image_bounds[1])
    #print(xmin)
    #print(ymax)
    new_col = abs((2 * np.amax(abs(new_image_bounds[0])))).astype(int)
    new_row = abs((2 * np.amax(abs(new_image_bounds[1])))).astype(int)
    #print(new_col)
    #print(new_row)
    #print(new_image_bounds)
    new_image = np.zeros((new_row,new_col,3))
    
    # Apply H to all points in coords_array?
    point_container = np.zeros((3,1))
    #new_coords_array = np.zeros(coords_array.shape)
    for ii in range(len(coords_array)):
        point_container[0] = coords_array[ii,0]
        point_container[1] = coords_array[ii,1]
        point_container[2] = 1
        new_point = (denormalized_H @ point_container).T[0,0:2]

        new_coli = ((new_point[0] - xmin)-1).astype(int)
        new_rowi = ((ymax + new_point[1])-1).astype(int)

        new_image[new_rowi,new_coli] = im[coords_array[ii,3].astype(int),coords_array[ii,2].astype(int)]


        #new_coords_array[ii,0:2] = new_point[0,0:2]
    #print(new_coords_array)
    #print(np.max(new_coords_array))
    new_final_image = np.fliplr(new_image)
    plt.imshow((new_final_image).astype(np.uint8))
    plt.show()
        
    ######### YAW KO NAAAAAA ############
    



    

    





    

