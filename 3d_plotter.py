import sys
import cv2
from signal import signal, SIGINT
import numpy as np
import time
import os, os.path
from glob import iglob
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from numpy import genfromtxt

saveDir = './output/'
frameWidth = 672
frameHeight = 376
pngdim = 256
density = 3
margin = 10
f_cam = 100
b_cam = 0.12

def main():
    rootdir_glob = '/home/aritra/Downloads/SVC/**/*' # Note the added asterisks
    # This will return absolute paths
    dir_list = [d for d in iglob(rootdir_glob, recursive=True) if os.path.isdir(d)]
    for d in dir_list:
        #print('The directory is: '+d) # Replace with desired operations
        justName = os.path.basename(d)
        print('Processing samples of:'+justName)
        if not os.path.exists(saveDir+justName):
            os.makedirs(saveDir+justName)

        rootsubdir = d+'/**/*'
        file_list = [f for f in iglob(rootsubdir, recursive=True) if os.path.isfile(f)]
        for f in file_list:
            print('current sample is: '+f)
            my_data = genfromtxt(f, delimiter=' ')
            justFile = Path(f).stem
            print('just file '+ justFile)
            rows,cols = my_data.shape
            p = np.zeros((3,rows))
            i=0
            j=0
            while(i<rows):
                if(my_data[i][0] > 0):
                    p[0][j] = my_data[i][0]
                    p[1][j] = my_data[i][1]
                    p[2][j] = 0
                    j+=1
                    i+=1
                else:
                    i+=1
            #
            print ('rows: '+str(rows)+ ' i: '+str(i)+' j: '+str(j))
            if(j<i):
                for k in range (j,i):
                    p[0][k] = p[0][j-1]
                    p[1][k] = p[1][j-1]
                    p[2][k] = p[2][j-1]

            p = p.T
            #p2 = np.unique(p, axis=0)
            _, idx = np.unique(p, return_index=True, axis=0)
            p2 = p[np.sort(idx)]
            #p2 = p
            #print(p2.shape)
            '''
            # Linear length along the line:
            distance = np.cumsum( np.sqrt(np.sum( np.diff(p2, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]

            # Interpolation for different methods:
            #interpolations_methods = ['slinear', 'quadratic', 'cubic']
            interpolations_methods = ['quadratic']
            alpha = np.linspace(0, 1, density*pngdim)

            interpolated_points = {}
            for method in interpolations_methods:
                interpolator =  interp1d(distance, p2, kind=method, axis=0)
                interpolated_points[method] = interpolator(alpha)

            #print(interpolated_points['quadratic'].shape)
            #PNG WRITING CODE SEGMENT START
            minx=1000
            miny=1000
            maxx=0
            maxy=0
            for i in range(0,density*pngdim):
                if(minx>interpolated_points['quadratic'][i][0]):
                    minx = interpolated_points['quadratic'][i][0]
                if(miny>interpolated_points['quadratic'][i][1]):
                    miny = interpolated_points['quadratic'][i][1]
                if(maxx<interpolated_points['quadratic'][i][0]):
                    maxx = interpolated_points['quadratic'][i][0]
                if(maxy<interpolated_points['quadratic'][i][1]):
                    maxy = interpolated_points['quadratic'][i][1]
            blank_image = np.zeros((pngdim,pngdim,3), np.uint8)
            redFactor = 0
            framedim = (pngdim - margin*2)
            if((maxx-minx)>(maxy-miny)):
                redFactor = float(framedim)/(float(maxx-minx))
                yadjust = int((framedim - (redFactor*(maxy-miny)))/2)
                for k in range(0,density*pngdim):
                    px = int((round(interpolated_points['quadratic'][k][0])-minx)*redFactor) + margin
                    py = yadjust + int((round(interpolated_points['quadratic'][k][1])-miny)*redFactor) + margin
                    cv2.circle(blank_image,(px,(pngdim-py)),2,(255,255,255),-1)
            else:
                redFactor = float(framedim)/(float(maxy-miny))
                xadjust = int((framedim - (redFactor*(maxx-minx)))/2)
                for k in range(0,density*pngdim):
                    px = xadjust + int((round(interpolated_points['quadratic'][k][0])-minx)*redFactor) + margin
                    py = int((round(interpolated_points['quadratic'][k][1])-miny)*redFactor) + margin
                    cv2.circle(blank_image,(px,(pngdim-py)),2,(255,255,255),-1)
            
            cv2.imshow('signature 2d',blank_image)
            print('saving... '+saveDir+justName+'/'+justFile+'.png')
            cv2.imwrite(saveDir+justName+'/'+justFile+'.png',blank_image)
            cv2.waitKey(10)

            #END OF PNG WRITING

            '''
            # Graph:
            fig2 = plt.figure(2)
            ax3d = fig2.add_subplot(111, projection='3d')
            l,_ = p2.shape
            x_ = np.zeros((l))
            y_ = np.zeros((l))
            z_ = np.zeros((l))
            for i in range (0,l):
                x_[i] = p2[i][0]
                y_[i] = p2[i][1]
                z_[i] = p2[i][2]

            tck, u = interpolate.splprep([x_,y_,z_], s=0, k=4)
            #print(tck)
            
            x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
            u_fine = np.linspace(0,1,3*l)
            #print(u_fine)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            #print(x_fine)
            ax3d.plot(x_fine, y_fine, z_fine, 'g')
            #ax3d.plot(x_knots, y_knots, z_knots, 'go')

            #ax3d.plot(x_, y_, z_, 'r*')

            fig2.show()
            plt.show()
            #print('saving... '+saveDir+justName+'/'+justFile+'.csv')
            #np.savetxt(saveDir+justName+'/'+justFile+'.csv', np.stack((x_fine,y_fine,z_fine), axis=1), delimiter=",")
            
            #plt.waitforbuttonpress(0)
            #plt.close()
            
            print("\nFinished processing "+ f)
            





if __name__ == "__main__":
    main()