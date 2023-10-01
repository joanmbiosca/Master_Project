from copy import deepcopy
import numpy as np 
import os.path
import sys
import time
import os
from pathlib import Path
import pickle

from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc


from scipy.signal import butter, lfilter, freqz
from scipy import signal
from scipy.stats import norm
from scipy import ndimage

from shutil import copyfile
from datetime import datetime

import itertools as it

#-----Functions-----

def grad_roll(u,mask):
    gradux = np.zeros_like(u)
    graduy = np.zeros_like(u)
    graduz = np.zeros_like(u)
    gradux[mask] = (0.5/h)*(-np.roll(u,1,0)[mask]+np.roll(u,-1,0)[mask])
    graduy[mask] = (0.5/h)*(-np.roll(u,1,1)[mask]+np.roll(u,-1,1)[mask])
    graduz[mask] = (0.5/h)*(-np.roll(u,1,2)[mask]+np.roll(u,-1,2)[mask])
    gu = [gradux,graduy,graduz]
    return gu

def calculateshearxy(gradvx,gradvy,maskT):

    result = np.zeros_like(gradvx[0])
    result[maskT] = (viscosityliq/2) * (gradvx[1][maskT] + gradvy[0][maskT])
    return result

def calculateshearyz(gradvy,gradvz,maskT):

    result = np.zeros_like(gradvz[0])
    result[maskT] = (viscosityliq/2) * (gradvy[2][maskT] + gradvz[1][maskT])
    return result

def calculateshearxz(gradvx,gradvz,maskT):

    result = np.zeros_like(gradvx[0])
    result[maskT] = (viscosityliq/2) * (gradvx[2][maskT] + gradvz[0][maskT])
    return result

def plot_shear_plane(phi,v_x,v_y,v_z): #Plot shear over time
    totalsum = 0.0
    totalsumxy= 0.0
    totalsumyz= 0.0
    totalsumxz= 0.0
    for z in range(Nz):
        for y in range(Ny):
            for x in range(Nx):
                if (np.absolute(phi[x,y,z]) < 0.8):
                    totalsum = totalsum + shear_total[x,y,z]
                    totalsumxy = totalsumxy + shear_xy[x,y,z]*shear_xy[x,y,z]
                    totalsumyz = totalsumyz + shear_yz[x,y,z]*shear_yz[x,y,z]
                    totalsumxz = totalsumxz + shear_xz[x,y,z]*shear_xz[x,y,z]

    shearplot.append(totalsum)
    shearplotxy.append(totalsumxy)
    shearplotyz.append(totalsumyz)
    shearplotxz.append(totalsumxz)

    #file5=os.path.join('./'+simulation+'/shearxy/shear_xy_t='+str(t)+'.txt')
    #file6=os.path.join('./'+simulation+'/shearxz/shear_xz_t='+str(t)+'.txt')
    #file7=os.path.join('./'+simulation+'/shearyz/shear_yz_t='+str(t)+'.txt')
    file8=os.path.join('./'+simulation+'/shear.txt')

    with open(file8,"w+") as f:
        f.write("t totalshear[t] shearxy[t] shearyz[t] shearxz[t]\n")
        for t in range(0,len(shearplot)):  
            f.write("{0} {1} {2} {3} {4}\n".format(t*jump,shearplot[t],shearplotxy[t],shearplotyz[t],shearplotxz[t]))
        f.close()
    
    #reduced volume plot
    plt.subplot(2,2,1)
    plt.plot(shearplot[2:])
    plt.xlabel("$\Delta t$",fontsize=12)
    plt.ylabel(r"${Shear\, total \, (t)}$",fontsize=13)

    #volume plot
    plt.subplot(2,2,2)
    plt.plot(shearplotxy[2:])
    plt.xlabel("$\Delta t$",fontsize=12)
    plt.ylabel(r"${Shear\, xy\, (t)}$",fontsize=13)

    #Energy plot   
    plt.subplot(2,2,3)
    plt.plot(shearplotyz[2:])
    plt.xlabel("$\Delta t$",fontsize=12)
    plt.ylabel(r"${Shear\, yz\, (t)}$",fontsize=13)

    plt.subplot(2,2,4)
    plt.plot(shearplotxz[2:])
    plt.xlabel("$\Delta t$",fontsize=12)
    plt.ylabel(r"${Shear\, xz\, (t)}$",fontsize=13)

    figure = plt.gcf()
    figure.set_size_inches(4.5, 4.5,forward=True)
    plt.tight_layout()
    file=os.path.join("./"+simulation+"/shearovertime.png")
    figure.savefig(file, dpi = 200)
    plt.close('all')

def save_vti_file(phi,vx,vy,vz, nx, ny, nz,name): #Save vti fields in different lists for easier visualisation
    
    pc_lista_novo_phi = []
    pc_lista_novo_vx = []
    pc_lista_novo_vy = []
    pc_lista_novo_vz = []
    pc_lista_novo_shear_xy = []
    pc_lista_novo_shear_xz = []
    pc_lista_novo_shear_yz = []
    pc_lista_novo_shear_total = []
    
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                t_phi = phi[x,y,z]
                t_vx = vx[x,y,z]
                t_vy = vy[x,y,z]
                t_vz = vz[x,y,z]
                t_shear_xy = shear_xy[x,y,z]
                t_shear_xz = shear_xz[x,y,z]
                t_shear_yz = shear_yz[x,y,z]
                t_shear_total = shear_total[x,y,z]
                pc_lista_novo_phi.append(t_phi)
                pc_lista_novo_vx.append(t_vx)
                pc_lista_novo_vy.append(t_vy)
                pc_lista_novo_vz.append(t_vz)
                pc_lista_novo_shear_xy.append(t_shear_xy)
                pc_lista_novo_shear_xz.append(t_shear_xz)
                pc_lista_novo_shear_yz.append(t_shear_yz)
                pc_lista_novo_shear_total.append(t_shear_total)

    pc_string_novo_phi = "    ".join([str(_) for _ in pc_lista_novo_phi]) 
    pc_string_novo_vx = "    ".join([str(_) for _ in pc_lista_novo_vx])
    pc_string_novo_vy = "    ".join([str(_) for _ in pc_lista_novo_vy])
    pc_string_novo_vz = "    ".join([str(_) for _ in pc_lista_novo_vz])
    pc_string_novo_shear_xy = "    ".join([str(_) for _ in pc_lista_novo_shear_xy])
    pc_string_novo_shear_xz = "    ".join([str(_) for _ in pc_lista_novo_shear_xz])
    pc_string_novo_shear_yz = "    ".join([str(_) for _ in pc_lista_novo_shear_yz])
    pc_string_novo_shear_total = "    ".join([str(_) for _ in pc_lista_novo_shear_total])
    file=os.path.join(output1+name)
    with open(file, "w" ) as my_file:
        my_file.write('<?xml version="1.0"?>')
        my_file.write('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n')
        my_file.write('  <ImageData WholeExtent="0 '+str(nx)+' 0 '+str(ny)+' 0 '+str(nz)+'" Origin="0 0 0" Spacing ="1 1 1">\n')
        my_file.write('    <Piece Extent="0 '+str(nx)+' 0 '+str(ny)+' 0 '+str(nz)+'">\n') 
        my_file.write('     <CellData>\n')
        my_file.write('     <DataArray Name="phi" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_phi)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="shear_xy" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_shear_xy)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="shear_xz" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_shear_xz)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="shear_yz" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_shear_yz)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="shear_total" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_shear_total)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="v_x" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_vx)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="v_y" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_vy)
        my_file.write('\n         </DataArray>\n')
        my_file.write('     <DataArray Name="v_z" type="Float64" format="ascii">\n')
        my_file.write('     ')
        my_file.write(pc_string_novo_vz)
        my_file.write('\n         </DataArray>\n')
        my_file.write('      </CellData>\n')
        my_file.write('    </Piece>\n')
        my_file.write('</ImageData>\n')
        my_file.write('</VTKFile>\n')
        my_file.close() 

#-----Main-----

#Variable inisialisation

h = 1.0
viscosityliq = 1.0
jump = 2000
Nx=55       # lattice size X
Ny=55          # lattice size Y
Nz=55        # lattice size Z 
R= ((Ny-1)/2) 
DataFiles = 1000
booll2 = np.zeros((Nx,Ny,Nz))
shear_total = np.zeros((Nx,Ny,Nz),float)
v_x = np.zeros((Nx,Ny,Nz),float)
v_y = np.zeros((Nx,Ny,Nz),float)
v_z = np.zeros((Nx,Ny,Nz),float)
phi = np.zeros((Nx,Ny,Nz),float)
shear_xy = np.zeros((Nx,Ny,Nz),float)
shear_yz = np.zeros((Nx,Ny,Nz),float)
shear_xz = np.zeros((Nx,Ny,Nz),float)
booll2 = np.zeros((Nx,Ny,Nz))
booll = np.zeros((Nx,Ny))
newbooll2 = np.zeros((Nx,Ny,Nz))
newbooll = np.zeros((Nx,Ny))
xi = []
omega = []
xi_y = []
omega_y = []
xi_x = []
omega_x = []
xi_z = []
omega_z = []
vol=[]
areat=[]
areat2=[]
volred1 = []
volred2 = []
shearplot = []
shearplotxy = []
shearplotyz = []
shearplotxz = []

#Create folders
simulation=str("name")
output1=os.path.join('./'+simulation+'/vti_shear/')
if not os.path.exists(output1):
    os.makedirs(output1)
    print("new folder vti shear")
print(simulation)

#Creating boolean matrix booll. It indicates the whole cylindrical system

for x in reversed (range (Nx)):
    for y in range (0,Ny):
        for k in range (0,Nz):
            if((x-R)**2 + (y-R)**2 <= R*R):
                booll2 [x,y,k] = 1
for x in reversed (range (Nx)):
    for y in range (0,Ny):
        if((x-R)**2 + (y-R)**2 <= R*R):
            booll [x,y] = 1

#Creating boolean matrix newbooll. It indicates the points where you calculate evolutions. Newbooll2 - booll2 = boundary points of the 3D cylinder

newbooll2 = booll2.copy()
newbooll2[int(Nx-1),int((Nx-1)/2),:] = 0; newbooll2[int((Nx-1)/2),int(Nx-1),:] = 0; newbooll2[0,int((Nx-1)/2),:] = 0; newbooll2[int((Nx-1)/2),0,:] = 0

#Creating boolean matrix newbooll. It indicates the points where you calculate evolutions. Newbooll - booll = boundary points of the 2D cylinder

newbooll = booll.copy()
newbooll[int(Nx-1),int((Nx-1)/2)] = 0; newbooll[int((Nx-1)/2),int(Nx-1)] = 0; newbooll[0,int((Nx-1)/2)] = 0; newbooll[int((Nx-1)/2),0] = 0
 
for i in range(1,Nx-1):
    for j in range(1, Ny-1):
        if(booll[i+1][j] == 0 or booll[i-1][j] == 0 or booll[i][j+1] == 0 or booll[i][j-1] == 0):
            newbooll[i][j] = 0
            newbooll2[i,j,:] = 0
maskT=newbooll2>0.5

for t in range(0,DataFiles): #Temporal loop

    file1=os.path.join('./'+simulation+'/velocity_x/velocity_x_t='+str(t*jump)+'.txt') 
    file2=os.path.join('./'+simulation+'/velocity_y/velocity_y_t='+str(t*jump)+'.txt') 
    file3=os.path.join('./'+simulation+'/velocity_z/velocity_z_t='+str(t*jump)+'.txt') 
    file4=os.path.join('./'+simulation+'/phi/phi_t='+str(t*jump)+'.txt') 
    a_v_x,b_v_x,c_v_x,z_v_x = np.loadtxt(file1,delimiter=' ', unpack=True)
    a_v_y,b_v_y,c_v_y,z_v_y = np.loadtxt(file2,delimiter=' ', unpack=True)
    a_v_z,b_v_z,c_v_z,z_v_z = np.loadtxt(file3,delimiter=' ', unpack=True)
    a_phi,b_phi,c_phi,z_phi = np.loadtxt(file4,delimiter=' ', unpack=True)
    now = datetime.now()
    sys.stdout.write("\r Data loaded "+str(float(int((t+1)/DataFiles*1000))/10)+"% - "+str(now.strftime("%d/%m/%Y %H:%M:%S")))
    sys.stdout.flush()
    for i in range(0,Nx):
        for j in range(0,Ny):
            for k in range(0,Nz):
                
                v_x[i][j][k] =  z_v_x[i*Ny*Nz +j*Nz+k] 
                v_y[i][j][k] =  z_v_y[i*Ny*Nz +j*Nz+k] 
                v_z[i][j][k] =  z_v_z[i*Ny*Nz +j*Nz+k]
                phi[i][j][k] =  z_phi[i*Ny*Nz +j*Nz+k]


    #Calculation of shear

    grad_vx = grad_roll(v_x,maskT)
    grad_vy = grad_roll(v_y,maskT)
    grad_vz = grad_roll(v_z,maskT)

    shear_xy = calculateshearxy(grad_vx,grad_vy,maskT)
    shear_yz = calculateshearyz(grad_vy,grad_vz,maskT)
    shear_xz = calculateshearxz(grad_vx,grad_vz,maskT)

    shear_total[maskT] = (shear_xy[maskT]*shear_xy[maskT] + shear_yz[maskT]*shear_yz[maskT] + shear_xz[maskT]*shear_xz[maskT])**(0.5)

    plot_shear_plane(phi,v_x,v_y,v_z) #Plot shear every step
    save_vti_file(phi,v_x,v_y,v_z, Nx, Ny, Nz, 'tot_t='+str(t*jump).zfill(8)+'.vti') #Save vti file for Paraview visualisation

    
    
    
    
    
    
    
