# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:19:27 2020

@author: cory1
"""
L=30000
xlist = [300]
ylist = [300]
zlist = [300]
position_array = np.zeros((3,L))
for ii in range(0, L):
    new = np.random.randint(-1,high = 2,size = 3)
    xlist.append(new[0])
    ylist.append(new[1])
    zlist.append(new[2])
    k = int(np.sum(xlist))
    p = int(np.sum(ylist))
    h = int(np.sum(zlist))
    position_array[0,ii] = k
    position_array[1,ii] = p
    position_array[2,ii] = h
    x1 = int(position_array[0,ii])
    y1 = int(position_array[1,ii])
    z1 = int(position_array[2,ii])
    if edge[x1,y1,z1] == 1:
        print("boo")
        break

outfile = TemporaryFile(r"C:\Users\cory1\OneDrive\Documents\test-folder\abnormaility-cuts)
np.save(outfile, edge)
np.load

edge = np.zeros((LMLO[1],500, CC[1]))
for k in range(0,LMLO[1]):
    for j in range(0, CC[1]):
        for i in range(0,500):
            if dimesional_array[k,i,j] == 1:
                edge[k,i,j] = 1
                break
x = []
y = []
z = []
for k in range(0, LMLO):
    for j in range(0, CC):
        for i in range(0, 500):
            if edge[k,i,j] == 1:
                x.append(i)
                y.append(j)
                z.append(k)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x,y,z)
ax.set_xlim3d(700,0)
ax.set_ylim3d(0,CC)
ax.set_zlim3d(0, LMLO)
ax.view_init(90,80)
for k in reversed(range(0,LMLO[1])):
    for j in reversed(range(0, CC[1])):
        for i in reversed(range(0,500)):
            if dimesional_array[k,i,j] == 1:
                edge[k,i,j] = 1
                break
            
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x,y,z)
ax.set_xlim3d(700,0)
ax.set_ylim3d(0,CC[1])
ax.set_zlim3d(0, LMLO[1])
ax.view_init(90,80)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x,y,z)
ax.set_xlim3d(700,0)
ax.set_ylim3d(0,CC[1])
ax.set_zlim3d(0, LMLO[1])
ax.view_init(40,80)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x,y,z)
ax.set_xlim3d(700,0)
ax.set_ylim3d(0,CC[1])
ax.set_zlim3d(0, LMLO[1])
ax.view_init(100,80)           