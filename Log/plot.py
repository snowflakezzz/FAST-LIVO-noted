import numpy as np
import matplotlib.pyplot as plt

def drawLine( tumpath ):
    cont = np.loadtxt(tumpath)
    time = cont[:,0]
    x = cont[:,1]
    y = cont[:,2]
    z = cont[:,3]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(x, y, z, label='3D Curve', color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

if __name__=="__main__":
    drawLine("tum.txt")