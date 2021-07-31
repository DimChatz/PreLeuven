from math import *
import matplotlib.pyplot as plt
import astropy.coordinates as coord
import astropy.units as u  # or: from astropy import units as u
import numpy as np


longitudes= []
latitudes= []

def get_data():
    global longitudes
    global latitudes

    with open("Out-Z=26.txt", "r") as f:  # opens file. This syntax doesn't need f.close
        lines = f.readlines()  # reads whole file and splits it in lines

        for i in lines:  # for all lines
            longitudes.append(float(i.split()[2]))  # adds to result_lon elements from second column
            latitudes.append(float(i.split()[3]))   # adds to result_lat elements from third column
    longitudes = [radians(x) for x in longitudes]   # it's better to convert degrees to rad in the beginning
    latitudes  = [radians(x) for x in latitudes]

    return len(longitudes)


# ------------------------------------------------------- #
# --------------  CALL GET_DATA FUNCTION  --------------- #
# ------------------------------------------------------- #
number_of_points = get_data()   #somehow you need to know how many points you have to scatter,
                                #so a way is to make get_data return the length of labels array

# ------------------------------------------------------- #
# --------------------  CREATE PLOT  -------------------- #
# ------------------------------------------------------- #
test_fig = plt.figure(figsize=(12, 9))
ax_test = test_fig.add_subplot(111, projection="mollweide", facecolor='LightCyan')  #projection options: mollweide, lambert, hammer, aitoff

# ------------------------------------------------------- #
# --------------  CONFIGURE PLOT'S DETAILS  --------------#
# ------------------------------------------------------- #
ax_test.set_title('After Depropagation Z=26\n',fontsize=30)
ax_test.set_xticklabels(np.arange(150, -180, -30), fontsize=18)
ax_test.set_yticklabels(np.arange(-75, +90, +15), fontsize=18)
ax_test.grid(True)

# -------------------------------------------------------#
# --------------------   FILL PLOT  -------------------- #
# -------------------------------------------------------#
for i in range(0, number_of_points):
    temp_lon = coord.Angle(longitudes[i] * u.radian)
    temp_lat = coord.Angle(latitudes[i] * u.radian)
    ax_test.scatter(-temp_lon.radian, temp_lat.radian, marker='.', color='b')
test_fig.savefig('hahah AD-Z=26')

plt.show()


