"""
Made on: 20240725
By @ Adrianna Zgraj
"""

## Imports
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.optimize import curve_fit

## Inputs Needed
# Load the image stack from a TIFF file
imageStack = tiff.imread('C:\\Users\\adriannazgraj\\Desktop\\PSF_2.tif')

# Define the real size of the pixel in nanometers (nm)
pixel_size_nm = 27  # Change this to your pixel size

image_transpose = 1  # Transposes the image to correct dimensions (if you get the image from scanimage); 1 = yes, 0 = no


## Preprocessing

if image_transpose == 1:
    imageStack = np.transpose(imageStack, (1,2,0))
else:
    pass

# Get the dimensions of the image stack
ny, nx, nz = imageStack.shape

# Sum the image stack along the z-axis to get a 2D projection
projection = np.sum(imageStack, axis=2)

# Find the coordinates of the brightest spot (approximate bead center)
maxIndex = np.unravel_index(np.argmax(projection, axis=None), projection.shape)
y, x = maxIndex

# Extract intensity profiles around the bead center
xProfile = np.asarray(imageStack[y, :, nz//2])
yProfile = np.asarray(imageStack[x, :, nz//2])
zProfile = np.asarray(imageStack[y, x, :])

xData = np.arange(len(xProfile))
yData = np.arange(len(yProfile))
zData = np.arange(len(zProfile))

# Create the gaussian function
def gaussian(x, a, b, c, d):
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2))) + d

## FWHM calculations
x_p0 = [xProfile.max(),xData.mean(),xData.std(),xProfile.min()]
x_popt, x_pcov = curve_fit(gaussian, xData, xProfile,x_p0)
x_fwhm = round((2*np.sqrt(2*np.log(2))*np.abs(x_popt[2])) * pixel_size_nm,3)
print("X FWHM is: " + str(x_fwhm))


y_p0 = [yProfile.max(),yData.mean(),yData.std(),yProfile.min()]
y_popt, y_pcov = curve_fit(gaussian, yData, yProfile, y_p0)
y_fwhm = round((2*np.sqrt(2*np.log(2))*np.abs(y_popt[2])) * pixel_size_nm,3)
print("Y FWHM is: " + str(y_fwhm))

z_p0 = [zProfile.max(),zData.mean(),zData.std(),zProfile.min()]
z_popt, z_pcov = curve_fit(gaussian, zData, zProfile, z_p0)
z_fwhm = round((2*np.sqrt(2*np.log(2))*np.abs(z_popt[2]))*pixel_size_nm,3)
print("Z FWHM is: " + str(z_fwhm))


## Plotting
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(xData*pixel_size_nm, xProfile, 'b.', markersize=10, label='Data')
axs[0].plot(xData*pixel_size_nm, gaussian(xData,*x_popt), 'r-', markersize=10, label='Fitted curve')
axs[0].set_title('X Profile and Gaussian Fit')
axs[0].set_xlabel('nm')
axs[0].set_ylabel('Intensity')
axs[0].text(x=(xData.max()/3),y=(xProfile.min() + (xProfile.max()-xProfile.min())/2),s=f"FWHM:{x_fwhm} nm")
axs[0].legend()

axs[1].plot(yData*pixel_size_nm, yProfile, 'b.', markersize=10, label='Data')
axs[1].plot(yData*pixel_size_nm, gaussian(yData,*y_popt), 'r-', markersize=10, label='Fitted curve')
axs[1].set_title('Y Profile and Gaussian Fit')
axs[1].set_xlabel('nm')
axs[1].set_ylabel('Intensity')
axs[1].text(x=(yData.max()/3),y=(yProfile.min() + (yProfile.max()-yProfile.min())/2),s=f"FWHM:{y_fwhm} nm")
axs[1].legend()

axs[2].plot(zData*pixel_size_nm, zProfile, 'b.', markersize=10, label='Data')
axs[2].plot(zData*pixel_size_nm, gaussian(zData,*z_popt), 'r-', markersize=10, label='Fitted curve')
axs[2].set_title('Z Profile and Gaussian Fit')
axs[2].set_xlabel('nm')
axs[2].set_ylabel('Intensity')
axs[2].text(x=(zData.max()/3),y=(zProfile.min() + (zProfile.max()-zProfile.min())/2),s=f"FWHM:{z_fwhm} nm")
axs[2].legend()

plt.tight_layout()
plt.show()

print("hi mom")
