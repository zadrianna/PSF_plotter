"""
Made on: 20240725
By @ Adrianna Zgraj
"""

## Imports
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

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


def find_fwhm(arr):
    max_idx = np.argmax(arr)
    half_max = round(arr.min() + ((arr.max()-arr.min())/2))
    indices = np.where(np.isclose(arr, half_max, atol=3000))[0]  # Get all indices where value is x
    less_indices = indices[indices < max_idx]  # Get all indices less than y
    greater_indices = indices[indices > max_idx]  # Get all indices greater than y
    if len(less_indices) > 0 and len(greater_indices) > 0:
        fwhm = (greater_indices[0] - less_indices[0])*pixel_size_nm

        return fwhm  # Return first index which is less than y and which is greater than y
    else:
        return None, None  # In case no matching indices found

## FWHM

x_fwhm = find_fwhm(xProfile)
y_fwhm = find_fwhm(yProfile)
z_fwhm = find_fwhm(zProfile)


## Plotting

xData = np.arange(len(xProfile))
yData = np.arange(len(yProfile))
zData = np.arange(len(zProfile))

fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(xData*pixel_size_nm, xProfile, 'b.', markersize=10, label='Data')
axs[0].set_title('X Profile and Gaussian Fit')
axs[0].set_xlabel('nm')
axs[0].set_ylabel('Intensity')
axs[0].text(x=(xData.max()/3),y=(xProfile.min() + (xProfile.max()-xProfile.min())/2),s=f"FWHM:{x_fwhm} nm")

axs[1].plot(yData*pixel_size_nm, yProfile, 'b.', markersize=10, label='Data')
axs[1].set_title('Y Profile and Gaussian Fit')
axs[1].set_xlabel('nm')
axs[1].set_ylabel('Intensity')
axs[1].text(x=(yData.max()/3),y=(yProfile.min() + (yProfile.max()-yProfile.min())/2),s=f"FWHM:{y_fwhm} nm")

axs[2].plot(zData*pixel_size_nm, zProfile, 'b.', markersize=10, label='Data')
axs[2].set_title('Z Profile and Gaussian Fit')
axs[2].set_xlabel('nm')
axs[2].set_ylabel('Intensity')
axs[2].text(x=(zData.max()/3),y=(zProfile.min() + (zProfile.max()-zProfile.min())/2),s=f"FWHM:{z_fwhm} nm")

plt.tight_layout()
plt.show()
