Assume Avec is the matrix storing the scattering distirbutions, the coloums are different wavelengths, the rows are different angles. lambdanum is number of wavelengths, lambdastart is the starting wavelength (in nm), lambdaend is the ending wavelength (in nm). d65 is the d65 specturm.

# convert spectral distribution to XYZ (in CIE 1931 space)
xyz = ToXYZ(Avec, lambdanum, lambdastart, lambdaend, d65);

# convert XYZ to rgb
rgb = XYZToRGB(xyz);

# optional: convert floating point numbers to 8 bit
uint8im = uint8(255 * rgbga);
