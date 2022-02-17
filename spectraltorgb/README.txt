d65 = data.d65;
scale = 5;
xyz = ToXYZ(Avec*scale, lambdanum, lambdastart, lambdaend, d65);
rgb = XYZToRGB(xyz);
rgbga = lin2rgb(rgb);
uint8im = uint8(255 * rgbga);

Rv = reshape(uint8im(:,1),[phionum, phiinum, thetanum]);
Gv = reshape(uint8im(:,2),[phionum, phiinum, thetanum]);
Bv = reshape(uint8im(:,3),[phionum, phiinum, thetanum]);
R = reshape(Rv(:,:,thetaindex),[phionum, phiinum]);
G = reshape(Gv(:,:,thetaindex),[phionum, phiinum]);
B = reshape(Bv(:,:,thetaindex),[phionum, phiinum]);
R = concatenateimg(R, phionum);
G = concatenateimg(G, phionum);
B = concatenateimg(B, phionum);

% num = 10;
% R = removeforward(R, num, phionum);
% G = removeforward(G, num, phionum);
% B = removeforward(B, num, phionum);
imgRGB = cat(3, R, G, B);
