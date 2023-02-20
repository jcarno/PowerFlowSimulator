%tranformer calculations
sBase=100;
sRated=200;
vBase=18;
vRated=18;
zPuRated=0.105;
xr=12;
name='T2';

znewPu=zPuRated*vRated^2/sRated/(vBase^2/sBase);
Rpu=znewPu*cos(atan(xr));
Xpu=znewPu*sin(atan(xr));

disp(['Transformer: ' name])
disp(['R: ' num2str(Rpu)])
disp(['X: ' num2str(Xpu)])