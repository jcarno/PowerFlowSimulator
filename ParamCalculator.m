%set up line params
e0=8.854E-12;
Dab=9.75*2;
Dbc=Dab;
Dca=Dab+Dbc;
n=2;
w=2*pi*60;
d=1.5;
Rpmi=0.385;
GMR=0.0217;
r=0.642/2/12;
metersToMile=1609;
Sbase=100;
Vbase=230;
Zbase=Vbase^2/Sbase;
Ybase=1/Zbase;
name='L6';
len=35;

%find Dsl and Dsc and Deq
Deq=sqrt(Dab^2+Dbc^2+Dca^2);

if n==1
    Dsc=r;
    Dsl=GMR;
elseif n==2
    Dsc=sqrt(d*r);
    Dsl=sqrt(d*GMR);
elseif n==3
    Dsc=nthroot(d^2*r,3);
    Dsl=nthroot(d^2*GMR,3);
else
    Dsc=1.091*nthroot(d^3*r,4);
    Dsl=1.091*nthroot(d^3*GMR,4);
end

Y=2*pi*e0/log(Deq/Dsc)*metersToMile*w*len;
X=2E-7*log(Deq/Dsl)*metersToMile*w*len;
R=Rpmi*len/n;

Rpu=R/Zbase;
Xpu=X/Zbase;
Ypu=Y/Ybase;

disp(['Line: ' name])
disp(['Length: ' num2str(len)])
disp(['Rpu: ' num2str(Rpu)])
disp(['Xpu: ' num2str(Xpu)])
disp(['Ypu: ' num2str(Ypu)])

    
    