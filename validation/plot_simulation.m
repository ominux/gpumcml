clear all
close all
clc

CMP = 1;

%% Parse simulation output files
[FileName_ref,PathName_ref] = uigetfile('*.mco','Select .mco file to plot');
ref = read_file_mco([PathName_ref FileName_ref]);

if CMP
    [FileName_cmp1,PathName_cmp1] = uigetfile('*.mco','Select comparison .mco file to plot');
    cmp1 = read_file_mco([PathName_cmp1 FileName_cmp1]);
end

%% Set parameters
na = ref.step_num(3);
da=pi/(2*na); 
a = (0:na-1)*da;

nlayers = ref.layers;
Nz=ref.step_num(1); 
Nr=ref.step_num(2);
dz=ref.step_size(1);
dr=ref.step_size(2);
Azr   = ref.abs_rz; 
Fzr   = ref.f_rz; 
sz = 12;  % Font size

%% Generate Reflectance and Transmittance Plots
% Create r, z vectors
z = (0:ref.step_num(1)-1)*ref.step_size(1);
r = (0:ref.step_num(2)-1)*ref.step_size(2);

figure
subplot(1,2,1); 
semilogy(r,ref.refl_r);hold on
if CMP
    semilogy(r,cmp1.refl_r,'r')
    legend(FileName_ref, FileName_cmp1)
end
%title('Reflectance(r)')
xlabel('r [cm]')
ylabel('Reflectance [1/cm^{2}]')


subplot(1,2,2); 
plot(a,ref.refl_a);hold on
if CMP
    plot(a,cmp1.refl_a,'r')
    legend(FileName_ref, FileName_cmp1)
end
%title('Reflectance(a)')
xlabel('a [rad]')
ylabel('Reflectance [1/sr]')


figure
subplot(1,2,1); 
plot(r,ref.trans_r);hold on
if CMP
    plot(r,cmp1.trans_r,'r')
    legend(FileName_ref, FileName_cmp1)
end
%title('Transmittance(r)')
xlabel('r [cm]')
ylabel('Transmittance [1/cm^{2}]')


subplot(1,2,2); 
plot(a,ref.trans_a);hold on
if CMP
    plot(a,cmp1.trans_a,'r')
    legend(FileName_ref, FileName_cmp1)
end
%title('Transmittance(a)')
xlabel('a [rad]')
ylabel('Transmittance [1/sr]')

%% Set common parameters for plots below
z     = ((1:Nz)' - 0.5)*dz;
r     = ((1:Nr)' - 0.5)*dr;
rm = [((-Nr+1:-1) - 0.5)*dr (r'- dr)]' + dr/2;
u = (2:length(rm)-2);
v = (1:Nz-1);

%% Plot A[r][z] Distribution
Azrm  = zeros(Nz, 2*Nr-1);
Azrm(:,Nr:2*Nr-1) = Azr;
Azrm(:,1:Nr) = Azr(:,Nr:-1:1);

figure(3);
set(figure(3),'position',[0   476   1200   320])
imagesc(rm(u),z(v),log10(Azrm(v,u)));
set(gca,'fontsize',sz)
xlabel('r [cm]')
ylabel('z [cm]')
ylim([0 0.6]); 
xlim([-2 2]); 
set(gca,'XTickLabel',{'2','1.5','1','0.5','0','0.5','1','1.5','2'})

title('log_1_0( A[r][z] ) - Absorption Probability Density [1/cm^3] ')
colorbar
colormap hot;

% Create textbox
annotation(figure(3),'textbox',[0.85 0.05 0.05236 0.09118],...
    'String',{'1/cm^3'},...
    'FitBoxToText','off',...
    'LineStyle','none');

%% Plot Fluence Distribution
Fzrm  = zeros(Nz, 2*Nr-1);
Fzrm(:,Nr:2*Nr-1) = Fzr;
Fzrm(:,1:Nr) = Fzr(:,Nr:-1:1);

figure(4);
set(figure(4),'position',[0   476   1200   320])
imagesc(rm(u),z(v),log10(Fzrm(v,u)));
set(gca,'fontsize',sz)
xlabel('r [cm]')
ylabel('z [cm]')
ylim([0 0.6]); 
xlim([-2 2]); 

set(gca,'XTickLabel',{'2','1.5','1','0.5','0','0.5','1','1.5','2'})

title('log_1_0( F[r][z] ) - Photon Fluence for Impulse Response [1/cm^2] ')
colorbar
colormap hot;

% Create textbox
annotation(figure(4),'textbox',[0.85 0.05 0.05236 0.09118],...
    'String',{'1/cm^2'},...
    'FitBoxToText','off',...
    'LineStyle','none');

%% Plot Fluence Contours
z=-z;
figure (5); 
set(figure(5),'position',[0   476   1200   320])
contourlevel=[1000 100 30 10 1 0.01 0.001 0.00001];
[C,h]=contour(rm(u), z(v), Fzrm(v,u), contourlevel);
clabel(C,h);
colormap hot; 

xlabel('r [cm]')
ylabel('z [cm]')
ylim([-0.6 0]); 
xlim([-2 2]); 
set(gca,'YTickLabel',{'0.5','0.4','0.3','0.2','0.1','0'})
set(gca,'XTickLabel',{'2','1.5','1','0.5','0','0.5','1','1.5','2'})
title('F[r][z] Contours - Photon Fluence for Impulse Response [1/cm^2] ')
colormap jet; 