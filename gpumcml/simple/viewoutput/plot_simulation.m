clear all
close all
clc

%% Parse simulation output files
[FileName_ref,PathName_ref] = uigetfile('*.mco','Select .mco file to plot');
ref = read_file_mco([PathName_ref FileName_ref]);

%% Set parameters
na = ref.step_num(3);
da=pi/(2*na); 
a = (0:na-1)*da;

nlayers = ref.layers;
Nz=ref.step_num(1); 
Nr=ref.step_num(2);
dz=ref.step_size(1);
dr=ref.step_size(2);
Azr  = ref.abs_rz; 
Fzr   = ref.f_rz; 

%% Generate Reflectance and Transmittance Plots
% Create r, z vectors
z = (0:ref.step_num(1)-1)*ref.step_size(1);
r = (0:ref.step_num(2)-1)*ref.step_size(2);

figure
subplot(1,2,1); 
semilogy(r,ref.refl_r);hold on
xlabel('r [cm]')
ylabel('Reflectance [1/cm^{2}]')

subplot(1,2,2); 
plot(a,ref.refl_a);hold on
xlabel('a [rad]')
ylabel('Reflectance [1/sr]')

figure
subplot(1,2,1); 
plot(r,ref.trans_r);hold on
xlabel('r [cm]')
ylabel('Transmittance [1/cm^{2}]')

subplot(1,2,2); 
plot(a,ref.trans_a);hold on
xlabel('a [rad]')
ylabel('Transmittance [1/sr]')

%% Set common parameters for plots below
z     = ((1:Nz)' - 0.5)*dz;
r     = ((1:Nr)' - 0.5)*dr;
rm = [((-Nr+1:-1) - 0.5)*dr (r'- dr)]' + dr/2;
u = (2:length(rm)-2);
v = (1:Nz-1);

%% Plot Absorption Probability Density Distribution
Azrm  = zeros(Nz, 2*Nr-1);
Azrm(:,Nr:2*Nr-1) = Azr;
Azrm(:,1:Nr) = Azr(:,Nr:-1:1);

figure
imagesc(rm(u),z(v),log10(Azrm(v,u)));
xlabel('r [cm]')
ylabel('z [cm]')
title('log_1_0( A[r][z] ) - Absorption Probability Density [1/cm^3] ')
colorbar
colormap jet;
 
%% Plot Fluence Distribution
Fzrm  = zeros(Nz, 2*Nr-1);
Fzrm(:,Nr:2*Nr-1) = Fzr;
Fzrm(:,1:Nr) = Fzr(:,Nr:-1:1);

figure
imagesc(rm(u),z(v),log10(Fzrm(v,u)));
xlabel('r [cm]');   
ylabel('z [cm]'); 
title('log_1_0( F[r][z] ) - Photon Fluence for Impulse Response [1/cm^2] ')
colorbar
colormap jet; 
   