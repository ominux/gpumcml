% MCMLr_f.m
%  
% 3/19/2012
%Edited by Andrew Stier
% 10/25/2016

function [MCoutput, dr, Ndr] = MCMLr_r(mua_e, mua_d, mus, thi, g, gamma)

% INPUTS
%   mua_e   - absorption coefficient for 1 layer, epi abs for 2 layer (cm^-1)
%   mua_d   - dermal absorption coefficient for two layer only (cm^-1)
%   mus     - scattering coefficient (cm^-1)
%   thi     - epidermal thickness (cm) (=0 for 1 layer)
%   g       - scattering anisotropy
%   d       - vector of collection fiber distances from source (cm)
%   r       - collection fiber radius (cm)
%
% OUTPUTS
%   R   - Light collected by fiber

%% Create Input File for MCML
%             n         mua     mus     g       d
if thi == 0
    layers = [1.33      mua_e   mus     g       1E9];
%     fprintf('debug 1')
else
    layers = [1.33      mua_e   mus     g       thi;
              1.33      mua_d   mus     g       1E9];
end

photons     = 1E5 ;   % Number of photon packets to simulate
n_above     = 1; % Refractive index of the medium above
n_below     = 1.33;  % Refractive index of the medium below
dz          = 0.001; % Spatial resolution of detection grid, z-direction [cm]
Ndz         = 1;   % Number of grid elements, z-direction
dr          = 1*10^(-5); % Spatial resolution of detection grid, r-direction [cm]
Ndr         = 1.5*10^5;  % Number of grid elements, r-direction

% dr = .05;
% Ndr = 2000;

Nda         = 1;    % Number of grid elements, angular-direction

create_MCML_input_file_1('mcml',photons,layers,n_above,n_below,dz,dr,Ndz,Ndr,Nda);

%% Run GPUMCML
system('GPUMCML mcml.mci') %% Random Seed
% system('GPUMCML_file_no_Rd_ra_scale mcml.mci') %% Random Seed %% remember to change the data.txt and the name of the program!

%% Run Conv (Edit conv_input.txt file to change Conv parameters)
% system('Conv.exe<conv_input.txt')
% 
% 
movefile('mcml.mco',['mcml_gamma' num2str(gamma) '_musp_' num2str(mus) '_g_' num2str(g) '.mco'])

MCoutput = read_file_mco(['mcml_gamma' num2str(gamma) '_musp_' num2str(mus) '_g_' num2str(g) '.mco']);

save(['Test/Simulation_gamma' num2str(gamma) '_musp_' num2str(mus) '_g_' num2str(g) '.mat'])

%% Compute Results
% dataRr = dlmread('out.Rrc','\t',1,0);
% distance = dataRr(:,1);
% refl = dataRr(:,2);

% figure(3)

% plot(distance,refl)
    

