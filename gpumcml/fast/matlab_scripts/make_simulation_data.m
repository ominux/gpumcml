clear all; close all; clc

mua_e = 0.01;
mua_d = 0;
thi = 0;



% gammas = linspace(0.95,1.27,6)
% musp_vs = linspace(1.9,3.8,6) * 10 %cm^-1
% gs = [.07, .14, .3]
% 
% gammas = linspace(0.95,1.27,20)
% musp_vs = linspace(1.0,6,50) * 10  %cm^-1
% gs = [.9]

% gammas = linspace(0.95,1.27,20);
gammas = 0.95;
% gammas = gammas(6:end)
% gammas = gammas(5);
% musp_vs = linspace(1.0,6,50) * 10 %cm^-1
% musp_vs = musp_vs(44:end);
gs = [.9]

% gammas = linspace(0.95,1.27,10);
% 
% gammas = gammas(2:end-1);
% 
% musp_vs = linspace(1.0,6,10) * 10; %cm^-1
% 
% musp_vs = musp_vs(2:end-1);
% 
% gs = [.85 .95];

%%
% gammas = 0.95;
musp_vs = 1*10;
% gs = .07;

% gammas = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];%[1.65, 1.75, 1.85, 1.95]
% musp_vs = [0.3, 0.5, 1, 3, 5, 10]*10;%linspace(10,60,5) %cm^-1
% gs =  [0.75, 0.85, 0.95]; %[.75:.05:0.95]

% gammas = [1.7500,  1.7795, 1.8089, 1.8384, 1.8679, 1.8974];




% gammas = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
% musp_vs = [0.5, 1.0510, 1.5408, 2.0306, 3.0102, 3.5000]*10;
% gs =  [0.9];

% %WAVELENGTH 1
% n_medium_input = 1.333;
% n_particle_input = 1.61;
% lambda = 450;
% 
% gammas = [1.03, 1.03, 1.8, 1.8, 2.08, 2.08];
% musp_vs = [3.6, 5.39, 2.26, 3.4, 2.16, 3.25]*10;
% gs = [.14, .14, .58, .58, .93, .93];
% d_particles = [.0878, .0878, .19, .19, 0.99, 0.99]*10^3;
% d_stds = [.01, .01, .01, .01, .03, .03];

% 
% %WAVELENGTH 3
% n_medium_input = 1.333;
% n_particle_input = 1.59;
% lambda = 620;
% 
% gammas = [.97, .97, 1.24, 1.24, 2.17, 2.17];
% musp_vs = [1.11, 1.67, 1.54, 2.31, 1.87, 2.80]*10;
% gs = [.07, .07, .30, .30, .92, .92];
% d_particles = [.0878, .0878, .19, .19, 0.99, 0.99]*10^3;
% d_stds = [.01, .01, .01, .01, .03, .03];

% gammas = [1.7500, 1.7795, 1.8089, 1.8384, 1.8679, 1.8974];
% musp_vs = [0.5, 1.0510, 1.5408, 2.0306, 3.0102, 3.5000]*10;
% gs = [.9];

%%

for gam = gammas
    for g = gs
        if gam > 1 + g
            continue
        end
%         if gam < 1.0847
%             continue
%         end
        musp_v_cm = musp_vs;
    
        RunMCw1gamma1g_original(gam,musp_v_cm,g)
    end  
end
%%
close all;
% gammas = linspace(0.95,1.27,20);

% for mua_e = linspace(0.01,5,10);
for mua_e = 0.01
    for gam = gammas
    %     if gam < 1.0847
    %         continue
    %     end
        for g = gs
            for musp_v_cm = musp_vs
                if gam > 1 + g
                    continue
                end
                data = load(['Test/Simulation_gamma' num2str(gam) '_musp_' num2str(musp_v_cm) '_g_' num2str(g) '_mua_' num2str(mua_e) '.mat']);

                fx = [.01 .025 .05:.05:1.8];
    %             fx = [0:.05:1];


    %             figure(1)
    %             semilogy(r_log, R_log)
    %             hold all;
    % 
    %             xlabel('distance (cm)')
    %             ylabel('R (cm^-^2)')


                r_log = [data.dr:data.dr:data.dr*data.Ndr] * 10;
                R_log = data.MCoutput.refl_r * 1/100;

    %             figure(1)
    %             semilogy(r_log, R_log)
    %             hold all;
    %             xlabel('distance (mm)')
    %             ylabel('R (1/mm)')



    %             figure(2)
                SFDR_1Y = ht(R_log,r_log,fx*2*pi);
    %             semilogy(fx,SFDR_1Y,'DisplayName',['mu ' num2str(musp_v_cm) ' gamma ' num2str(gam) ' g ' num2str(g)])
    %             hold all;
    %             xlabel('f (1/mm)')
    %             ylabel('R')

                save(['Test/SFDR/SFDR_mu_' num2str(musp_v_cm) '_gamma_' num2str(gam) '_g_' num2str(g) '_mua_' num2str(mua_e) '.mat'],'SFDR_1Y');
            end
        end
    end
end

%%
% xlabel('f (1/mm)')
% ylabel('R_M_C_M')
% figure(2)
% legend show;