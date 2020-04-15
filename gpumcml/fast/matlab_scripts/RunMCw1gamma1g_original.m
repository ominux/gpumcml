%% Run MC simulations using Modified HG (Sub-diffuse)
% 2/24/2019
% Yao Zhang

function MCoutput = RunMCw1gamma1g_original(gamma,musp_vs,g1)
    %% Input parameters
%     g1 = 0.9; % fix g1
    % gamma = 2.14; % Note that you can input Only One value for gamma in this program

    Flag_Plot = 0; % 1: Plot the histogram of the scattering angles to check the phase function; 0: no plotting

    % musp_vs = [30];% reduced scattering vector (cm^-1)  (Test one value here but you can have multiple values)
    mua_v  = [0.01]; % absorption vector (cm^-1)
%     mua_v = linspace(0.01,5,10);

    photons     = 1E7;   % Number of photon packets to simulate
    n_above     = 1; % Refractive index of the medium above
    n_below     = 1.33;  % Refractive index of the medium below
    dz          = 0.01; % Spatial resolution of detection grid, z-direction [cm]
    dr          = 1*10^(-5); % Spatial resolution of detection grid, r-direction [cm]
    Ndz         = 1;   % Number of grid elements, z-direction
    Ndr         = 1.5*10^5;  % Number of grid elements, r-direction
    Nda         = 1;    % Number of grid elements, angular-direction

    %% Sampling for MHG to generate inverse CDF in data.txt
    % (You can skip this step if you made a file for the current pair of g and gamma previously)
    tic
    N=20000;
    epsilon=linspace(0,1,N); % Uniform Distribution
    A=[];

    for Num=1:size(gamma,2)
        g=(5*g1*gamma(Num) - 5*gamma(Num) + (25*g1^2*gamma(Num).^2 + 40*g1^2 - 50*g1*gamma(Num).^2 + 30*g1*gamma(Num) + 25*gamma(Num).^2 - 30*gamma(Num) + 9).^(1/2) + 3)/(10*g1);
        a=g1./g;% Right value for Beta

        costC=zeros(N,1);
        for time=1:N
            randnum=epsilon(time);
            costT = 0;
            for x=linspace(-1,1,N)
                derror = randnum-(a*(1-g*g)/(2*g)*((1+g*g-2*g*x)^(-0.5)-(1+g*g+2*g)^(-0.5))+(1-a)*(x*x*x+1)/(2));
                if derror>0
                    costT = x;
                    continue
                else
                    costT = (x + costT)/2;
                    break
                end
            end
            costC(time)=costT;
        end
        A=[A costC];
        % Plot
        if Flag_Plot
            figure
            h0=histogram(costC,21,'Normalization','pdf');
            hold on
            %theoretical MHG phase function
            cost0=linspace(-1,1,N);
            pcost0=a*(1-g*g)./(2*(1+g*g-2*g*cost0).^(3/2))+(1-a)*3/(2)*cost0.*cost0;
            plot(cost0,pcost0);
            set(gca, 'YScale', 'log')
            xlabel('cos(\theta)')
            ylabel('probability')
            legend('Sampling (Numeric/Discretized)','MHG phase function')
            title(['g1=',num2str(g1),' gamma=',num2str(gamma(Num))])
            xlim([-1 1])
        end
    end
    toc

    fileID = fopen(['CDF_g_' num2str(g1) 'gamma_' num2str(gamma) '.txt'],'w');
    formatSpec='%8.6f \n';
    fprintf(fileID,formatSpec,A'); % Very important!!! Pay attention to the writing format!
    fclose(fileID);
    save(['CDF_g_' num2str(g1) 'gamma_' num2str(gamma) '.mat'])


    %% Always run this to replace data.txt with the desired inverse CDF file
    movefile(['CDF_g_' num2str(g1) 'gamma_' num2str(gamma) '.txt'],'data.txt')


    %% Run simulation one by one
    g       = g1;         % scattering anisotropy
    gammas  = gamma;      % Gamma
    for mua_e = mua_v
        mua_d = 100;
        thi = 0;
        for gamma = gammas % To run multiple values of gammas we will need to change the exe.file. Please use one gamma each time for now
            for musp_v = musp_vs
                mus = musp_v/(1-g);
                %% Create Input File for MCML
                %             n         mua     mus     g   d    gamma
                if thi == 0
                    layers = [1.37      mua_e   mus     g   1E2 1]; % One gamma can use the same exe file
                else
                    layers = [1.37      mua_e   mus     g   thi  1;
                        1.37      mua_d   mus     g   1E9  1];
                end

                create_MCML_input_file('mcml','data.txt',photons,layers,n_above,n_below,dz,dr,Ndz,Ndr,Nda);

                %% Run GPUMCML
                system('./gpumcml.sm_20 mcml.mci') %% Random Seed %% remember to change the data.txt and the name of the program!

                movefile('mcml.mco',['mcml_gamma' num2str(gamma) '_musp_' num2str(musp_v) '_g_' num2str(g1) '.mco'])

                MCoutput = read_file_mco(['mcml_gamma' num2str(gamma) '_musp_' num2str(musp_v) '_g_' num2str(g1) '.mco']);

                %% Plot the simulation results
                %         figure
                %         r = 0:dr:dr*Ndr-dr;
                %         Rd_r = MCoutput.refl_r;
                %         semilogy(r,Rd_r,'--'); % Rd_r from MCML simulation output
                %         xlabel('Radius r [cm]')
                %         ylabel('Diffuse reflectance R_d (cm^-^2)')
                %         title(['musp =', num2str(musp_v/10),' mm^-^1 g1 =',num2str(g), ' gamma =',num2str(gamma)])

                save(['Test/Simulation_gamma' num2str(gamma) '_musp_' num2str(musp_v) '_g_' num2str(g1) '_mua_' num2str(mua_e) '.mat'],'dr','MCoutput','Ndr')
            end
        end
    end
end


