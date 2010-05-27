clear all
close all
clc

GPUstr = './../../sim_output/gpumcml/Fermi/EA/sevenlayer_600nm_100M.mco'
CPUstr = './../../sim_output/cpumcml/MT_RNG/sevenlayer_600nm_100M_run1.mco'
GPU_Astr = './../../sim_output/gpumcml/Fermi/EA/No absorption detection/sevenlayer_600nm_100M.mco'

CPUtime = 18482; % sec
GPUtime = 23.23; % sec
GPU_Atime = 16.6; % sec

%[FileName_ref,PathName_ref] = uigetfile('*.mco','Select reference .mco file');
%[FileName_cmp1,PathName_cmp1] = uigetfile('*.mco','Select comparison .mco file1');
%[FileName_cmp2,PathName_cmp2] = uigetfile('*.mco','Select comparison .mco file2');

gpu = read_file_mco(GPUstr);
cpu = read_file_mco(CPUstr);
gpu_a = read_file_mco(GPU_Astr);

r = (1:cpu.step_num(2))*cpu.step_size(2);

na = cpu.step_num(3);
a = 1:na;


figure
subplot(1,2,1)
semilogy(r,cpu.refl_r,'k')
hold on
semilogy(r,gpu.refl_r,'r')
semilogy(r,gpu_a.refl_r,'b')

xlabel('r [cm]')
ylabel('Reflectance [cm^{-2}]')

legend(['CPUMCML - ' num2str(CPUtime) ' s (1x)'], ['GPUMCML - ' num2str(GPUtime) ' s (' num2str(round(CPUtime/GPUtime)) 'x)'],['GPUMCML -A - ' num2str(GPU_Atime) ' s (' num2str(round(CPUtime/GPU_Atime)) 'x)'] )

subplot(1,2,2)
plot(a,cpu.refl_a,'k')
hold on
plot(a,gpu.refl_a,'r')
plot(a,gpu_a.refl_a,'b')
xlabel('Angle [rad]')
ylabel('Reflectance [sr^{-1}]')




% 
% %ref = read_file_mco([PathName_ref FileName_ref]);
% %cmp1 = read_file_mco([PathName_cmp1 FileName_cmp1]);
% %cmp2 = read_file_mco([PathName_cmp2 FileName_cmp2]);
% 
% %Make sure the two are comparable
% if sum(ref.step_size ~= cmp1.step_size) ~= 0
%     error('Simulations are not comparable')
% end
% if sum(ref.step_num ~= cmp1.step_num) ~= 0
%     error('Simulations are not comparable')
% end
% if ref.layers ~= cmp1.layers
%     error('Simulations are not comparable')
% end
% if sum(sum(ref.layer ~= cmp1.layer)) ~= 0
%     error('Simulations are not comparable')
% end
% 
% % Create r, z vectors
% z = (1:ref.step_num(1))*ref.step_size(1);
% r = (1:ref.step_num(2))*ref.step_size(2);
% 
% na = ref.step_num(3);
% a = 1:na;
% 
% nlayers = ref.layers;
% 
% % We may compare the two simulations:
% 
% % Scalars
% disp( ['Relative difference in specular reflectance: ' num2str(100*abs(ref.spec_refl-cmp1.spec_refl)/ref.spec_refl) ' %' ]);
% disp( ['Relative difference in diffuse reflectance: ' num2str(100*abs(ref.diff_refl-cmp1.diff_refl)/ref.diff_refl) ' %' ]);
% disp( ['Relative difference in absorbed fraction: ' num2str(100*abs(ref.abs_frac-cmp1.abs_frac)/ref.abs_frac) ' %' ]);
% disp( ['Relative difference in transmitted fraction: ' num2str(100*abs(ref.trans_frac-cmp1.trans_frac)/ref.trans_frac) ' %' ]);
% 
% rel_abs_layer = 100*abs(ref.abs_layer-cmp1.abs_layer)./ref.abs_layer;
% rel_abs_z = 100*abs(ref.abs_z-cmp1.abs_z)./ref.abs_z;
% rel_refl_r = 100*abs(ref.refl_r-cmp1.refl_r)./ref.refl_r;
% rel_refl_a = 100*abs(ref.refl_a-cmp1.refl_a)./ref.refl_a;
% rel_trans_r = 100*abs(ref.trans_r-cmp1.trans_r)./ref.trans_r;
% rel_trans_a = 100*abs(ref.trans_a-cmp1.trans_a)./ref.trans_a;
% rel_abs_rz = 100*abs(ref.abs_rz-cmp1.abs_rz)./ref.abs_rz;
% rel_refl_ra = 100*abs(ref.refl_ra-cmp1.refl_ra)./ref.refl_ra;
% rel_trans_ra = 100*abs(ref.trans_ra-cmp1.trans_ra)./ref.trans_ra;
% rel_layer = 100*abs(ref.layer-cmp1.layer)./ref.layer;
% rel_f_rz = 100*abs(ref.f_rz-cmp1.f_rz)./ref.f_rz;
% 
% if nlayers == 1
%     disp( ['Relative difference in absorption by layer: ' num2str(100*abs(ref.rel_abs_layer-cmp1.rel_abs_layer)/ref.rel_abs_layer) ' %' ]);
% else
%     figure
%     plot(rel_abs_layer);
%     title('Relative difference in absoption by layer')
%     xlabel('Layer [-]')
%     ylabel('Relative difference [%]')
% end
% 
% figure
% plot(z, rel_abs_z);
% title('Relative difference in absoption(z)')
% xlabel('z [cm]')
% ylabel('Relative difference [%]')
% 
% figure
% rel_abs_rz(rel_abs_rz>10)=10;
% imagesc(r,z,rel_abs_rz)
% colorbar
% title('Relative difference in abs(r,z) [%]')
% xlabel('r [cm]')
% ylabel('z [cm]')
% colormap hot; 
% ylim([0 0.8]); 
% 
% figure
% stem(r, rel_refl_r, 'MarkerSize',3);
% title('Relative difference in reflectance(r)')
% xlabel('r [cm]')
% ylabel('Relative difference [%]')
% ylim([0 50]); 
% 
% figure
% stem(r, rel_trans_r, 'MarkerSize',3);
% title('Relative difference in transmittance(r)')
% xlabel('r [cm]')
% ylabel('Relative difference [%]')
% ylim([0 50]); 
% 
% figure
% rel_f_rz(rel_f_rz>10)=10;
% imagesc(r,z,rel_f_rz)
% colorbar
% title('Relative difference in fluence(r,z) [%]')
% xlabel('r [cm]')
% ylabel('z [cm]')
% colormap hot; 
% 
% 
% 
% if na == 1
%     disp( ['Relative difference in refl_a: ' num2str(100*abs(ref.refl_a-cmp1.refl_a)/ref.refl_a) ' %' ]);
%     disp( ['Relative difference in trans_a: ' num2str(100*abs(ref.trans_a-cmp1.trans_a)/ref.trans_a) ' %' ]);
%     
%     figure
%     plot(r, rel_refl_ra);
%     title('Relative difference in reflectance(r,a)')
%     xlabel('r [cm]')
%     ylabel('Relative difference [%]')
% 
%     figure
%     plot(r, rel_trans_ra);
%     title('Relative difference in transmittance(r,a)')
%     xlabel('r [cm]')
%     ylabel('Relative difference [%]')
%     
% else
%     figure
%     plot(a, rel_refl_a);
%     title('Relative difference in reflectance(a)')
%     xlabel('a []')
%     ylabel('Relative difference [%]')
% 
%     figure
%     plot(a, rel_trans_a);
%     title('Relative difference in transmittance(a)')
%     xlabel('a []')
%     ylabel('Relative difference [%]')
% 
%     figure
%     rel_refl_ra(rel_refl_ra>10)=10;
%     imagesc(r,a,rel_refl_ra)
%     colorbar
%     title('Relative difference in reflectance(r,a) [%]')
%     xlabel('r [cm]')
%     ylabel('a []')
% 
%     figure
%     rel_trans_ra(rel_trans_ra>10)=10;
%     imagesc(r,a,rel_trans_ra)
%     colorbar
%     title('Relative difference in transmittance(r,a) [%]')
%     xlabel('r [cm]')
%     ylabel('a []')
%     
% end
% 
% 
% 
% 
% 
