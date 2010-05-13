clear all
close all
clc

CMP = 1;


[FileName_ref,PathName_ref] = uigetfile('*.mco','Select .mco file to plot');
ref = read_file_mco([PathName_ref FileName_ref]);

if CMP
    [FileName_cmp1,PathName_cmp1] = uigetfile('*.mco','Select comparison .mco file to plot');
    cmp1 = read_file_mco([PathName_cmp1 FileName_cmp1]);
end


% Create r, z vectors
z = (0:ref.step_num(1)-1)*ref.step_size(1);
r = (0:ref.step_num(2)-1)*ref.step_size(2);

na = ref.step_num(3);
da=pi/(2*na); 
a = (0:na-1)*da;

nlayers = ref.layers;


figure
subplot(2,2,1); 
semilogy(r,ref.refl_r);hold on
if CMP
    semilogy(r,cmp1.refl_r,'r')
    legend(FileName_ref, FileName_cmp1)
end
%title('Reflectance(r)')
xlabel('r [cm]')
ylabel('Reflectance [1/cm^{2}]')


subplot(2,2,2); 
plot(a,ref.refl_a);hold on
if CMP
    plot(a,cmp1.refl_a,'r')
    legend(FileName_ref, FileName_cmp1)
end
%title('Reflectance(a)')
xlabel('a [rad]')
ylabel('Reflectance [1/sr]')


subplot(2,2,3); 
plot(r,ref.trans_r);hold on
if CMP
    plot(r,cmp1.trans_r,'r')
    legend(FileName_ref, FileName_cmp1)
end
%title('Transmittance(r)')
xlabel('r [cm]')
ylabel('Transmittance [1/cm^{2}]')


subplot(2,2,4); 
plot(a,ref.trans_a);hold on
if CMP
    plot(a,cmp1.trans_a,'r')
    legend(FileName_ref, FileName_cmp1)
end
%title('Transmittance(a)')
xlabel('a [rad]')
ylabel('Transmittance [1/sr]')



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
