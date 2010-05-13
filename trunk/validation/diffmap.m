clear all
close all
clc

[FileName_ref,PathName_ref] = uigetfile('*.mco','Select reference .mco file');
[FileName_cmp1,PathName_cmp1] = uigetfile('*.mco','Select comparison .mco file1');
[FileName_cmp2,PathName_cmp2] = uigetfile('*.mco','Select comparison .mco file2');

ref = read_file_mco([PathName_ref FileName_ref]);
cmp1 = read_file_mco([PathName_cmp1 FileName_cmp1]);
cmp2 = read_file_mco([PathName_cmp2 FileName_cmp2]);

%Make sure the two are comparable
if sum(ref.step_size ~= cmp1.step_size) ~= 0
    error('Simulations are not comparable')
end
if sum(ref.step_num ~= cmp1.step_num) ~= 0
    error('Simulations are not comparable')
end
if ref.layers ~= cmp1.layers
    error('Simulations are not comparable')
end
if sum(sum(ref.layer ~= cmp1.layer)) ~= 0
    error('Simulations are not comparable')
end

% Create r, z vectors
z = (1:ref.step_num(1))*ref.step_size(1);
r = (1:ref.step_num(2))*ref.step_size(2);

na = ref.step_num(3);
a = 1:na;
nlayers = ref.layers;

% We may compare the two simulations:
rel_abs_rz1 = 100*abs(ref.abs_rz-cmp1.abs_rz)./ref.abs_rz;
rel_abs_rz2= 100*abs(ref.abs_rz-cmp2.abs_rz)./ref.abs_rz;

figure
subplot(2,1,1); 
rel_abs_rz1(rel_abs_rz1>10)=10;
imagesc(r,z,rel_abs_rz1)
colorbar
xlabel('r [cm]')
ylabel('z [cm]')
title ('(a)','FontWeight','bold'); 
colormap hot; 
ylim([0 0.8]); 

subplot(2,1,2); 
rel_abs_rz2(rel_abs_rz2>10)=10;
imagesc(r,z,rel_abs_rz2)
colorbar
xlabel('r [cm]')
ylabel('z [cm]')
title ('(b)','FontWeight','bold'); 
colormap hot; 
ylim([0 0.8]); 
