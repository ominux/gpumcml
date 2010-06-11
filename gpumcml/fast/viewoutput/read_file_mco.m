% [s] = read_file_mco;
%
% Reads an output file from MCML into a MatLab structure. The file is
% selected using a GUI.
% 
% The structure contains 18 fields:
% step_size   (Step sizes in z and r)
% step_num    (Number of steps in z, r and a)
% spec_refl   (Specular reflectance at the surface)
% diff_refl   (Reflectance at the surface)
% abs_frac    (Fratcion of the light being absorbed by the sample)
% trans_frac  (Transmittance at the lowest surface)
% abs_layer   (Fraction of the light being absorbed by the different layers)
% abs_z       (Absorption as a function of depth)
% refl_r      (Reflectance at the surface as a function of the radius)
% refl_a      (Reflectance at the surface as a function of the deflection angle)
% trans_r     (Transmittance at the lowest surface as a function of the radius)
% trans_a     (Transmittance at the lowest surface as a function of the deflection angle)
% abs_rz      (Absorption as a function of depth and radius)
% refl_ra     (Reflectance at the surface as a function of the deflection angle and radius)
% trans_ra    (Tranmittance at the surface as a function of the deflection angle and radius)
% layers      (Number of layers used)
% layer       (Properties of the different layers)
% f_rz        (Fluence as a function of depth and radius)
% 
% Descriptions of the fields can be found in the MCML manual. 

function [s]= read_file_mco(varargin)
%clc

%addition by Erik Alerstam 2008-03-26
if nargin==0
    [FileName,PathName] = uigetfile('*.mco','Select MCO-file to open')
    fid=fopen([PathName FileName]);           % �ppnar filen
else
    if nargin==1
        FileName=varargin{1};
        fid=fopen(FileName);           % �ppnar filen
    end
end

%fid=fopen([PathName FileName]);           % �ppnar filen
%disp(FileName);

for i=1:13                      % Tar bort on�diga rader
    tline = fgetl(fid);
end

nphoton= fscanf(fid,'%g');     % [dz dr] 
tline = fgetl(fid);               % Tar bort on�diga rader       
step_size = fscanf(fid,'%g');     % [dz dr] 
tline = fgetl(fid);               % Tar bort on�diga rader            
step_num = fscanf(fid,'%g');      % [dz dr da]
tline = fgetl(fid);               % Tar bort on�diga rader            
tline = fgetl(fid);               % Tar bort on�diga rader            
layers = fscanf(fid,'%g');        % [Antal lager]
tline = fgetl(fid);               % Tar bort on�diga rader            
tline = fgetl(fid);               % Tar bort on�diga rader            
tline = fgetl(fid);               % Tar bort on�diga rader
layer=[];
for i=1:layers
    layer(i,:)=fscanf(fid,'%g');
    tline = fgetl(fid);
end


rad='abc';                       % Letar upp r�tt st�lle att b�rja l�sa data
crit='RAT';
while rad~=crit
    tline = fgetl(fid);     
    if size(tline)==[0 0]
        rad='abc';
    else
        rad=tline(1:3);
    end
end

spec_refl = fscanf(fid,'%g');     % Specular reflectance 
tline = fgetl(fid);               % Tar bort on�diga rader
diff_refl = fscanf(fid,'%g');     % Diffuse reflectance 
tline = fgetl(fid);               % Tar bort on�diga rader
abs_frac = fscanf(fid,'%g');      % Absorbed fraction 
tline = fgetl(fid);               % Tar bort on�diga rader
trans_frac = fscanf(fid,'%g');    % Transmittance


for i=1:3
    tline = fgetl(fid);           % Tar bort on�diga rader
end

abs_layer = fscanf(fid,'%g');      % Absorption as a function of layer
tline = fgetl(fid);               % Tar bort on�diga rader
abs_z = fscanf(fid,'%g');         % Absorption as a function z
tline = fgetl(fid);               % Tar bort on�diga rader
refl_r = fscanf(fid,'%g');        % Reflectance as a function r
tline = fgetl(fid);               % Tar bort on�diga rader
refl_a = fscanf(fid,'%g');        % Reflectance as a function a
tline = fgetl(fid);               % Tar bort on�diga rader
trans_r = fscanf(fid,'%g');       % Transmittance as a function r
tline = fgetl(fid);               % Tar bort on�diga rader
trans_a = fscanf(fid,'%g');       % Transmittance as a function a

for i=1:6
    tline = fgetl(fid);           % Tar bort on�diga rader
end

abs_rz = fscanf(fid,'%g');                                   % Absorption as a function r and z
abs_rz = reshape(abs_rz,step_num(1),step_num(2));            % Stuvar om matrisen

for i=1:6
    tline = fgetl(fid);                                      % Tar bort on�diga rader
end

refl_ra = fscanf(fid,'%g');                                  % Reflectance as a function r and a
refl_ra = reshape(refl_ra,step_num(2),step_num(3));          % Stuvar om matrisen

for i=1:6
    tline = fgetl(fid);                                      % Tar bort on�diga rader
end

trans_ra = fscanf(fid,'%g');                                 % Transmittance as a function r and a
trans_ra = reshape(trans_ra,step_num(2),step_num(3));        % Stuvar om matrisen

% *********** Lagrar allt i en struktur **********************************
s=struct('step_size',{step_size},'step_num',{step_num},'spec_refl',{spec_refl},'diff_refl',{diff_refl},...
    'abs_frac',{abs_frac},'trans_frac',{trans_frac},'abs_layer',{abs_layer},'abs_z',{abs_z},'refl_r',{refl_r},...
    'refl_a',{refl_a},'trans_r',{trans_r},'trans_a',{trans_a},'abs_rz',{abs_rz},'refl_ra',{refl_ra},...
    'trans_ra',{trans_ra},'layers',{layers},'layer',{layer});
% *************************************************************************


maxdepth=sum(s.layer(:,5));
for i=1:s.step_num(1)
    depth=i*s.step_size(1);
    if depth>maxdepth
        abs=0;
    else
        oklar=1;
        layer=1;
        ldepth=s.layer(layer,5);        
        while (oklar==1)
            if depth>ldepth
                layer=layer+1;
                ldepth=ldepth+s.layer(layer,5);
            else
                abs=s.layer(layer,2);
                oklar=0;
            end
        end
    end
    for j=1:s.step_num(2)
        flod(i,j)=s.abs_rz(i,j)/abs;
    end
end

s.f_rz=flod;


status = fclose(fid);