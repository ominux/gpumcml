    
clear all;

Nphotons='25M'; 

%% IMPORTANT: Check these parameters before executing the script
nz = 500;
nr = 200;
dr=0.01 %cm
dz=0.002; %cm

% dz=-0.01; %cm

%% IMPORT Sim Outputs
origFileName=['origfastrand/sevenlayer_600nm_10M.mco']; 
origFileName2=['origfastrand/2sevenlayer_600nm_10M.mco']; 
%origFileName2=['orig/2Real_' Nphotons '.mco']; 
tm4FileName=['sevenlayer_600nm_10M.mco']; 


origFID = fopen(origFileName, 'r');  
origFID2 = fopen(origFileName2, 'r');  
compFID = fopen(tm4FileName, 'r');

A = fscanf(origFID, '%e');
A2 = fscanf(origFID2, '%e');
B = fscanf(compFID, '%e');

r=0:dr:(nr-1)*dr; 
z=0:dz:(nz-1)*dz; 

for ir=1:nr
    rVector((ir-1)*nz+1:nz*ir)=r(ir); 
end 

for ir=1:nr
    zVector((ir-1)*nz+1:nz*ir)=z; 
end 

zVector=zVector'; 
rVector=rVector'; 


%% Display Abs Percent Error 2D Color Map 
AbsPercentError=abs(A-B)./A * 100; 
AbsPercentError(isnan(AbsPercentError))= A(isnan(AbsPercentError));

% for i = 1:nr*nz
%     if A(i)==0
%         AbsPercentError(i)=B(i); 
%     else 
%         AbsPercentError(i)=abs(A(i)-B(i))/A(i) * 100; 
%     end
% end
AbsPercentError=AbsPercentError'; 

AbsPercentError_MCML=abs(A2-A)./A2 * 100; 
AbsPercentError_MCML(isnan(AbsPercentError_MCML))=A2(isnan(AbsPercentError_MCML));
% for i = 1:nr*nz
%     if A2(i)==0
%         AbsPercentError_MCML(i)=A(i); 
%     else 
% 
%     end
% end
AbsPercentError_MCML=AbsPercentError_MCML'; 


%figure1 = figure('FileName','B:\ece1373\validation\2Dmaps\1M.pdf','PaperPosition',[1.333 3.313 5.833 4.375]);

figure, subplot(2,1,1); 
scatter(rVector,zVector,10,AbsPercentError,'filled');
title ('(a)', 'FontSize',12, 'FontWeight','bold');
%title (['Absolute Percent Error (TM4 vs. MCML) at ' Nphotons ' photons']); 
xlabel ('Radius (cm)'); 
ylabel ('Depth (cm)'); 
%xlim([0 1]); 
ylim([0 0.8]); 
caxis ([0 10]); 
set(gca,'YDir','reverse')
colormap hot; 

colorbar; 

subplot(2,1,2); 
scatter(rVector,zVector,10,AbsPercentError_MCML,'filled');
title ('(b)', 'FontSize',12, 'FontWeight','bold');
%title (['Absolute Percent Error (MCML vs. MCML) at ' Nphotons ' photons']); 
xlabel ('Radius (cm)'); 
ylabel ('Depth (cm)'); 
%xlim([0 1]); 
ylim([0 0.8]); 
caxis ([0 10]); 
set(gca,'YDir','reverse')
colormap hot;
% colormap(jet(10));
% hcb = colorbar('YTickLabel',...
% {'0%','2%','4%','6%','8%','10%'});
% set(hcb,'YTickMode','manual')
colorbar; 