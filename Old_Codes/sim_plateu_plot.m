clf
addpath('cmap')

set(0,'defaulttextInterpreter','latex')
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)

%%%%%%%%%%%%%
% nhstr = 'No_Hydro_Py_Test_phi_0.34' %'New_No_Hydro_MSD';   
% lsty = ':';
%%%%%%%%%%%%%
nhstr = 'Py_Test_phi_0.34' %'New_MSD';
lsty = '-.';
%%%%%%%%%%%%%
% nhstr = 'Py_Test_phi_0.66' %'New_MSD';
% lsty = '-.';
%%%%%%%%%%%%%
% nhstr = 'Py_Test_phi_0.02' %'New_MSD';
% lsty = '-.';
% cc = 1;


plateau_data = dlmread(['./Count_Data_Cpp/' nhstr '_N_stats.txt']);
Ls = round(plateau_data(:,1),3);

cols = parula(3);
cols(3,:) = 0.9*cols(3,:);


errorbar(Ls,2*plateau_data(:,3),plateau_data(:,4),'o','color',cols(cc,:))
hold all

plateau_data = dlmread(['./Count_Data_Cpp/' nhstr '_N_stats_mean_var.txt']);
Ls = round(plateau_data(:,1),3);
errorbar(Ls,2*plateau_data(:,3),plateau_data(:,4),'o','color',cols(cc+1,:))
hold all
%set(gca, 'YScale', 'log', 'XScale', 'log')
