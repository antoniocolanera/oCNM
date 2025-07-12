%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% main_ocnm_demo.m
%
%   Demo script for standard CNM and orbital CNM (oCNM).
%   A modulated 2D signal is used as test case to:
%     - Compare performance between CNM and STFT-based oCNM
%     - Visualize reconstructions in time and frequency domains
%     - Inspect transition matrices Q_ij
%     - Compare temporal correlations
%
%   Reference:
%   A. Colanera, N. Deng, M. Chiatto, L. de Luca, B. R. Noack,
%   "Orbital cluster-based network modelling", COMPHY, 2025.
%
%   Author: Antonio Colanera
%   Created: July 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; close all; clear;

%% ---------------------- Generate test signal ----------------------------
nt = 200000;
t = linspace(0, 100, nt);
dt = t(2) - t(1);
fs = 1 / dt;

% Define amplitude- and phase-modulated signal (2D)
amp = cos(2*pi*2*t);                % Amplitude modulation
modu = 75*pi*cos(2*pi*0.25*t);      % Phase modulation
x = amp .* sin(2*pi*100*t + modu);
y = amp .* cos(2*pi*100*t + modu);
X = [x; y];

% Plot raw signal
figure('units','centimeters','Position',[5 5 14 4]);
plot(t(t<=2), x(t<=2), 'k', 'LineWidth', 0.8);
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$b_1$', 'Interpreter', 'latex');
set(gca,'ticklabelinterpreter','latex');

figure('units','centimeters','Position',[5 5 4 4]);
plot(y(t<=0.5), x(t<=0.5), 'k', 'LineWidth', 0.8);
xlabel('$b_2$', 'Interpreter', 'latex');
ylabel('$b_1$', 'Interpreter', 'latex');
set(gca,'ticklabelinterpreter','latex');

%% ----------------- Standard CNM in physical space -----------------------
K = 16;           % Number of clusters
L = 1;            % Markov model order
rng(1)
cnmSTANDARD = CNM(X, dt, K, L);
Xrec = cnmSTANDARD.predict(X(:,1), t(end), dt);

%% ------------------- Orbital CNM with STFT embedding --------------------
nseg = 1000;
novl = ceil(0.50 * nseg);
rng(1)
ocnm = oCNM(X, t, nseg, novl, K, L, 'linear');
Xrec_ocnm = ocnm.predict();

% Compute STFT of CNM-reconstructed signal for comparison
win = hamming(nseg, 'periodic');
[Xstftrec, fvecrec, tvecrec] = stft(Xrec', fs, 'Window', win, ...
    'OverlapLength', novl, 'FrequencyRange', 'onesided');

%% ---------------------- Time-domain reconstruction ---------------------
figure('units','centimeters','Position',[5 5 15 4]);
plot(t, X(1,:), 'k', 'linewidth', 1); hold on
plot(cnmSTANDARD.t, Xrec(1,:), 'b', 'linewidth', 1);
plot(ocnm.TIME, Xrec_ocnm(1,:), 'r', 'linewidth', 1);
xlim([0 1]); ylim([-1 1])
xlabel('$t$', 'Interpreter', 'latex');
ylabel('$b_1$', 'Interpreter', 'latex');
legend({'Data', 'CNM ($K=16$)', 'STFT-CNM ($K=16$)'}, ...
    'Interpreter','latex','Location','northoutside','Orientation','horizontal','Box','off');
set(gca,'ticklabelinterpreter','latex','looseinset',[0,0,0.05,0]);
box on;

%% ------------------- Time-frequency contour plots -----------------------
figure('units','centimeters','Position',[5 5 15 4.5]);
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize',11);
hold on;
contourf(ocnm.tvec,ocnm.fvec,abs(ocnm.Xstft_ocnm(:,:,1))/max(max(abs(ocnm.Xstft_ocnm(:,:,1))')),...
    linspace(-1,1,100) ...
    ,'LineColor','none');
xlabel('$t$','interpreter','latex');
ylabel('$f$','interpreter','latex');
set(gca,'ticklabelinterpreter','latex')
% text(5,200,['STFT-CNM ($K=' num2str(K) '$, $L=' num2str(L) '$)'],'color','k','interpreter','latex')
text(2,200,['STFT-oCNM ($K=' num2str(K) '$)'],'color','k','interpreter','latex','EdgeColor','k')

set(gca,'ylim',[0 220])
set(gca,'xlim',[0 40])
% colormap(fireicemy(256));
colormap(redblueTecplot(256));
caxis([-1 1]);
box on;

figure('units','centimeters','Position',[5 5 15 4.5]);
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize',11);
hold on;
contourf(tvecrec,fvecrec,abs(Xstftrec(:,:,1))/max(max(abs(Xstftrec(:,:,1))')),...
    linspace(-1,1,100) ...
    ,'LineColor','none');
xlabel('$t$','interpreter','latex');
ylabel('$f$','interpreter','latex');
set(gca,'ticklabelinterpreter','latex')
% text(5,200,['STFT-CNM ($K=' num2str(K) '$, $L=' num2str(L) '$)'],'color','k','interpreter','latex')
text(2,200,['CNM ($K=' num2str(K) '$)'],'color','k','interpreter','latex','EdgeColor','k')

set(gca,'ylim',[0 220])
set(gca,'xlim',[0 40])
% colormap(fireicemy(256));
colormap(redblueTecplot(256));
caxis([-1 1]);
box on;

figure('units','centimeters','Position',[5 5 15 4.5]);
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize',11);
hold on;box on,
contourf(ocnm.tvec,ocnm.fvec,abs(ocnm.Xstft_original(:,:,1))/max(max(abs(ocnm.Xstft_original(:,:,1))')),...
    linspace(-1,1,100) ...
    ,'LineColor','none');
xlabel('$t$','interpreter','latex');
ylabel('$f$','interpreter','latex');
set(gca,'ticklabelinterpreter','latex')
text(2,200,'Original data','color','k','interpreter','latex','EdgeColor','k')
set(gca,'ylim',[0 220])
set(gca,'xlim',[0 40])
% colormap(fireicemy(256));
colormap(redblueTecplot(256));
caxis([-1 1]);
%% ------------------ Cluster spectral structure --------------------------
nf=ocnm.nf

figure('units','centimeters','Position',[5 5 18 10]);
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize',11);
hold on;box on
cluamp=ocnm.cnm.clusters(:,1:nf).^2+ocnm.cnm.clusters(:,nf+1:2*nf).^2+ocnm.cnm.clusters(:,2*nf+1:3*nf).^2+ocnm.cnm.clusters(:,3*nf+1:4*nf).^2;

cluamp=cluamp./max(cluamp,[],2);

CMAPS=colormap('turbo');

for ii=1:K
    plot3(ii*ones(size(ocnm.fvec)),ocnm.fvec,cluamp(ii,:),'linewidth',1 ,'Color',CMAPS(ceil(ii/K*256),:))
end
xlabel('Cluster','interpreter','latex');

ylabel('$f$','interpreter','latex');
zlabel('$\mathrm{PSD}(\tilde \mathbf{c}_i)$','interpreter','latex');
set(gca,'ticklabelinterpreter','latex')
set(gca,'xtick',[1:20])
set(gca,'ylim',[0 200])
set(gca,'xlim',[0 K])
set(gca,'zlim',[0 1.2])

caxis([-1 1]);
view([-45 45])

%% ------------------ Transition matrices Qij ----------------------------

figure('units','centimeters','Position',[5 5 13 11]);
    % matrix
     imagesc(ocnm.cnm.Qij);
colormap(redblueTecplot(256));

caxis([-1 1])

 cc=colorbar('location','eastoutside','FontSize',10);
    cc.Limits=[0 1];
    %title(cc,'$t$','interpreter','latex');
    hold on

    % lines
    for irow = 1:K+1
        plot([irow-0.5,irow-0.5], [0.5,K+0.5], '-k', 'LineWidth',0.3) % horizontal up
        plot([0.5,K+0.5], [irow-0.5,irow-0.5], '-k', 'LineWidth',0.3) % horizontal below
        hold on
    end
    % Border
    plot([0.5,0.5], [0.5,K+0.5], '-k', 'LineWidth',1.5) % vertical left
    hold on
    plot([K+0.5,K+0.5], [0.5,K+0.5], '-k', 'LineWidth',1.5) % vertical right
    hold on
    plot([0.5,K+0.5], [0.5,0.5], '-k', 'LineWidth',1.5) % horizontal up
    hold on
    plot([0.5,K+0.5], [K+0.5,K+0.5], '-k', 'LineWidth',1.5) % horizontal down
    axis equal
    title('$Q_{ij}$ oCNM', 'interpreter','latex')
    set(gca, 'XLim',[0.5, K+0.5], 'YLim',[0.5, K+0.5]); 
    tickVector = 1:K;
    set(gca, 'XTick',tickVector, 'YTick',tickVector);

    set(gca,'linewidth',0.5,'fontsize',10,'fontname','Times','TickDir','none');
%     set(gca, 'looseinset', [0,0,0,0])

    xlabel('$j$', 'interpreter','latex')
    ylabel('$i$', 'interpreter','latex')


  figure('units','centimeters','Position',[5 5 13 11]);
    % matrix
     imagesc(cnmSTANDARD.Qij);
colormap(redblueTecplot(256));

caxis([-1 1])

 cc=colorbar('location','eastoutside','FontSize',10);
    cc.Limits=[0 1];
    hold on

    % lines
    for irow = 1:K+1
        plot([irow-0.5,irow-0.5], [0.5,K+0.5], '-k', 'LineWidth',0.3) % horizontal up
        plot([0.5,K+0.5], [irow-0.5,irow-0.5], '-k', 'LineWidth',0.3) % horizontal below
        hold on
    end
    % Border
    plot([0.5,0.5], [0.5,K+0.5], '-k', 'LineWidth',1.5) % vertical left
    hold on
    plot([K+0.5,K+0.5], [0.5,K+0.5], '-k', 'LineWidth',1.5) % vertical right
    hold on
    plot([0.5,K+0.5], [0.5,0.5], '-k', 'LineWidth',1.5) % horizontal up
    hold on
    plot([0.5,K+0.5], [K+0.5,K+0.5], '-k', 'LineWidth',1.5) % horizontal down
    axis equal
    title('$Q_{ij}$ CNM', 'interpreter','latex')
    set(gca, 'XLim',[0.5, K+0.5], 'YLim',[0.5, K+0.5]); 
    tickVector = 1:K;
    set(gca, 'XTick',tickVector, 'YTick',tickVector);

    set(gca,'linewidth',0.5,'fontsize',10,'fontname','Times','TickDir','none');
%     set(gca, 'looseinset', [0,0,0,0])

    xlabel('$j$', 'interpreter','latex')
    ylabel('$i$', 'interpreter','latex')
%% -------------------- Temporal correlation analysis ---------------------
Tcorr=1;
R =    autocorrelation(t,X',Tcorr,'fft');
R=R./(fliplr(0:(length(R)-1) )')*(length(R)-1);
R=R/R(1);
RCNM = autocorrelation(cnmSTANDARD.t,Xrec',Tcorr,'fft');
RCNM=RCNM./(fliplr(0:(length(RCNM)-1) )')*(length(RCNM)-1);
RCNM=RCNM/RCNM(1);
RSTFTCNM = autocorrelation(ocnm.TIME,Xrec_ocnm',Tcorr,'fft');
RSTFTCNM=RSTFTCNM./(fliplr(0:(length(RSTFTCNM)-1) )')*(length(RSTFTCNM)-1);
RSTFTCNM=RSTFTCNM/RSTFTCNM(1);
%

figure('units','centimeters','Position',[5 5 15 6]);
tcl = tiledlayout(1,2);
nexttile
plot(t,R,'k','linewidth',1);hold on;
plot(t,RCNM,'b','linewidth',1);
plot(t,RSTFTCNM,'r--','linewidth',1)
ylim([0 1.1])
xlim([0 15 ])
xlabel('$\tau$','interpreter','latex');
ylabel('$R/R(0)$','interpreter','latex');
set(gca,'ticklabelinterpreter','latex');
box on;hold on;%grid on

nexttile
plot(t,R,'k','linewidth',1);hold on;
plot(t,RCNM,'b','linewidth',1);
plot(t,RSTFTCNM,'r--','linewidth',1)
%ylim([0 1.1])
xlim([0 0.1 ])
xlabel('$\tau$','interpreter','latex');
ylabel('$R/R(0)$','interpreter','latex');
set(gca,'ticklabelinterpreter','latex','XTick',0:0.02:0.1);
box on;hold on;%grid on
tcl.TileSpacing='compact'

hL=legend({'Data', ['CNM ($K='  num2str(K) '$)'], ...
    ['STFT-oCNM ($K='  num2str(K) '$)']},'interpreter','latex' ...
    ,'Orientation','horizontal','box','off')

hL.Layout.Tile = 'north';
