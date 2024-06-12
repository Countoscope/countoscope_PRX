clf
clear all

set(0,'defaulttextInterpreter','latex')
set(0, 'defaultAxesTickLabelInterpreter','latex'); 
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)

%%%%%%%%%%%%%
nhstr = 'Py_Test_phi_0.1';
lsty = '-.';
dtBig = 0.5;
%%%%%%%%%%%%%


plateau_data = dlmread(['./Count_Data_Cpp/' nhstr '_N_stats.txt']);
Ls = round(plateau_data(:,1),6);
cols = (bone(length(Ls)+3));


for kk = 1:length(Ls)
    L_0 = Ls(kk);
    plateau = 2*plateau_data(kk,3); 
    
    MSD = dlmread(['./Count_Data_Cpp/' nhstr '_MSDmean_BoxL_' num2str(L_0,'%.6f') '.txt']);
    simTime = dtBig*(1:length(MSD)-1);
    simNC = MSD(2:end);
    simErr = dlmread(['./Count_Data_Cpp/' nhstr '_MSDerror_BoxL_' num2str(L_0,'%.6f') '.txt']);
    simErr(1) = [];

    xt = simTime;
    yt = (1-simNC./plateau);
    ep = (1-(simNC+simErr)./plateau);
    em = (1-(simNC-simErr)./plateau);


    
    ylim([1e-7 1.2])
    
    
    %%%%%%% find points to use to fit tail (method 1)
    %%% look at the errbar ratio to get a scaled measume of uncertianty
    thresh = 0.15;
    ferr = abs(yt-ep)./abs(yt); %(abs(yt-em)./abs(yt-ep))-1;
    idxf = find(ferr > thresh,1);
    if isempty(idxf)
        idxf = round(0.9*length(yt));
    end
    idxs  = idxf-round(0.5*idxf);

    %%%%%%% find points to use to fit tail (method 2)
    %%% find the peak in the (smoothed) log derivative 
    % ye = [yt(1) yt'];
    % dx = xt(2)-xt(1);
    % yp = diff(ye)./dx;
    % logdir = yp./yt';
    % smoothed_log_logdir = medfilt1(log(abs(logdir)),500);
    % %[~,idxf] = min(smoothed_log_logdir);
    % [pks,locs,~] = findpeaks( - smoothed_log_logdir, 'MinPeakDistance', length(yt)/5);
    % idxf = locs(1);
    % idxs  = idxf-round(0.5*idxf);
    % idxs = max(idxs,1);
    
    %disp(num2str(yt(idxf)))
    tol = sqrt(1e-5);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    h1 = plot(xt, yt.^2, lsty, 'linewidth',5,'color',[cols(kk,:)],'MarkerFaceColor',[0.8*cols(kk,:)],'markersize',5);
    hold all
    eyu = ep(1:end-1)';
    eyd = em(1:end-1)';
    %%%%%%%%%%%%
    %%% plot errorbar (slower plotting)
    % hfil = fill([xt(1:end-1) flip(xt(1:end-1))],[eyu.^2 flip(eyd.^2)],cols(kk,:));
    % set(hfil,'facealpha',0.2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if((yt(idxf) > sqrt(1e-4)) && yt(1) > sqrt(0.1))
        
        %Const = polyfit(log(xt(idxs:idxf)),log(yt(idxs:idxf)), 2);
        %clin = polyfit(log(xt(idxf-10:idxf)),log(yt(idxf-10:idxf)), 1);
        x0 = [-100,-1]; %[clin(end)]; %
        fitfun = fittype('(b*x + a) - log(1 + exp(b*x+a))','dependent',{'y'},'independent',{'x'},...
        'coefficients',{'a','b'});
        fxt = log(xt(idxs:idxf)'); %
        fyt = log(yt(idxs:idxf)); %idxs


        [fitted_curve,gof] = fit(fxt,fyt,fitfun,'StartPoint',x0);
        coeffvals_lt = coeffvalues(fitted_curve);
        %%%%%%%%%%%%%%%%%%%%%%%%
        
        
        Xfit = xt(1:idxs);
        dx = Xfit(2)-Xfit(1);
        Yfit = yt(1:idxs)';
        spacing = dx;
        while Yfit(end)> tol
            Xnew = Xfit(end)+spacing;
            Ynew = exp(fitted_curve(log(Xnew))); %fitted_curve(Xnew); %exp(polyval(Const,log(Xnew)));
            Xfit = [Xfit Xnew];
            Yfit = [Yfit Ynew];
            reldiff = 2.0*abs(Yfit(end) - Yfit(end-1))/abs(Yfit(end) + Yfit(end-1));
            if reldiff < 1e-2
                spacing = 2*spacing;
            else
                spacing = 0.5*spacing;
            end
            if Xnew > 1e6
                break
            end
        end
        hold all
        h2 = plot(Xfit,Yfit.^2,'-','color',[cols(kk,:) 0.5],'linewidth',4);
        hold all
        h3 = plot(xt(idxs:idxf),yt(idxs:idxf).^2,'-','color','r','linewidth',7);
    else
        idxf = find(yt < tol,1);
        Xfit = xt(1:idxf);
        Yfit = yt(1:idxf)';
        xt(idxf+1:end) = [];
        yt(idxf+1:end) = [];
        h2 = plot(Xfit,Yfit.^2,'-','color',[cols(kk,:) 0.5],'linewidth',4);
        hold all
    end
    
   
    
    %%%%%%%%% fit start
    x0 = [0.0444/(4*L_0^2)]; %0.0444/(4*L_0^2)
    fitfun = fittype('(sqrt(x/(abs(a)*pi)).*(exp(-abs(a)./x)-1) + erf(sqrt(abs(a/x))) ).^2','dependent',{'y'},'independent',{'x'}, 'coefficients',{'a'});

    fxt = [1e-8 xt(1:1)];
    fyt = [1 yt(1:1)'];
    [fitted_curve,gof] = fit(fxt',fyt',fitfun,'StartPoint',x0);
    coeffvals = coeffvalues(fitted_curve);
    
    xsmall = logspace(-8,log10(dtBig),5000); %[1e-8:1e-3:0.5];
    xsmall(end) = [];
    ysmall = fitted_curve(xsmall);
    hold all
    h4 = plot(xsmall,ysmall.^2,':','color',cols(kk,:),'linewidth',2);
    %%%%%%%%%%%%%%%%%%%
    
    
    Tscale_fit = 2*trapz(Xfit,Yfit.^2)+2*trapz(xsmall,ysmall.^2);
    Tscale_data = 2*trapz(xt,yt.^2);
    rel_err = abs((Tscale_fit-Tscale_data)/(0.5*(Tscale_fit+Tscale_data)));
    disp(num2str([Tscale_fit Tscale_data rel_err]))
    
    Small_int(kk) = trapz(xsmall,ysmall);
    Tscale(kk) = Tscale_fit;
    Tscale_d(kk) = Tscale_data;
    set(gca, 'YScale', 'log', 'XScale', 'log')
    if(kk==8)
        leg = legend([h1,h2,h3,h4],'Data','long time fit','long time fit region','short time fit');
        set(leg,'AutoUpdate','off','fontsize',30,'location','southwest')
    end

end
xlim([1e-3 1e5])
xticks(10.^(-3:1:5))
yticks(10.^(-7:1:0))
grid on
