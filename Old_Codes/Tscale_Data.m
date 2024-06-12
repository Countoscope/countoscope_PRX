phi_0p34 = [1.5875e+03
   2.6731e+02
   7.8134e+01
   2.7873e+01
   1.2326e+01
   7.8194e+00
   3.5074e+00
   2.0335e+00
   1.2733e+00
   1.9139e+00];

phi_0p34_no_fit = [1.3328e+03
   3.3891e+02
   8.0062e+01
   2.7016e+01
   1.1527e+01
   7.0442e+00
   2.8540e+00
   1.6022e+00
   1.0871e+00
   1.8255e+00];

NH_phi_0p34 = [1.4739e+03
   3.5292e+02
   9.5047e+01
   2.6783e+01
   1.0568e+01
   7.0409e+00
   2.9224e+00
   9.9445e-01
   4.9426e-01
   3.1404e-01];

NH_phi_0p34_no_fit = [1.8950e+03
   4.1810e+02
   1.0406e+02
   2.8134e+01
   9.7819e+00
   6.2895e+00
   2.3002e+00
   6.0685e-01
   3.3779e-01
   2.2997e-01];

phi_0p66 = [NaN
   3.6336e+01
   1.6778e+01
   9.3859e+00
   1.1313e+01
   1.1986e+01
   8.1096e+00
   2.2596e+00
   5.1246e-01
   1.6586e-01];

phi_0p66_no_fit = [NaN
   4.1565e+01
   2.4861e+01
   9.0289e+00
   1.1009e+01
   1.1497e+01
   7.4276e+00
   1.8438e+00
   2.6648e-01
   5.6720e-02];

phi_0p02 = [NaN
   4.7393e+03
   8.7004e+02
   1.7210e+02
   4.7860e+01
   1.2398e+01
   3.2183e+00
   8.4547e-01
   9.1740e-01
   1.7216e+00];

phi_0p02_no_fit = [2.0954e+03
   1.3413e+03
   6.0154e+02
   1.6933e+02
   4.6874e+01
   1.1719e+01
   2.7040e+00
   7.4859e-01
   7.6486e-01
   1.6330e+00];

Ls = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125];
D0 = 0.0444;

phi=0.66;
D0star66 = D0*(1+phi)/((1-phi)^3);
sigma = 2*1.395;
phi_scale = (Ls.^2./(4*D0))';

cols = parula(3);
cols(3,:) = 0.9*cols(3,:);
plot(Ls./sigma,phi_0p02./phi_scale,'-o','color',cols(1,:))
hold all
plot(Ls./sigma,phi_0p02_no_fit./phi_scale,'--s','color',cols(1,:))
hold all
plot(Ls./sigma,phi_0p34./phi_scale,'-o','color',cols(2,:))
hold all
plot(Ls./sigma,phi_0p34_no_fit./phi_scale,'--s','color',cols(2,:))
hold all
plot(Ls./sigma,NH_phi_0p34./phi_scale,'-+','color',0.6*cols(2,:))
hold all
plot(Ls./sigma,phi_0p66./phi_scale,'-o','color',cols(3,:))
hold all
plot(Ls./sigma,phi_0p66_no_fit./phi_scale,'--s','color',cols(3,:))
hold all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alphat = 0.56;
betat = 4*(4-pi)/16; % the factor 4 here is bullshit
plot(Ls./sigma,alphat*phi_scale./phi_scale,'--','color','k')
hold all
plot(Ls./sigma,betat*(sigma^2/(4*D0))./phi_scale,':','color','m')
hold all
plot(Ls./sigma,alphat*(Ls.^2./(4*D0star66))./phi_scale',':','color',cols(3,:))
set(gca, 'YScale', 'log', 'XScale', 'log')
grid on
legend('$$\phi=0.02$$','$$\phi=0.02$$ no fit',...
       '$$\phi=0.34$$','$$\phi=0.34$$ no fit',...
       '$$\phi=0.34$$ no hydro',...
       '$$\phi=0.66$$','$$\phi=0.66$$ no fit',...
       '$$T \sim L^2/4D_0$$',...
       '$$T \sim \sigma^2/4D_0$$',...
       '$$T \sim L^2/4D_0^{*}$$')