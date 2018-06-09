clear all;
close all;

addpath('~/MATLAB/utilities/');

file = 'example_sheet2_prb1_data.txt';

fid = fopen(file);

table = textscan(fid,'%f %f %f %f','CommentStyle','#');

xs = table{1};
xerrs = table{2};
ys = table{3};
yerrs = table{4};

N = length(xs);

% figure(1)
% errorbar_xy2(xs,ys,xerrs,yerrs,'MarkerSize',20)
% 
% set(gca,'FontSize',18)
% 
% xlabel('Measured x','FontSize',16)
% ylabel('Measured y','FontSize',16)
% title(['Data : N = ' num2str(N)])


%%

n_mc = 1e5;

logposterior = @(pp) logL_regress_gx1(ys,yerrs,xs,xerrs,pp) ; 

mc = zeros(n_mc,5);

% jumping scales
jumps = [3; 1; 0.3; 0.3; 0.5] / 10;

acc = 0;

theta = [3; 1.1; 0.5; -1; 0.75] + 30*jumps.*rand(5,1);

logpost_curr = logposterior(theta);

disp('Begin MCMC: ')
disp(' ')
tic

for i=1:n_mc
    
       if(mod(i,n_mc/100)==0)
           disp(['mcmc step i = ' num2str(i,'%.0f') ' : acc = ' num2str(acc/i,'%.2f') ' : logpost = ' num2str(logpost_curr)]);
       end
    
       theta_prop = theta + jumps.* randn(5,1);
       
       logpost_prop = logposterior(theta_prop);
       
       logr = logpost_prop-logpost_curr;
       
       if log(rand) < logr
           theta = theta_prop;
           acc = acc+1;
           logpost_curr = logpost_prop;
       else
           % theta stays the same;
           % logpost_curr stays the same;
       end
       
       mc(i,:) = theta;
       
end

acc = acc/n_mc;
runtime = toc;

disp(' ')
disp(['End MCMC: Accept rate = ' num2str(acc,'%.2f')])
disp(['Runtime = ' num2str(runtime,'%.2f')])

% examine chains

%% load saved chain

load('MCMC_5e6_all4.mat','mc')

%% compute G-R ratio


% plot multiple chains
gr = mcmc_calcrhat(mc)

max_gr = max(gr)

%%

thin = 1000;
burn = n_mc/5;

mc = mc(burn:thin:end,:,:);

mc = mcmc_combine(mc,0);

%%

% figure(6)
% plot(mc(:,2));
% ylabel('\beta')

figure(7)
histogram(mc(:,2),'Normalization','pdf')
xlabel('\beta');
ylabel('Posterior pdf P(\beta | D)')

post_means = mean(mc)'
post_stds = std(mc)'

%% 1D KDE

p1 = 2;

[fi,xi] = ksdensity(mc(:,2));
hold on
plot(xi,fi,'-k','LineWidth',3)

hold off


%% 2D KDE

% 2D scatter plots

p1 = 2;
p2 = 5;

p1_bnd = quantile(mc(:,p1),[0.0001,0.999]);

p2_bnd = quantile(mc(:,p2),[0.0001,0.999]);

figure(1000)
[contoursout, themode, X, Y, density] = hpd_smoothed_contours_bnds([mc(:,p1), mc(:,p2)],p1_bnd,p2_bnd,1,32);
close(1000)

figure(10)
contour(X,Y,density,0.2:0.1:0.9,'LineWidth',1);

hold on
plot(themode(1),themode(2),'xk','MarkerSize',15,'LineWidth',2)

plot(contoursout{1}(1,:),contoursout{1}(2,:),'-k','LineWidth',2)
plot(contoursout{2}(1,:),contoursout{2}(2,:),'-k','LineWidth',2)

title('2D Marginal Posterior')
xlabel('\beta')
%ylabel('\tau')
hold off
