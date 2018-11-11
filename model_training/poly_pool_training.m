clear all;
% result = dlmread('cnn_mnist_results.txt');
% input_feature = dlmread('cnn_mnist_features.txt');
% data = dlmread('fc_random.txt');
% data = dlmread('fc_famous_other.txt');
data_raw = dlmread('res_pool.txt');
% data_raw = dlmread('fc_famous.txt');
% data = [];
data = data_raw;
threshold = 2;
% for i = 1:size(data_raw, 1)
%     if data_raw(i, 4) > threshold
%        data = [data; data_raw(i, :)];
%     end
% end

% data = dlmread('fc_famous_first.txt');
% data = dlmread('fc_random_fc1.txt');
% data = dlmread('fc_random_fc_others.txt');
% data = dlmread('conv_data.txt');
result = data(:,end - 1: end);
ipt_all = data(:,1: end - 2);

[row, col] = size(ipt_all);

X_runtime = [];
for i = 1:row
    ipf = (ipt_all(i, :));
    % linear
%     tmp1 = [ipf, log2(ipf)];
    tmp1 = [ipf(1:5),ipf(9)];
%     tmp1 = [ipf];
    % quadratic 
    tmp2 = [];
    for j  = 1:length(tmp1)
        for k = j: length(tmp1)
            tmp2 = [tmp2, tmp1(j)*tmp1(k)];
        end
    end
    % cubic
    tmp3 = [];
    for j  = 1:length(tmp1)
        for k = j: length(tmp1)
            for l = k: length(tmp1)
                tmp3 = [tmp3, tmp1(j)*tmp1(k)*tmp1(l)];
            end
        end
    end     
%     tmp = [];
%     tmp = [tmp1, tmp2];
    tmp = [tmp1, tmp2, tmp3];
% 
    tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)*ipf(5)*ipf(6)]; %operations data
    tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)]; %output data
    tmp = [tmp, ipf(1)*ipf(9)*ipf(10)*ipf(4)]; %input data


    X_runtime = [X_runtime; tmp];
end

X_power = [];
for i = 1:row
    ipf = (ipt_all(i, :));
    % linear
%     tmp1 = [ipf, log(ipf)];
    tmp1 = [ipf(1:5),ipf(9)];
    tmp11 = [tmp1, log2(tmp1)];
    % quadratic 
    tmp2 = [];
    for j  = 1:length(tmp11)
        for k = j: length(tmp11)
            tmp2 = [tmp2, tmp11(j)*tmp11(k)];
        end
    end
    % cubic
    tmp3 = [];
    for j  = 1:length(tmp1)
        for k = j: length(tmp1)
            for l = k: length(tmp1)
                tmp3 = [tmp3, tmp1(j)*tmp1(k)*tmp1(l)];
            end
        end
    end     
%     tmp = [];
    tmp = [tmp11, tmp2];
% 
    tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)*ipf(5)*ipf(6)]; %operations data
    tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)]; %output data
    tmp = [tmp, ipf(1)*ipf(9)*ipf(10)*ipf(4)]; %input data


    X_power = [X_power; tmp];
end

% %% add log & exponential terms
% for i = 1:size(X,2)
%     tmp = X(:,i);
%     X = [X, log(tmp)];
% %     X = [X, log(tmp + 1), exp(tmp)];
% end

% %% normalize
% X_norm = [];
% Y_norm = [];
% for i = 1:size(X,2)
%     tmp = X(:,i);
%     X_norm = [X_norm, tmp/max(tmp)];
% %     if (max(tmp) == min(tmp))
% %         X_norm = [X_norm, tmp];
% %     else
% %         X_norm = [X_norm, (tmp - min(tmp))/(max(tmp) - min(tmp))];
% %     end
% end
% X = X_norm;

runtime = result(:,1); % runtime
power = result(:,2);
rmse = [];
% % X = dlmread('cnn_mnist_features.txt');
% for i = 1:1
% [p,S] = polyfit(X,y,i);
% 
% y_poly = polyval(p,X);
% % rmse = [rmse, sqrt(mean(((y_poly - y)) .^ 2))];
%  rmse = [rmse, sqrt(mean(((y_poly - y)./y) .^ 2))];
% end

%% lasso
y1 = runtime;
[B1,FitInfo1] = lasso(X_runtime, y1, 'CV', 10);
fprintf('Runtime model complexity: %d\n', sum(B1(:,FitInfo1.IndexMinMSE) ~= 0) + 1)

y_runtime = X_runtime * B1(:,FitInfo1.IndexMinMSE) + FitInfo1.Intercept(FitInfo1.IndexMinMSE);
mspe_runtime = sqrt(mean(((y_runtime - y1)./y1) .^ 2));
  mse_runtime = sqrt(mean(((y_runtime - y1)) .^ 2));
%  mse_runtime = sqrt(mean(((exp(y_runtime) - exp(y1))) .^ 2));
fprintf('%.4f, %.4f\n', mspe_runtime, mse_runtime);

y2 = power;
[B2,FitInfo2] = lasso(X_power, y2, 'CV', 10);
fprintf('Power model complexity: %d\n', sum(B2(:,FitInfo2.IndexMinMSE) ~= 0) + 1)

y_power = X_power * B2(:,FitInfo2.IndexMinMSE) + FitInfo2.Intercept(FitInfo2.IndexMinMSE);
mspe_power = sqrt(mean(((y_power - y2)./y2) .^ 2));
	mse_power = sqrt(mean(((y_power - y2)) .^ 2));
fprintf('%.4f, %.4f\n', mspe_power, mse_power);
% 
% save('pool_regre.mat');
coeffi_runtime =  [B1(:,FitInfo1.IndexMinMSE)', FitInfo1.Intercept(FitInfo1.IndexMinMSE)];
coeffi_power =  [B2(:,FitInfo2.IndexMinMSE)', FitInfo2.Intercept(FitInfo2.IndexMinMSE)];
dlmwrite('coeff_pool.txt',coeffi_runtime,'delimiter',',');
dlmwrite('coeff_pool.txt',coeffi_power,'delimiter',',','-append');


% coeffi_runtime =  [B1(:,FitInfo1.IndexMinMSE)', FitInfo1.Intercept(FitInfo1.IndexMinMSE)];
% coeffi_power =  [B2(:,FitInfo2.IndexMinMSE)', FitInfo2.Intercept(FitInfo2.IndexMinMSE)];
% 
% csvwrite('coeff_pool.txt', [coeffi_runtime; coeffi_power]);

% y3 = runtime.*power;
% [B3,FitInfo3] = lasso(X, y3, 'CV', 10);
% y_energy = X * B3(:,FitInfo3.IndexMinMSE) + FitInfo3.Intercept(FitInfo3.IndexMinMSE);
% mse_energy = sqrt(mean(((y_energy - y3)./y3).^2));
% % mse_energy = sqrt(mean(((y_energy - y3)).^2));
% % mse_energy = sqrt(mean(((y_power.*y_runtime - y3)./y3).^2));
% % disp(mse_energy);
% 
% fprintf('%.4f\n%.4f\n%.4f\n', mse_runtime, mse_power, mse_energy)
% fprintf('%.4f \t%.4f \t%.4f \t%.4f\n', min(runtime),max(runtime), mean(runtime), median(runtime))
% fprintf('%.4f \t%.4f \t%.4f \t%.4f\n', min(power),max(power), mean(power), median(power))
% % fprintf('%.4f, %.4f, %.4f, %.4f\n', min(runtime),max(runtime), mean(runtime), median(runtime))
% % fprintf('%.4f, %.4f, %.4f, %.4f\n', min(power),max(power), mean(power), median(power))

% figure(1)
% subplot(1,3,1)
% scatter(y1, y_runtime,'b');
% hold on;
% scatter(y1, y1,'k');
% hold off;
% subplot(1,3,2)
% scatter(y2, y_power,'b');
% hold on;
% scatter(y2, y2,'k');
% hold off;
% subplot(1,3,3)
% scatter(y3, y_energy,'b');
% hold on;
% scatter(y3, y3,'k');
% hold off;
% %% linfit
% % y = runtime;
% B = X\y;
% mse = sqrt(mean(((X*B - y)./y).^2))
% 
% 
% X(:,1:3) = log2(X(:,1:3) + 1);
% X(:,10:11) = log2(X(:,10:11) + 1);
% [B,FitInfo] = lasso(X, y, 'CV', 10);
% y_ls = X * B(:,FitInfo.IndexMinMSE) + FitInfo.Intercept(FitInfo.IndexMinMSE);
% mse_ls = sqrt(mean(((y_ls - y)./y) .^ 2));
% disp(mse_ls);