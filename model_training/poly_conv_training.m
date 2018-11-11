clear all;
data_raw = dlmread('res_conv.txt');

data = [];
for i = 1:size(data_raw, 1)
    if data_raw(i, 2) ~= 1 %|| data_raw(i,5) == 1
        data = [data; data_raw(i, :)];
    end
end

% data = data_raw;


result = data(:,end - 1: end);
ipt_all = data(:,1: end - 2);
[row, col] = size(ipt_all);

X = [];
X_runtime = [];
X_power = [];

for i = 1:row
    ipf = ipt_all(i, :);
%     tmp1 = [ipf(1:3),ipf(6:8),ipf(14)];
    tmp1 = [ipf(1:3),ipf(6:8),ipf(14)];
%     tmp1 = [tmp1, log2(tmp1)];
%     tmp1 = [ipf(1:3),ipf(5:8),ipf(14:15)];
    % quadratic 
    tmp2 = [];
%     tmp22 = [];
    for j  = 1:length(tmp1)
        for k = j: length(tmp1)
            tmp2 = [tmp2, tmp1(j)*tmp1(k)];
        end
    end
%     tmp22 = [tmp2, log2(tmp22)];
    tmp3 = [];
    for j  = 1:length(tmp1)
        for k = j: length(tmp1)
            for l = k: length(tmp1)
                tmp3 = [tmp3, tmp1(j)*tmp1(k)*tmp1(l)];
            end
        end
    end    
%     tmp = [tmp1, tmp2];
    tmp = [tmp1, tmp2, tmp3]; %is the best 40%
    tmp = [tmp, ipf(2)*ipf(3)*ipf(4)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(13)*ipf(2)*ipf(3)*ipf(4)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(13)*ipf(2)*ipf(3)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)]; %output data
    tmp = [tmp, ipf(5)*ipf(6)*ipf(7)*ipf(8)]; %filter data
    tmp = [tmp, ipf(13)*ipf(14)*ipf(15)*ipf(16)]; %input data
    tmp = [tmp, ipf(13)*ipf(15)*ipf(16)]; %input data (padding)
    tmp = [tmp, ipf(13)*ipf(14)*ipf(16)]; %input data (padding)

    X_runtime = [X_runtime; tmp];
end

for i = 1:row
    ipf = ipt_all(i, :);
%     tmp1 = [ipf(1:3),ipf(6:8),ipf(14)];
    tmp1 = [ipf(1:3),ipf(6:8),ipf(14)];
    tmp11 = [tmp1, log2(tmp1)];
%     tmp1 = [ipf(1:3),ipf(5:8),ipf(14:15)];
    % quadratic 
    tmp2 = [];
    tmp22 = [];
    for j  = 1:length(tmp11)
        for k = j: length(tmp11)
            tmp2 = [tmp2, tmp11(j)*tmp11(k)];
        end
    end
%     tmp22 = [tmp2, log2(tmp22)];
    tmp3 = [];
    for j  = 1:length(tmp1)
        for k = j: length(tmp1)
            for l = k: length(tmp1)
                tmp3 = [tmp3, tmp1(j)*tmp1(k)*tmp1(l)];
            end
        end
    end    
%     tmp = [tmp1, tmp2];
    tmp = [tmp11, tmp2]; %is the best 40%
    tmp = [tmp, ipf(2)*ipf(3)*ipf(4)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(13)*ipf(2)*ipf(3)*ipf(4)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(13)*ipf(2)*ipf(3)*ipf(5)*ipf(6)*ipf(7)]; %output pixels * # of operations per pixel
    tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)]; %output data
    tmp = [tmp, ipf(5)*ipf(6)*ipf(7)*ipf(8)]; %filter data
    tmp = [tmp, ipf(13)*ipf(14)*ipf(15)*ipf(16)]; %input data
    tmp = [tmp, ipf(13)*ipf(15)*ipf(16)]; %input data (padding)
    tmp = [tmp, ipf(13)*ipf(14)*ipf(16)]; %input data (padding)

    X_power = [X_power; tmp];
end

% %% add log & exponential terms
% for i = 1:size(X,2)
%     tmp = X(:,i);
%     X = [X, log(tmp + 1)];
% %     X = [X, log(tmp + 1), exp(tmp)];
% end
% %% normalize
% X_norm = [];
% Y_norm = [];
% for i = 1:size(X,2)
%     tmp = X(:,i);
%     if (max(tmp) == min(tmp))
%         X_norm = [X_norm, tmp];
%     else
%         X_norm = [X_norm, (tmp - min(tmp))/(max(tmp) - min(tmp))];
%     end
% end

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
% 

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

save('conv_regre.mat');
% 
coeffi_runtime =  [B1(:,FitInfo1.IndexMinMSE)', FitInfo1.Intercept(FitInfo1.IndexMinMSE)];
coeffi_power =  [B2(:,FitInfo2.IndexMinMSE)', FitInfo2.Intercept(FitInfo2.IndexMinMSE)];
csvwrite('coeff_conv.txt', [coeffi_runtime; coeffi_power]);


% y3 = runtime.*power;
% [B3,FitInfo3] = lasso(X, y3, 'CV', 10);
% y_energy = X * B3(:,FitInfo3.IndexMinMSE) + FitInfo3.Intercept(FitInfo3.IndexMinMSE);
% mspe_energy = sqrt(mean(((y_energy - y3)./y3).^2));
% mse_energy = sqrt(mean(((y_energy - y3)).^2));
% % mse_energy = sqrt(mean(((y_power.*y_runtime - y3)./y3).^2));
% fprintf('%.4f, %.4f\n', mspe_energy, mse_energy);

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