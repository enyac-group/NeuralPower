clear all;
% result = dlmread('cnn_mnist_results.txt');
% input_feature = dlmread('cnn_mnist_features.txt');
% data = dlmread('fc_random.txt');
% data = dlmread('fc_famous_other.txt');
% data_raw = dlmread('fc_famous_batch.txt');
data_raw = dlmread('res_fc.txt');
data = data_raw;

data = [];
threshold = 100; %runtime
for i = 1:size(data_raw, 1)
    if data_raw(i, 6) < threshold
       data = [data; data_raw(i, :)];
    end
end

% data = dlmread('fc_famous_first.txt');
% data = dlmread('fc_random_fc1.txt');
% data = dlmread('fc_random_fc_others.txt');
% data = dlmread('conv_data.txt');
result = data(:,end - 1: end);
ipt_all = data(:,1: end - 2);
%% construct X
% conv1_size conv2_size conv3_size kernel1_size kernel2_size kernel3_size po1_size po2_size po3_size fc1_size fc2_size 

[row, col] = size(ipt_all);

%log scal
% input_feature(:,1:3) = log2(input_feature(:,1:3) + 1);
% input_feature(:,10:11) = log2(input_feature(:,10:11) + 1);

X_runtime = [];
X_power = [];
for i = 1:row
    ipf = (ipt_all(i, :));
%     ipf = [ipt_all(i,1), ipt_all(i,2), ipt_all(i,3) *ipt_all(i,4) *ipt_all(i,5)];
    % linear
%     tmp1 = [ipf, log(ipf)];
    tmp1 = [ipf];
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
%     tmp = [tmp1];
    tmp = [tmp1, tmp2];
    tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)*ipf(5)];

    X_runtime = [X_runtime; tmp];
end

for i = 1:row
    ipf = (ipt_all(i, :));
%     ipf = [ipt_all(i,1), ipt_all(i,2), ipt_all(i,3) *ipt_all(i,4) *ipt_all(i,5)];
    % linear
    tmp1 = [ipf, log2(ipf)];
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
%     tmp = [tmp1];
    tmp = [tmp1, tmp2];
%     tmp = [tmp, ipf(1)*ipf(2)*ipf(3)*ipf(4)*ipf(5)];

    X_power = [X_power; tmp];
end

% %% add log & exponential terms
% for i = 1:size(X,2)
%     tmp = X(:,i);
%     X = [X, log(tmp)];
% %     X = [X, log(tmp + 1), exp(tmp)];
% end

runtime = result(:,1); % runtime
power = result(:,2);
rmse = [];

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

% save('fc_regre.mat');

coeffi_runtime =  [B1(:,FitInfo1.IndexMinMSE)', FitInfo1.Intercept(FitInfo1.IndexMinMSE)];
coeffi_power =  [B2(:,FitInfo2.IndexMinMSE)', FitInfo2.Intercept(FitInfo2.IndexMinMSE)];

%csvwrite('coeff_fc.txt', [coeffi_runtime; coeffi_power]);
dlmwrite('coeff_fc.txt',coeffi_runtime,'delimiter',',');
dlmwrite('coeff_fc.txt',coeffi_power,'delimiter',',','-append');


% y3 = runtime.*power;
% [B3,FitInfo3] = lasso(X, y3, 'CV', 10);
% y_energy = X * B3(:,FitInfo3.IndexMinMSE) + FitInfo3.Intercept(FitInfo3.IndexMinMSE);
% mse_energy = sqrt(mean(((y_energy - y3)./y3).^2));
% % mse_energy = sqrt(mean(((y_energy - y3)).^2));
% % mse_energy = sqrt(mean(((y_power.*y_runtime - y3)./y3).^2));
% disp(mse_energy);

% figure(1)
% subplot(1,3,1)
% scatter(y2, y_power,'b');
% hold on;
% scatter(y2, y2,'k');

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