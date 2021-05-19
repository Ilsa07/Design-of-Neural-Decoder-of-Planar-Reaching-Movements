function [modelParameters] = positionEstimatorTraining(trainingData)

    % Optimal parameters (found using grid-search)
    reg_window = 168; % Regressor window size
    class_window = 90; % Classifier window size
    class_slider = 18; % Classifier slider
    class_delay = 165; % Classifier delay
    contribution = 0.64; % Parameter used in testing: controls how much prediction and mean contribute to the final prediction value
    explanation_threshold = 94; % Explanation threshold for PCA
    

    data_len = length(trainingData);

    % Trajectory removal
    dist_ok = remove_bad_trajectories(trainingData);
    
    % === Remove the bad neurons that have low firing rates ===
    neurons = zeros(98,1);
    for t =1:length(trainingData)
        for a = 1:8
            for n=1:98
                neurons(n) = neurons(n) + sum(trainingData(t,a).spikes(n,:));
            end
        end
    end
    
    a = neurons(:) / (8*data_len);
    good_neurons = find(0.3 < a); % Find the indices of neurons that fire over a certain threshold
    
    % Store the neurons that are good in the final training dataset
    for t = 1:length(trainingData)
        for a = 1:8
            trainingData(t,a).spikes = trainingData(t,a).spikes(good_neurons,:);
        end
    end
    
    % Compute mean spiking rates
    window_size = class_window;
    slider = class_slider;
    [mean_spikes_rounded] = compute_spike_features(trainingData,length(good_neurons), window_size, slider);    
    
    % ===== Store the parameters in the models struct ===== %
    delay = class_delay;
    models(9).contribution = contribution;
    models(9).good_neurons = good_neurons;
    models(9).slider=slider;
    models(9).delay = delay;
    models(9).reg_window = reg_window;
    models(9).class_window = window_size;
    models(9).mean_spikes_rounded = mean_spikes_rounded;

    
    % ===== Train the regression models ===== %
    
    % Initialize data structures for fast computation
    means_x = zeros(8,1000);
    means_y = zeros(8,1000);
    % Initialize angle datasets 
    step = 10; % How many neurons we sum along the dimension of 1 to 98
    feature_len = ceil(length(good_neurons)/step); % Get the feature length based on step size
    models(9).feature_len = feature_len;
    models(9).neuron_step = step;
    for angle = 1:8
        datasets(angle).idx = 0;
        datasets(angle).x = zeros(2*95,4000);
        datasets(angle).y = zeros(2,4000);
    end
    
    window_size = reg_window;
    window_center = window_size/2;

    for trial = 1:data_len  
        for angle = 1:8
            if dist_ok(trial,angle)==1
                for ms = 260:20:length(trainingData(trial,angle).spikes)
                    win_start = ms-delay-window_center; % Set window start to account for the delay and window center
                    win_end = win_start + window_size;
                    
                    % Fing the segment from the spike data based on the
                    % window
                    segment = trainingData(trial,angle).spikes(:,win_start:win_end);
                    features = sum(segment,2); % Sum the spikes to get the firing rates
                    
                    % Second window that starts at the end of the first
                    % window
                    segment_2 = trainingData(trial,angle).spikes(:,win_end:win_end+70);
                    features_2 = sum(segment_2,2);
                    
                    features = [features; features_2];
                    
                    % Store the labels of each segment for training
                    y = trainingData(trial,angle).handPos(1:2,ms);

                    % Every 20 ms compute the mean (this will be used in test)
                    if mod(ms,20)==0 & ms > 319
                        means_x(angle,ms) = means_x(angle,ms) + y(1);
                        means_y(angle,ms) = means_y(angle,ms) + y(2);
                        % Keep track of counts for division later
                        means_x(angle,ms-1) = means_x(angle,ms-1)+1;
                        means_y(angle,ms-1) = means_y(angle,ms-1)+1;
                    end

                    % create the datasets
                    % dataset index counter for each angle to know where to
                    % update
                    datasets(angle).idx = datasets(angle).idx+1;
                    datasets(angle).x(:,datasets(angle).idx) = features;
                    datasets(angle).y(:,datasets(angle).idx) = y;
                end
            end
        end
    end
    
    % Remove zeroes that were not filled
    for i=1:8
        datasets(i).x = datasets(i).x(:,1:datasets(i).idx);
        datasets(i).y = datasets(i).y(:,1:datasets(i).idx);
    end
    
    % Find the average for each segment
    for a=1:8
        for ms = 260:20:900
            if means_x(a,ms) == 0 || means_x(a,ms-1) == 0
                means_x(a,ms) = 0;
            elseif means_y(a,ms) == 0 || means_y(a,ms-1) == 0
                means_y(a,ms) = 0;
            else
                means_x(a,ms) = means_x(a,ms)/means_x(a,ms-1);
                means_y(a,ms) = means_y(a,ms)/means_y(a,ms-1);
            end
        end
    end
    models(10).means.x = means_x;
    models(10).means.y = means_y;
    
    % ========================= Model Training start ======================
    for i=1:8 
        % Find linearly independent dimensions
        [Xsub,indices]=licols(datasets(i).x',0.1);
        % PCA 
        % change this var to tune PCA 
        [xtrain_pca,mu,coeff,idx] = compute_pca(Xsub,explanation_threshold);        
        
        
        y = datasets(i).y';
        data = [xtrain_pca y];
        
        % Train the models on our data
        [trainedModel_x] = trainRegressionModel_tree_x(data,1,i);
        [trainedModel_y] = trainRegressionModel_tree_y(data,1,i);
        
        % Put the models and PCA into a struct
        models(i).reg_x = trainedModel_x;
        models(i).reg_y = trainedModel_y;
        models(i).pca.mu = mu;
        models(i).pca.coeff = coeff(:,1:idx);
        models(i).pca.idx = indices;
    end
  
% ========================= Model Training end ====================== 
    % Output  
    [modelParameters] = models;

end

% PCA function
function [x_train_pca,c_mu,c_coeff,idx] = compute_pca(x_train,explanation_threshold)
    [coeff,scoreTrain,~,~,explained,mu] = pca(x_train);
    sum_explained = 0;
    idx = 0;
    while sum_explained < explanation_threshold
        idx = idx + 1;
        sum_explained = sum_explained + explained(idx);
    end
    c_mu =  mu;
    c_coeff = coeff(:,1:idx);
    x_train_pca = scoreTrain(:,1:idx);
end


% ============================ REGRESSION ============================= %
% ============================ REGRESSION ============================= %
% ============================ REGRESSION ============================= %
%1. Linear least squares 2. Binary Decision Tree 3. Binary Decision Tree - Optimised 
%4. Bagged Decision Trees 5. Support Vector Machine 6. Gaussian Process

function [trainedModel] = trainRegressionModel_tree_x(trainingData, type,trial)	
    inputTable = trainingData;  
    num_predictors = size(inputTable,2)-2;
    predictors = inputTable(:,1:num_predictors);
    response = inputTable(:,num_predictors+1);

    if type == 1
            % Linear model - least squares method
        regressionLM = fitlm(...
            predictors, ...
            response);
        trainedModel.regressor = regressionLM;
    end
    %Binary Decision Trees - Initial Version
    if type == 2        
        rng default
        regressionTree = fitrtree(...
            predictors, ...
            response,'MinParentSize',40 ,'Surrogate', 'off');
        trainedModel.regressor = regressionTree;
    end
    
    %Binary Decision Tree - Optimised
     if type == 3
        switch trial
            case 1
                leaf=27;
            case 2
                leaf=3;
            case 3
                leaf=22;
            case 4
                leaf=25;
            case 5
                leaf=23;
            case 6
                leaf=21;
            case 7
                leaf=30;
            case 8
                leaf=28;
        end
        rng default
        regressionTree = fitrtree(...
            predictors, ...
            response,'MinParentSize',40 ,'Surrogate', 'off',...
            'MinLeafSize', leaf, ...
            'MinParentSize',40 ,'Surrogate', 'off');
        trainedModel.regressor = regressionTree;
     end
    
     %Bagged Decision Trees
    if type == 4
        %figure(2)
        %hold on
        Mdl = TreeBagger(10,predictors,response,'Method','regression','OOBPrediction','On');
        trainedModel.regressor = Mdl;

    end
    
    %Support Vector Machine with Linear Kernel Function
    if type == 5
        regressionSVM = fitrsvm(...
            predictors, ...
            response, ...
            'KernelFunction', 'linear', ...           
            'Standardize', true);
        trainedModel.regressor = regressionSVM;
    end
    
    if type == 6
            % Gaussian Process with Exponential Kernel Function
        regressionGP = fitrgp(...
            predictors, ...
            response, ...
            'BasisFunction', 'constant', ...
            'KernelFunction', 'exponential', ...
            'Standardize', true);
        trainedModel.regressor = regressionGP;
    end

end

function [trainedModel] = trainRegressionModel_tree_y(trainingData, type,trial)

    inputTable = trainingData;  
    num_predictors = size(inputTable,2)-2;
    predictors = inputTable(:,1:num_predictors);
    response = inputTable(:,num_predictors+2);

   if type == 1
            % Linear model - least squares method
        regressionLM = fitlm(...
            predictors, ...
            response);
        trainedModel.regressor = regressionLM;
    end
    %Binary Decision Trees - Initial Version
    if type == 2        
        rng default
        regressionTree = fitrtree(...
            predictors, ...
            response,'MinParentSize',40 ,'Surrogate', 'off');
        trainedModel.regressor = regressionTree;
    end
    
    %Binary Decision Tree - Optimised
     if type == 3
        switch trial
               case 1
                leaf=3;
            case 2
                leaf=3;
            case 3
                leaf=19;
            case 4
                leaf=6;
            case 5
                leaf=2;
            case 6
                leaf=17;
            case 7
                leaf=36;
            case 8
                leaf=69;
        end
        rng default
        regressionTree = fitrtree(...
            predictors, ...
            response,'MinParentSize',40 ,'Surrogate', 'off',...
            'MinLeafSize', leaf, ...
            'MinParentSize',40 ,'Surrogate', 'off');
        trainedModel.regressor = regressionTree;
     end
    
     %Bagged Decision Trees
    if type == 4
        %figure(2)
        %hold on
        Mdl = TreeBagger(10,predictors,response,'Method','regression','OOBPrediction','On');
        trainedModel.regressor = Mdl;

    end
    
    %Support Vector Machine with Linear Kernel Function
    if type == 5
        regressionSVM = fitrsvm(...
            predictors, ...
            response, ...
            'KernelFunction', 'linear', ...           
            'Standardize', true);
        trainedModel.regressor = regressionSVM;
    end
    
    if type == 6
            % Gaussian Process with Exponential Kernel Function
        regressionGP = fitrgp(...
            predictors, ...
            response, ...
            'BasisFunction', 'constant', ...
            'KernelFunction', 'exponential', ...
            'Standardize', true);
        trainedModel.regressor = regressionGP;
    end

end

% ===================== Regression end =============================
% ===================== Regression end =============================
% ===================== Regression end =============================


% Remove linearly dependend columns
function [Xsub,idx]=licols(X,tol)
     if ~nnz(X) %X has no non-zeros and hence no independent columns
         Xsub=[]; idx=[];
         return
     end
     if nargin<2, tol=1e-10; end
       [Q, R, E] = qr(X,0); 
       if ~isvector(R)
        diagr = abs(diag(R));
       else
        diagr = R(1);   
       end
       %Rank estimation
       r = find(diagr >= tol*diagr(1), 1, 'last'); %rank estimation
       idx=sort(E(1:r));
       Xsub=X(:,idx);
end


% Function to remove bad trajectories
function dist_ok = remove_bad_trajectories(trainingData)
    [dim1,dim2]=size(trainingData);    
    trajectories=zeros(2,350,dim1,dim2);
    median_trajectory=zeros(2,350,dim2);
    dist_ok=ones(dim1,dim2,1);
    dist=ones(dim1,dim2,1);
    %obtain the trajectory from the central values
    for angle=1:dim2
        for trial=1:dim1
            trajectory=trainingData(trial,angle).handPos(1:2,201:550);
            %normalize the trajectory along each coordinate (so x varies from
            %-1 to 1, y from -1 to 1 and z from -1 to 1
            norm_trajectory=trajectory./sqrt( sum(trajectory.^2,2) );
            trajectories(:,:,trial,angle)=norm_trajectory;
        end
        %obtain the median trajectory from the obtained values
        median_trajectory(:,:,angle)=median(trajectories(:,:,:,angle),3);  
        %Compute the distance between each line and the median trajectory
        if (angle==2||angle==5)
                threshold=0.29;
         elseif angle==8
                threshold=0.35;
         elseif (angle==1||angle==7)
              threshold=0.26;
         else
            threshold=0.18;
         end
        for trial=1:dim1
            traj=trajectories(1:2,:,trial,angle);
            dist(trial,angle)=abs(norm(traj - median_trajectory(1:2,:,angle)));
            %If the distance is greater than 0.15 (we can choose other value),
            %label it
            if dist(trial,angle)>threshold
                dist_ok(trial,angle)=0;
            end
        end
    end
end

% Function to compute the spike vectors which we will use to classify the
% angle
function [mean_spikes] = compute_spike_features(trial,neur_count, window_size, slider)
    
    % Find windows 
    w_set_size = length(1:slider:320);

    spike_counts = zeros(neur_count,8,w_set_size); % Count the total number of spikes for each angle    
    for t = 1:length(trial)
        for a = 1:8
            w_counter = 1;
            for w = 1:slider:320
                for n = 1:neur_count
                    if w + window_size > 400 % end when over the limit
                        break
                    end
                    
                    neur_sum = sum(trial(t,a).spikes(n,w:w+window_size)); % Total spikes in the current trial_angle for a specific neuron
                    spike_counts(n,a,w_counter) = spike_counts(n,a,w_counter) + neur_sum;

                end
                w_counter = w_counter +1;
                % Store each window here
                
            end
           
        end
    end
    mean_spikes = spike_counts / length(trial); % Divide the spikes with the number of trials to get the mean
%     mean_spikes_rounded = round(mean_spikes); % round 'em
end