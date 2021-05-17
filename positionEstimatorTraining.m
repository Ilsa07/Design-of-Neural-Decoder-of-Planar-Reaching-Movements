% Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.

function [modelParameters] = positionEstimatorTraining(trainingData)
    % Trajectory removal
    dist_ok = remove_bad_trajectories(trainingData);
    
    % Remove the baddies
    neurons = zeros(98,1);
    % Find and remove neurons that have low firing rates
    for t =1:length(trainingData)
        for a = 1:8
            for n=1:98
                neurons(n) = neurons(n) + sum(trainingData(t,a).spikes(n,:));
            end
        end
    end
    
    % Remove all inactive neurons
    a = neurons(:) / (8*80);
    good_neurons = find(0.3 < a); % Find the indices of neurons that fire over a certain threshold
    
    for t = 1:length(trainingData)
        for a = 1:8
            trainingData(t,a).spikes = trainingData(t,a).spikes(good_neurons,:);
        end
    end
    % Compute mean spiking rates
    window_size = 109;
    slider = 22;
    [mean_spikes_rounded] = compute_spike_features(trainingData,length(good_neurons), window_size, slider);
    %mean_spikes = compute_mean_spikes(trainingData, length(good_neurons));
    
    
    % ===== Initialize variables ===== %

    % Window center will affect performance, set default to be the middle
    window_center = window_size/2;
    delay = 170;
    
    
%     % =========== Classification =========== %
%     % Extract segments from the raw data
%     x_train = zeros(50000,    length(good_neurons));
%     y_train = zeros(50000,1);
%     
%     % !!! You can tune the following parameters !!!
%     classification_window = 160; % window length for classifier
%     w_c = classification_window/2;
%     start_c = 200; % start of neural data extraction (not all of 320ms will be used)
%     step_size = 10; % How many ms we skip until we extract data again
%     explanation_threshold = 50;  % Tune this variable for PCA (lower value = more neurons removed)
%     
%     % 50 pca is good 0.83 acc
%     
%     counter = 1;
%     reg_x_train = zeros(100000,    length(good_neurons));
%     reg_y_train = zeros(100000,2);
%     angle_labels = zeros(100000,1);
%     reg_counter = 1;
%     tic
%     for trial = 1:80 % Tune this
%         for angle = 1:8 
%             if dist_ok(trial,angle)==1
%                 % Create a dataset for building the 10th regressor 
%                 for ms = start_c:step_size:320
%                     win_start = ms-w_c;
%                     win_end = win_start + classification_window;
%                     
%                     % If window end is over 320 ms we break the loop
%                     if win_end > 320
%                         break
%                     end
%                     
%                     % Get the spikes based on window size
%                     segment = trainingData(trial,angle).spikes(:,win_start:win_end);
%                     features = sum(segment,2);
%                     % Get the hand position in the future delay steps away
%                     y = trainingData(trial,angle).handPos(1:2,ms+delay); 
%                     reg_x_train(reg_counter,:) = features;
%                     reg_y_train(reg_counter,:) = y;
%                     % Angle labels
%                     angle_labels(reg_counter,1) = angle;
%                     reg_counter = reg_counter +1;
% 
%                 end
%             end
%         end
%     end
%     
%     % Remove unfilled fields
%     x_train = x_train(1:counter-1,:);
%     y_train = y_train(y_train~=0);
%     reg_x_train = reg_x_train(1:reg_counter-1,:);
%     reg_y_train = reg_y_train(1:reg_counter-1,:);
%     angle_labels = angle_labels(1:reg_counter-1,:);
%      
%     
%     
%     fprintf('\nSegmentation for classification done \n')
%     toc
% 
%     tic
%     
% %     % ===== Regression training (for classifier) ===== %
% %     % 1. Reduce the dimensionality of the data
% %     %     - Remove linearly dependent dimensions
% %     %     - Do PCA on the remaining dimensions
% %     
% %     % Find linearly independent dimensions
% %     [X_reg_sub,reg_indices]=licols(reg_x_train,0.1); % remove dependend dimensions
% %     [x_reg_train_pca,reg_c_mu,reg_c_coeff,reg_idx] = compute_pca(X_reg_sub,explanation_threshold); % PCA
% %     reg_data = [x_reg_train_pca reg_y_train]; % concatenate features with responses
% %     dim = size(reg_data,2); % get the dim after PCA
% %     
% %     
%     % 2. Split the data into various training and test splits
%     %     - Create a train and test set for the regressor
%     %     - Create a 
%     reg_data_train = reg_data; % Data for training the regressor
%     class_data_train = reg_data; % Data for predicting the coordinates for training the classifier
%     class_data_labels = angle_labels; % Angle labels for training the classifier
% 
%     
%     % Temporary: Write the data into a csv file for using the learner
%     % app. 
%     reg_x_train = reg_data_train(:,1:dim-1);
%     reg_y_train = reg_data_train(:,1:dim-2);
%     y_reg = reg_data_train(:,dim);
%     reg_y_train = [reg_y_train y_reg];
%     writematrix(reg_x_train, 'reg_x_train.csv'); % Remove this later
%     writematrix(reg_y_train, 'reg_y_train.csv'); % Remove this later
% % 
% %     writematrix(reg_data_train(:,1:dim-2), 'reg_data_input.csv'); % Remove this later
% %     writematrix(reg_data_train(:,dim-1), 'reg_data_target_x.csv'); % Remove this later
% %     writematrix(reg_data_train(:,dim), 'reg_data_target_y.csv'); % Remove this later
% % 
% %     
%     % 3. Train the regressor
%     %     - Train a simple neural network
%     hiddenSizes = 15;
%     net = fitnet(hiddenSizes, 'trainlm'); % set the network parameters
%     net_y = fitnet(hiddenSizes, 'trainlm'); % set the network parameters
%     x_train = reg_data_train(:,1:dim-2); 
%     y_train = reg_data_train(:,dim-1);
%     y_train_2 = reg_data_train(:,dim); 
% 
%     net = train(net,x_train',y_train'); % train for x
%     net_y = train(net_y,x_train',y_train_2'); % train for y
%  
%     % linear tree
%     [trainedModel_x] = trainRegressionModel_tree_x(reg_data,1);
%     [trainedModel_y] = trainRegressionModel_tree_y(reg_data,1);
%     
%     
%     % 4. Predict coordinates from training data (I know, temporary)
%     pred_coord = net(class_data_train(:,1:dim-2)'); 
%     pred_coord_y = net_y(class_data_train(:,1:dim-2)'); 
%     pred_coord = pred_coord';
%     pred_coord_y = pred_coord_y';

    % 4. Prepare the dataset for classifier training
    %     - Concatenate the corresponding angle labels to the predicted
    %       coodinates
% %     class_data = [Y class_data_labels];
%     class_data = [pred_coord class_data_labels];
%     class_data_y = [pred_coord_y class_data_labels];
% 
% 
%     % ===== Classifier training (from regression outputs ===== %
%     writematrix(class_data, 'class_data.csv'); % Remove this later
%     [trainedClassifier, validationAccuracy] = trainClassifier(class_data); % Train a classifier on tree regression data
%     [trainedClassifier_y, validationAccuracy] = trainClassifier(class_data_y); % Train a classifier on tree regression data
%     fprintf('Classifier training done \n')
%     toc
    
    
    % FINALLY test the classifier on unseen data (for easier debugging)
%     x_pred = trainedModel_x.regressor.predict(class_test_x(:,1:size(class_test_x,2)-2)); % First predict the hand coordinages
%     y_pred = trainedModel_y.regressor.predict(class_test_x(:,1:size(class_test_x,2)-2));
%     % Prepare the testing dataset for classifier
%     class_data = [x_pred y_pred]; % Append the x and y predictions for classification
%     angle_pred = trainedClassifier.ClassificationKNN.predict(class_data);        % Predict the angle from these coordinates
%     angle_pred_vs_truth = [angle_pred class_test_y]; % Create a ground prediction vs ground truth matrix for easy comparison
% 
%     % Do the same for NN approach
%     [Y_nn,Xf,Af] = myNeuralNetworkFunction(class_test_x(:,1:size(class_test_x,2)-2),a,a); % NN
%     x_pred_nn = Y_nn(:,1);
%     y_pred_nn = Y_nn(:,2);
%     % Prepare the testing dataset for classifier
%     class_data = [x_pred_nn y_pred_nn]; % Append the x and y predictions for classification
%     angle_pred = trainedClassifier.ClassificationKNN.predict(class_data);        % Predict the angle from these coordinates
%     angle_pred_vs_truth_nn = [angle_pred class_test_y]; % Create a ground prediction vs ground truth matrix for easy comparison
%     
     
    
    % Add the classifier and PCA from earlier to our modelParams
    % which we can use later in position estimator
%     models(9).classifier = trainedClassifier;
%     models(9).classifier_y = trainedClassifier_y;
%     models(9).classifier.step_size = step_size;
%     models(9).classifier.window_size = classification_window;
%     models(9).classifier.start_c = start_c;
%     models(9).pca.mu = reg_c_mu;
%     models(9).pca.coeff = reg_c_coeff(:,1:reg_idx); 
%     models(9).licols.idx = reg_indices; % return the dimensions that need to be retained from licols
%     models(9).reg_net_x = net; % store the regressor network
%     models(9).reg_net_y = net_y; % store the regressor network
%     models(9).reg_tree_x = trainedModel_x; % store tree/gp regressors
%     models(9).reg_tree_y = trainedModel_y;
     models(9).good_neurons = good_neurons;
     models(9).slider=slider;
     models(9).delay = delay;
     models(9).window_size = window_size;
     models(9).mean_spikes_rounded = mean_spikes_rounded; % Store the mean spikes for classification
    
     window_size = 150;
     window_center = window_size/2;

     
    % Set the NN classifier as 11th model for now
    
    % ===== Traing the regression models ===== %
    % initialize velocities structure
    % apologies for bad code here
    tic
    velocities = struct('vel', repmat({zeros(1)}, 8, 1));
    for i = 1:8
        velocities(i).vel = [];
    end
    
    % Initialize data structures for fast computation
    means_x = zeros(8,1000);
    means_y = zeros(8,1000);
    % Initialize angle datasets 
    for angle = 1:8
        datasets(angle).idx = 0;
        datasets(angle).x = zeros(length(good_neurons),4000);
        datasets(angle).y = zeros(2,4000);
    end
    
    divisor=0;
    
    % We are iterating in steps of 10
    for trial = 1:40  %length(trainingData)
        for angle = 1:8
            if dist_ok(trial,angle)==1
                for ms = 260:10:length(trainingData(trial,angle).spikes)
                    win_start = ms-delay-window_center;
                    win_end = win_start + window_size;

                    segment = trainingData(trial,angle).spikes(:,win_start:win_end);
                    features = sum(segment,2);
                    y = trainingData(trial,angle).handPos(1:2,ms);


                    % compute velocities
    %                 velocity = compute_vel(trainingData,trial,angle,ms);
    %                 
    %                 velocities(angle).vel = [velocities(angle).vel velocity];

                    % Every 20 ms compute the mean (this will be used in test)

                    if mod(ms,10)==0 & ms > 319
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
    
    
    
    % Find the average for each n ms segment
    for a=1:8
        for ms = 260:10:900
            means_x(a,ms) = means_x(a,ms)/means_x(a,ms-1);
            means_y(a,ms) = means_y(a,ms)/means_y(a,ms-1);
        end
    end
    models(10).means.x = means_x;
    models(10).means.y = means_y;
    
    fprintf('Feature extraction from training data done \n');
    toc
    % ========================= Model Training start ======================
    % ========================= Model Training start ======================
    % ========================= Model Training start ======================
    % ========================= Model Training start ======================
    
    
    reg_set = [];
    
    tic;
    for i=1:8
        
        % Find linearly independent dimensions
        [Xsub,indices]=licols(datasets(i).x',0.1);
        % PCA 
        % change this var to tune PCA 
        explanation_threshold = 80;
        [xtrain_pca,mu,coeff,idx] = compute_pca(Xsub,explanation_threshold);

%         xtrain_pca = vertcat(xtrain_pca',velocities(i).vel);
        
        
        y = datasets(i).y';
        data = [xtrain_pca y];
%         writematrix(data, 'reg_data.csv');
        
        % linear tree
        [trainedModel_x] = trainRegressionModel_tree_x(data,1);
        [trainedModel_y] = trainRegressionModel_tree_y(data,1);

        
        % Put the models and PCA into a struct
        models(i).reg_x = trainedModel_x;
        models(i).reg_y = trainedModel_y;
        models(i).pca.mu = mu;
        models(i).pca.coeff = coeff(:,1:idx);
        models(i).pca.idx = indices;
    end
    
    
    
    fprintf('PCA and Regression training done \n');
    toc;
    
    
% ========================= Model Training end ======================
% ========================= Model Training end ======================
% ========================= Model Training end ======================
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

% Velocity function
function velocity = compute_vel(trainingData, trial, angle, ms)
    vel_slice = 20;
    start_p = trainingData(trial,angle).handPos(1:2,ms-vel_slice);
    end_p = trainingData(trial,angle).handPos(1:2,ms);
    velocity = sqrt((end_p(1) - end_p(1))^2 ...
        + (end_p(2) - start_p(2))^2);         
end

% ============ CLASSIFICATION ============== %
% ============ CLASSIFICATION ============== %
% ============ CLASSIFICATION ============== %

function [trainedClassifier, validationAccuracy, partitionedModel] = trainClassifier(trainingData)


    % Extract predictors and response
    % This code processes the data into the right shape for training the
    % model.
    inputTable = trainingData;
    % Find dimension of the data
    dim = size(inputTable,2);
    predictors = inputTable(:, 1:dim-1);
    response = inputTable(:,dim);

    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
%     classificationKNN = fitcknn(...
%     predictors, ...
%     response, ...
%     'Distance', 'Euclidean', ...
%     'Exponent', [], ...
%     'NumNeighbors', 1, ...
%     'DistanceWeight', 'Equal', ...
%     'Standardize', true, ...
%     'ClassNames', [1; 2; 3; 4; 5; 6; 7; 8]);

    classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true, ...
    'ClassNames', [1; 2; 3; 4; 5; 6; 7; 8]);


    trainedClassifier.ClassificationKNN = classificationKNN;

    % Compute resubstitution accuracy
    validationAccuracy = 1 - resubLoss(trainedClassifier.ClassificationKNN, 'LossFun', 'ClassifError');
    
end


% ============================ REGRESSION ============================= %
% ============================ REGRESSION ============================= %
% ============================ REGRESSION ============================= %


function [trainedModel] = trainRegressionModel_tree_x(trainingData, type)

    inputTable = trainingData;  
    num_predictors = size(inputTable,2)-2;
    predictors = inputTable(:,1:num_predictors);
    response = inputTable(:,num_predictors+1);

    if type == 1
        % Train a regression model
        % This code specifies all the model options and trains the model.
        regressionTree = fitrtree(...
            predictors, ...
            response, ...
            'MinLeafSize', 4, ...
            'Surrogate', 'off');
            % CHANGE HERE LATER
        trainedModel.regressor = regressionTree;
    end
    
    if type == 2
            % Gaussian Process!
        regressionGP = fitrgp(...
            predictors, ...
            response, ...
            'BasisFunction', 'constant', ...
            'KernelFunction', 'exponential', ...
            'Standardize', true);
        trainedModel.regressor = regressionGP;
    end

end

function [trainedModel] = trainRegressionModel_tree_y(trainingData, type)

    inputTable = trainingData;  
    num_predictors = size(inputTable,2)-2;
    predictors = inputTable(:,1:num_predictors);
    response = inputTable(:,num_predictors+2);

    if type == 1
        % Train a regression model
        % This code specifies all the model options and trains the model.
        regressionTree = fitrtree(...
            predictors, ...
            response, ...
            'MinLeafSize', 4, ...
            'Surrogate', 'off');
            % CHANGE HERE LATER
        trainedModel.regressor = regressionTree;
    end
    
    if type == 2
            % Gaussian Process!
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
%Extract a linearly independent set of columns of a given matrix X
%
%    [Xsub,idx]=licols(X)
%
%in:
%
%  X: The given input matrix
%  tol: A rank estimation tolerance. Default=1e-10
%
%out:
%
% Xsub: The extracted columns of X
% idx:  The indices (into X) of the extracted columns
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
    trajectories=zeros(3,350,dim1,dim2);
    median_trajectory=zeros(3,350,dim2);
    dist_ok=ones(dim1,dim2,1);
    dist=ones(dim1,dim2,1);
    %obtain the trajectory from the central values
    for angle=1:dim2
        for trial=1:dim1
            trajectory=trainingData(trial,angle).handPos(:,201:550);
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

% function mean_spikes_rounded = compute_mean_spikes(trial,neur_count)
%     spike_counts = zeros(neur_count,8); % Count the total number of spikes for each angle
%     sp_start = 1;
%     sp_end = 320;
%     for t = 1:length(trial)
%         for a = 1:8
%             for n = 1:neur_count
%                 neur_sum = sum(trial(t,a).spikes(n,sp_start:sp_end)); % Total spikes in the current trial_angle for a specific neuron
%                 spike_counts(n,a) = spike_counts(n,a) + neur_sum;
%             end
%         end
%     end
% 
%     mean_spikes = spike_counts / length(trial); % Divide the spikes with the number of trials to get the mean
%     mean_spikes_rounded = round(mean_spikes); % round 'em
% end

function [mean_spikes_rounded] = compute_spike_features(trial,neur_count, window_size, slider)
    
    % Find windows 
    w_set_size = length(1:slider:320);


    spike_counts = zeros(neur_count,8,w_set_size); % Count the total number of spikes for each angle
%     spike_max = zeros(neur_count,8,w_set_size); % Keep track of the max number of spikes 
%     spike_min = zeros(neur_count,8,w_set_size) + 1000;
%     sp_start = 1;
%     sp_end = 320;
    
    
    
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
    mean_spikes_rounded = round(mean_spikes); % round 'em
end