function [x, y, newParameters] = positionEstimator(test_data, modelParameters)
    
    % Store all the parameters in more convenient variable names
    newParameters = modelParameters;
   	current_coordinate = length(test_data.spikes);
    class_window = newParameters(9).class_window;
    reg_window = newParameters(9).reg_window;
    window_center = reg_window/2;
    delay = newParameters(9).delay;
    contribution = newParameters(9).contribution;

    % Extract the good neurons
    test_data.spikes = test_data.spikes(newParameters(9).good_neurons,:);

    % ===== Predict the angle ===== %
    if current_coordinate < 420
        spikes = test_data.spikes; % Get the spikes
        mean_spikes = newParameters(9).mean_spikes_rounded; % Get the mean spikes
        slider = newParameters(9).slider; % Get the slider
        predicted_angle = classify_angle(spikes, mean_spikes,class_window, slider);
        newParameters(9).predicted_angle = predicted_angle;

    end
    
    predicted_angle = newParameters(9).predicted_angle;
    
    % If there are multiple predicted angles (both angles equal
    % probability) Then just randomly pick the first prediction in the list
    % as the predicted angle
    if length(predicted_angle) > 1
        newParameters(9).predicted_angle = predicted_angle(1);
    end
   

    % ===== Predict the hand coordinate ===== %
    % Extract the predictor segment based on the window
    win_start = current_coordinate-delay-window_center;
    win_end = win_start+reg_window;
    segment = test_data.spikes(:,win_start:win_end);
    % Sum the spikes over window timesteps
    features = sum(segment,2);
    
    segment_2 = test_data.spikes(:,win_end:win_end+70);
    features_2 = sum(segment_2,2); 
    features = [features; features_2];
    
    % Use PCA on the features
    predicted_angle = newParameters(9).predicted_angle;
    Xsub = features(newParameters(predicted_angle).pca.idx,:);

    mu = newParameters(predicted_angle).pca.mu;
    coeff = newParameters(predicted_angle).pca.coeff;
    x_test_pca = (Xsub'-mu)*coeff;
    
    % Predict the coordinates
    x_regressor = newParameters(predicted_angle).reg_x.regressor;
    y_regressor = newParameters(predicted_angle).reg_y.regressor;
    
    x_pred = x_regressor.predict(x_test_pca);
    y_pred = y_regressor.predict(x_test_pca);
    
    % access the average of the current coordinate
    x_med = newParameters(10).means.x(predicted_angle,current_coordinate);
    y_med = newParameters(10).means.y(predicted_angle,current_coordinate);
    
    %No post-processing
    
    % Outputs
    x = x_pred;
    y = y_pred;
   
    
end


% Remove linearly dependend columns
function [Xsub,idx]=licols(X,tol)
%Extract a linearly independent set of columns of a given matrix X
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



% Angle classifier
function predicted_angle = classify_angle(spikes, mean_spikes,window_size, slider)

    sp_start = 1; % How much of the trial we use
    sp_end = 320;
    
    totals = zeros(8,1); % A list for keeping track of the distances for each corresponding angle
    num_segments = size(mean_spikes,3);

    
    for m = 1:8 % Compare against each mean
        seg_idx = 1; % index to keep track of the current segment
        total = 0; % to keep track of the total distance

        for s = 1:num_segments % Loop over all segments
            if seg_idx+window_size > length(spikes) % Break the loop if we are over 320
                break
            end
            seg = sum(spikes(:,seg_idx:seg_idx+window_size),2); % Grab the current segment
            difference_mean = abs(seg - mean_spikes(:,m,s));
            dist_from_mean = sum(difference_mean);
            total = total+dist_from_mean;
            seg_idx = seg_idx + slider; % Increment the segment index by the slider size
        end
        totals(m) = total;
    end
    predicted_angle = find(totals == min(totals)); % The mean value that is closest to our angle will be the prediction!
end