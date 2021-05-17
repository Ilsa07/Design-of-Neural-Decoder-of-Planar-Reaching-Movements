function [x, y, newParameters] = positionEstimator(test_data, modelParameters)
    
    newParameters = modelParameters;
   	current_coordinate = length(test_data.spikes);
    window_size = newParameters(9).window_size;
    window_center = window_size/2;
    delay = newParameters(9).delay;
    
    
    % Extract the good neurons
    test_data.spikes = test_data.spikes(newParameters(9).good_neurons,:);

    % ===== Predict the angle ===== %
    if current_coordinate < 420
        spikes = test_data.spikes; % Get the spikes
        mean_spikes = newParameters(9).mean_spikes_rounded; % Get the mean spikes
        slider = newParameters(9).slider; % Get the slider
        predicted_angle = classify_angle(spikes, mean_spikes,window_size, slider);
        newParameters(9).predicted_angle = predicted_angle;
        
%         if test_data.true_angle == predicted_angle
%         else
%             predicted_angle
%         end

    end
    
    predicted_angle = newParameters(9).predicted_angle;
    
    if length(predicted_angle) > 1
        newParameters(9).predicted_angle = predicted_angle(1);
    end
    

    
    % TODO: Add more stuff here 
    % Predict the angle with a classifier
    % Predict the angle with cosine similarity
    % Make a prediction based on the highest confidence of any given method
    

    
    
%     predicted_angle = test_data.true_angle;



   

    % ===== Predict the hand coordinate ===== %
    % Extract the predictor segment based on the window
    win_start = current_coordinate-delay-75;
    win_end = win_start+150;
    segment = test_data.spikes(:,win_start:win_end);
    % Sum the spikes over window timesteps
    features = sum(segment,2);
    %features = features';
    
    % Use PCA on the features
    predicted_angle = newParameters(9).predicted_angle;
    Xsub = features(newParameters(predicted_angle).pca.idx,:);

    mu = newParameters(predicted_angle).pca.mu;
    coeff = newParameters(predicted_angle).pca.coeff;
    x_test_pca = (Xsub'-mu)*coeff;
    
    % Append velocity feature to the PCA vector
%     x_test_pca = [x_test_pca velocity];
    
    % Predict the coordinates
    x_regressor = newParameters(predicted_angle).reg_x.regressor;
    y_regressor = newParameters(predicted_angle).reg_y.regressor;
    
    x_pred = x_regressor.predict(x_test_pca);
    y_pred = y_regressor.predict(x_test_pca);
    
    % access the average of the current coordinate
    x_med = newParameters(10).means.x(predicted_angle,current_coordinate);
    y_med = newParameters(10).means.y(predicted_angle,current_coordinate);
    
    % If the prediction is anomalous, adjust it based on prior knowledge
    % about expected behaviour
    threshold = 4;
    contribution=0.79;
    if abs(x_pred-x_med) > threshold
        x_pred = contribution*x_med+(1-contribution)*x_pred;
    end
    if abs(y_pred-y_med) > threshold
        y_pred = contribution*y_med+(1-contribution)*y_pred;
    end
    
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




function predicted_angle = classify_angle(spikes, mean_spikes,window_size, slider)


    % IMPORTANT! Try out different starting positions again properly

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