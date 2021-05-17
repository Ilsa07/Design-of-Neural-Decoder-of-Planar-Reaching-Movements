 
load('monkeydata0.mat');

window_size = 100;
slider = 20;
delay = 150;
sp_start = 320 - delay; 


[mean_spikes,mean_x_displacement,mean_y_displacement] = compute_spike_features(trial,98,window_size,slider,sp_start,delay); % Compute the mean spikes


features = zeros(10,10000); % This will contain the training features
responses_x = zeros(1,10000); % This will contain the corresponding x displacement responses
responses_y = zeros(1,10000); % This will contain the corresponding y displacement responses
feature_idx = 1;

for t = 1:100 
    for a = 1:8
        sp_end = length(trial(t,a).spikes)-delay; % Get the end of the current trial
        segment_idx = 1;
        for w = sp_start:slider:sp_end % go over all segments
            
            curr_spikes = sum(trial(t,a).spikes(:,w-window_size:w),2); % Find the current spikes
            curr_x_coord = trial(t,a).handPos(1,w+delay); % Find the current x coordinate
            curr_y_coord = trial(t,a).handPos(2,w+delay); % Find the current y coordinate
            prev_x_coord = trial(t,a).handPos(1,w+delay-20); % Find the previous x coordinate
            prev_y_coord = trial(t,a).handPos(2,w+delay-20); % Find the previous y coordinate
            % Response displacements
            displacement_x = curr_x_coord - prev_x_coord; % Find displacements
            displacement_y = curr_y_coord - prev_y_coord;  
            % Feature displacements
            prev_prev_x_coord = trial(t,a).handPos(1,w+delay-40); % For finding displacement that we will use to predict the current displacement
            prev_prev_y_coord = trial(t,a).handPos(2,w+delay-40);
            feature_displacement_x = prev_x_coord - prev_prev_x_coord; % Find the displacement in the previous 20ms step
            feature_displacement_y = prev_y_coord - prev_prev_y_coord;

            diff_mean = zeros(8,1);
            for m = 1:8
                % Compute the difference of the current spike against all
                % angles
                comparison_spikes = mean_spikes(:,m,segment_idx);
                diff = curr_spikes - comparison_spikes;
                diff_mean(m) = sum(abs(diff));

                
            end
            segment_idx = segment_idx + 1;
            % Use differences as features for prediction
            features(1:8,feature_idx) = diff_mean;
            
            features(9,feature_idx) = feature_displacement_x; % Append the x displacement as a feature
            features(10,feature_idx) = feature_displacement_y; % Append the y displacement as a feature
            
            responses_x(feature_idx) = displacement_x;
            responses_y(feature_idx) = displacement_y;

            feature_idx = feature_idx+1;
        end
      
    end
end

x_train = features(:,1:10000)';
y_train = responses_x(1:10000)';

x_test = features(:,10001:end)';
y_ground_truth = responses_x(:,10001:end)';

writematrix([x_train y_train], 'reg_train.csv');


regressionTree = fitrtree(...
    x_train, ...
    y_train, ...
    'MinLeafSize', 4, ...
    'Surrogate', 'off');

y_prediction = regressionTree.predict(x_test);


% hold on
% plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
% plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')



% Compute mean spikes and displacements for each 20ms segment 
function [mean_spikes,mean_x_displacement,mean_y_dislpacement] = compute_spike_features(trial,neur_count, window_size, slider,sp_start,delay)
    spike_counts = zeros(neur_count,8,40); % Count the total number of spikes for each angle
    x_displacement = zeros(8,40);
    y_displacement = zeros(8,40);
    for t = 1:length(trial)
        for a = 1:8
            sp_end = length(trial(t,a).spikes)-delay; % Get the end of the current trial
            w_counter = 1;
            for w = sp_start:slider:sp_end
                spike_counts(:,a,w_counter) = spike_counts(:,a,w_counter) + ...
                    sum(trial(t,a).spikes(:,w-window_size:w),2); % Total spikes in the current trial_angle 
                % Compute the displacement starting from 320 - 300 ms
                % coordinates
                pos = trial(t,a).handPos(1:2,w+delay); % Get the handposition at current ms
                pos_last_step = trial(t,a).handPos(1:2,w+delay-20); % Get the handposition 20ms earlier
                
                x_displacement(a,w_counter) = x_displacement(a,w_counter) +...
                    (pos(1) - pos_last_step(1)); % Compute displacement in x coordinate
                y_displacement(a,w_counter) = y_displacement(a,w_counter) +...
                    (pos(2) - pos_last_step(2)); % Compute displacement in y coordinate
                w_counter = w_counter +1; % Increment the segment index
            end
        end
    end
    mean_spikes = spike_counts / length(trial); % Divide the spikes with the number of trials to get the mean
    mean_x_displacement = x_displacement / length(trial); % get mean displacements for x
    mean_y_dislpacement = y_displacement / length(trial);
end
