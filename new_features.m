% Create an optimal specialised regressor

load('monkeydata0.mat');

% First: Find the mean difference features of neurons
% Train the model on the neural activity to predict velocities
% Use the predited velocities to infer the next coordinates

window_size = 80;
slider = 20;
delay = 140;
start_spikes = 320-delay - window_size;

features = compute_spike_features(trial,window_size,slider,delay);
correct_1 = 0;
correct_2 = 0;

for t = 1:length(trial)
    for a = 1:8
        counter = 1;
        combined_features = zeros(2,8); % To store both features for all angles
        
        feature_counter = 1;
        
        predictions = [];
        predictions_2 = [];

        for i = start_spikes:20:320
            
            if i+window_size > 320
                break % break if we are over the limit
            end
            
           
            mean_current_trial = sum(trial(t,a).spikes(:,i:i+window_size),2); % Grab the mean of the current trial of the current segment
            
            
            earlier_idx = i-20; % index for taking a segment from earlier to compute the difference
            seg_earlier = sum(trial(t,a).spikes(:,earlier_idx:earlier_idx+window_size),2); % grab the earlier segment
            current_spike_change = mean_current_trial-seg_earlier;
            

            counter = counter+1;
            
            
            for m = 1:8 % Compare against each angle
                mean_seg = features.means(:,m,counter); % grab the current mean segment from previously computed set
                mean_spike_change = features.spike_change(:,m,counter); % grab the current mean spike change from previously computed set
                feature_1 = mean_seg - mean_current_trial; % Compute the first feature
                feature_1 = sum(abs(feature_1)); % Sum it into a scalar
                
                feature_2 = mean_spike_change-current_spike_change; % Compute the second feature
                feature_2 = sum(abs(feature_2)); % Sum it into a scalar
                
                combined_features(1,m) = combined_features(1,m) + feature_1;
                combined_features(2,m) = combined_features(2,m) + feature_2;
                feature_counter = feature_counter + 1 ;
                
                
            end
            pred_1 = find(combined_features(1,:) == min(combined_features(1,:))); % predict the angle based on first features
            pred_2 = find(combined_features(2,:) == min(combined_features(2,:))); % predict the angle based on first features

            predictions = [predictions pred_1];
            predictions_2 = [predictions_2 pred_2];

        end
        
        duo = combined_features(1,:) + combined_features(2,:);
        
        predicted_angle_1 = find(combined_features(1,:) == min(combined_features(1,:))); % predict the angle based on first features
        predicted_angle_2 = find(combined_features(2,:) == min(combined_features(2,:))); % predict the angle based on second features
        predicted_angle_3 = find(predictions == mode(predictions));
        
        if predicted_angle_1 == a 
            correct_1 = correct_1 + 1;
        end

    end
end

n_trials = 100*8;
accuracy_1 = correct_1/n_trials
% accuracy_2 = correct_2/n_trials

    
    
    
function [features] = compute_spike_features(trial, window_size, slider,delay)
    

    start_spikes = 320-delay - window_size;

    
    features.means = zeros(98,8,40);
    features.spike_change = zeros(98,8,40);
    
    for t = 1:length(trial)
        for a = 1:8      
            counter = 1;
            for i = start_spikes:20:length(trial(t,a).spikes)-delay % Loop until the end of the spikes - delay
                if i+window_size >320
                    break
                end
                
                segment = sum(trial(t,a).spikes(:,i:i+window_size),2); % grab the spike segment
                feature_1 = segment; % feature 1 will be the difference from mean spikes
                
                % Get the segment from an earlier index for subtraction
                % (change in neural activity)
                earlier_idx = i-20;
                segment_earlier = sum(trial(t,a).spikes(:,earlier_idx:earlier_idx+window_size),2);
                
                % Find the change in neural activity
                feature_2 = abs(segment - segment_earlier);
                

                features.means(:,a,counter) = features.means(:,a,counter)  + feature_1;
                features.spike_change(:,a,counter) = features.spike_change(:,a,counter) + feature_2;
                
                counter = counter + 1;
                
                
            end
           
        end
    end
    features.means = features.means /100;
    features.spike_change = features.spike_change / 100;
    
end