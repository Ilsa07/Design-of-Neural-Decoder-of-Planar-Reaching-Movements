load('monkeydata0.mat');

spike_counts = zeros(98,8); % Count the total number of spikes for each angle
spike_max = zeros(98,8); % Keep track of the max number of spikes 
spike_min = zeros(98,8) + 1000;
sp_start = 1;
sp_end = 320;


for t = 1:100
    for a = 1:8
        for n = 1:98
            neur_sum = sum(trial(t,a).spikes(n,sp_start:sp_end)); % Total spikes in the current trial_angle for a specific neuron
            spike_counts(n,a) = spike_counts(n,a) + neur_sum;
            
            % Update the maximum spike count for a certain neuron
            if neur_sum > spike_max(n,a) % If the max spike is higher than current for that particular trial and angle
                spike_max(n,a) = neur_sum;
            end
            
            % Update the minimum spike counte for a certian neuron
            if neur_sum < spike_min(n,a)
                spike_min(n,a) = neur_sum;
            end
        end
    end
end

mean_spikes = spike_counts / 100; % Divide the spikes with the number of trials to get the mean

% Remove spikes with high variance
range = spike_max-spike_min;
good_idx = find(range(:,4) < 11);


mean_spikes_rounded = round(mean_spikes); % round 'em

% Angles of particular interest
mean_4 = mean_spikes(:,4);
mean_5 = mean_spikes(:,5);
diff_4_5 = mean_5 - mean_4;
biggest_pos_diff = find(diff_4_5 >0.5); % Get the indices of neurons that are greater than x in difference
biggest_neg_diff = find(diff_4_5 <-0); % Get the indices of neurons that are less than x in difference
total_diff = [biggest_pos_diff; biggest_neg_diff];


% find neurons that on average are more active in angle 4
neg_idx = find(diff_4_5 <-0.9); % Get the indices of neurons that are less than x in difference



% Calculate the distance of the current prediction from the important
% neurons
comparison_matrix = mean_spikes_rounded(total_diff,4:5);

% Calculate the distance of the current prediction from each of these
% neurons




% TEST 
difference = zeros(98,1); % Initialize difference arrays for each angle
abs_diff = zeros(98,8);
sum_diff = zeros(8,1);
total_predictions = 8*100;
correct_predictions = 0;

for t = 1:100
    for a = 1:8
        current_trial_spikes = sum(trial(t,a).spikes(:,sp_start:sp_end),2); % Get the current angle
        for m = 1:8 % Compare it to each mean
            difference = current_trial_spikes - mean_spikes_rounded(:,m); % Compare against each mean
            abs_diff(:,m) = abs(difference); % Find the absolute differences
            summed = sum(abs_diff,1);
            sum_diff(m) = summed(m); % Sum the differences into a single comparable scalar value
        end
        

        
        
        % sum_diff contains all the differences from angle 1 to 8, find the smallest difference 
        predicted_angle = find(sum_diff == min(sum_diff));
        
        
        % If the angle is 4 or 5, compare them with the most important
        % neurons
        predicted_angle = predicted_angle(1);
        if predicted_angle == 4 || predicted_angle == 5
            current_spikes = current_trial_spikes(neg_idx); % find the neurons that fire more often in angle 4
            summed_unknown = sum(current_spikes);
            summed_spikes_4 = sum(mean_4(neg_idx)); % sum of angle 4 spikes in that index
            summed_spikes_5 = sum(mean_5(neg_idx));
            dist_from_4 = abs(summed_unknown-summed_spikes_4);
            dist_from_5 = abs(summed_unknown-summed_spikes_5);
            % if difference is too great, discard neuron from total
            % calculation

            
            % Pick the angle that has the shortest distance
            if dist_from_4 < dist_from_5
                predicted_angle = 4;
            else
                predicted_angle = 5;
            end
            
        end

        
        
        correct_pred = a; % Correct prediction is the current angle
        if correct_pred == predicted_angle % If the angle is correctly predicted
            correct_predictions = correct_predictions+1;
        else
            [t, a]
        end
        
    end
end

correct_rate = correct_predictions / total_predictions % Calculate the percentage of correct predictions from 800 trials