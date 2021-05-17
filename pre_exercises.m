% Unpack the Data from one Trial
% How to reference the data: First index is the trial number and the second
% intex is the orientation For example this references the first trial in
% orientation 1 (There are 100 trials for each 8 orientations)
trial(1,1).spikes;

% Function Calling Examples
% *************************
% Exercise 1
%raster_plot_single_trial(trial(1,1))
% Exercise 2
%raster_for_single_neuron_in_direction(trial, 1, 1)
% Exercise 3
%psth_for_trial(trial, 1, 1)
% Exercise 4
%plot_all_trajectories(trial)
% Exercise 5
%get_tuning_curves(trial)


% Function Code
% *************
% Exercise 1: Create a raster plot of a single trial along with the
% trajectory
% *****************************************************************
function raster_plot_single_trial(trial)
    hand_position = trial.handPos;
    x_axis_scale = 1:1:size(trial.spikes, 2);
    figure
    subplot(3,2,1)
    % Create a raster plot for a single measurement set
    % For each neuron, plot the spikes as th enumber of the neuron
    % so you can have neuron number on the y axis
    for n=1 : size(trial.spikes, 1)
        % Change the 0 values to NaN, so they will not be plotted and
        % change the spikes (1-s) to the number of the neuron, so the plot
        % will not overlap
        y_values = trial.spikes(n,:);
        y_values(y_values==0)=nan;
        y_values(y_values==1)=n;
        scatter(x_axis_scale,y_values, 'filled','b')
        hold on
    end
    % Add Legends and show the final plot
    title('Roster Plot for one Trial')
    xlabel('Time (ms)') 
    ylabel('Neuron Number')
    
    % Plot the 3D Trajectory
    subplot(3,2,2)
    plot3(hand_position(1,:), hand_position(2,:), hand_position(3,:))
    title('Trajectory During the Trial')
    xlabel('X Coordinate') 
    ylabel('Y Coordinate')
    zlabel('Z Coordinate')
    
    % Plot each part (x,y,z) of the trajectory
    % ****************************************
    subplot(3,2,3)
    plot(x_axis_scale, hand_position(1,:))
    xlabel('Time (ms)')
    ylabel('X Coordinate')
    
    subplot(3,2,4)
    plot(x_axis_scale, hand_position(2,:))
    xlabel('Time (ms)')
    ylabel('Y Coordinate')
    
    subplot(3,2,5)
    plot(x_axis_scale, hand_position(3,:))
    xlabel('Time (ms)')
    ylabel('Z Coordinate')
    hold off
end

% Exercise 2: raster plot for one neural unit over many trials, one
% direction
% ***************************************************************
function raster_for_single_neuron_in_direction(trials, neuron, direction)
    % trials: struct provided in the assignment
    % neuron: neuron number to be looked at (1-98)
    % direction: direction number to be looked at (1-8)
    figure
    subplot(2,2,1)
    % Iterate through all the trials in a direction
    for trial=1 : size(trials, 1)
        % Select the current trial
        current_trial = trials(trial, direction);
        
        % Unpack and format the spikes
        y_values = current_trial.spikes(neuron,:);
        x_axis_scale = 1:1:size(current_trial.spikes, 2);
        y_values(y_values==0)=nan;
        y_values(y_values==1)=trial;
        
        % Plot the spikes on the same graph
        scatter(x_axis_scale,y_values, 'filled','b')
        hold on
    end
    title('Roster Plot for a single neuron across all trials in a given direction')
    xlabel('Time (ms)') 
    ylabel('Trial number')
    
    % Plot the X Coordinates
    for trial=1 : size(trials, 1)
        % Select the trial and create time scale
        current_trial = trials(trial, direction);
        x_axis_scale = 1:1:size(current_trial.spikes, 2);
        
        % Unpack the X, Y, Z Coordinates
        hand_position = current_trial.handPos;
        
        % Plot the X, Y, Z Coordinates for the trials
        subplot(2,2,2)
        plot(x_axis_scale, hand_position(1,:), "b")
        hold on
    end
    title('X-axis movement During the Trial')
    xlabel('Time (ms)') 
    ylabel('X Coordinate')
    
    % Plot the Y Coordinates
    for trial=1 : size(trials, 1)
        % Select the trial and create time scale
        current_trial = trials(trial, direction);
        x_axis_scale = 1:1:size(current_trial.spikes, 2);
        
        % Unpack the X, Y, Z Coordinates
        hand_position = current_trial.handPos;
        
        % Plot the X, Y, Z Coordinates for the trials
        subplot(2,2,3)
        plot(x_axis_scale, hand_position(2,:), "b")
        hold on
    end
    title('Y-axis movement During the Trial')
    xlabel('Time (ms)') 
    ylabel('Y Coordinate')
    
    % Plot the Z Coordinates
    for trial=1 : size(trials, 1)
        % Select the trial and create time scale
        current_trial = trials(trial, direction);
        x_axis_scale = 1:1:size(current_trial.spikes, 2);
        
        % Unpack the X, Y, Z Coordinates
        hand_position = current_trial.handPos;
        
        % Plot the X, Y, Z Coordinates for the trials
        subplot(2,2,4)
        plot(x_axis_scale, hand_position(3,:), "b")
        hold on
    end
    title('Z-axis movement During the Trial')
    xlabel('Time (ms)') 
    ylabel('Z Coordinate')
    hold off
end

% Exercise 3: Peri-Stimulus Time Histogram (histogram of neurons firing)
% **********************************************************************
function psth_for_trial(trials, trial_index, direction)
    current_trial = trials(trial_index, direction);
    spikes = current_trial.spikes();
    
    % First Plot the Histogram
    x_axis_scale = 1:1:size(spikes, 2);
    figure;
    subplot(2,1,1);
    % Create a raster plot for a single measurement set
    % For each neuron, plot the spikes as th enumber of the neuron
    % so you can have neuron number on the y axis
    for n=1 : size(spikes, 1)
        % Change the 0 values to NaN, so they will not be plotted and
        % change the spikes (1-s) to the number of the neuron, so the plot
        % will not overlap
        y_values = spikes(n,:);
        y_values(y_values==0)=nan;
        y_values(y_values==1)=n;
        scatter(x_axis_scale,y_values, 'filled','b');
        hold on
    end
    % Add Legends and show the final plot
    title('Roster Plot for one Trial');
    xlabel('Time (ms)') ;
    ylabel('Neuron Number');
    
    % Create a histogram for each time step of all the neurons
    histogram_data = zeros(1,size(spikes, 2));
    subplot(2,1,2);
    for time=1 : size(spikes, 2)
        for neuron=1 : size(spikes, 1)
            histogram_data(time) = histogram_data(time)+spikes(neuron, time);
        end
    end
    % Plot the histogram
    bar(x_axis_scale, fnval(csaps(x_axis_scale,histogram_data),x_axis_scale), 1);
    title('Bar Plot of Neural Activity');
    xlabel('Time (ms)');
    ylabel('Number of Neurons Firing');
    hold off
end

% Exercise 4: Plot all of the hand trajectories at once
% *****************************************************
function plot_all_trajectories(trials)
    % For each direction
    for direction=1 : size(trials, 2)
        % For each trial
        for trial=1 : size(trials, 1)
            % Unpack the selected trial and hend position
            current_trial = trials(trial, direction);
            hand_position = current_trial.handPos;
            plot3(hand_position(1,:), hand_position(2,:), hand_position(3,:), 'b')
            hold on
        end
    end
    % Show the final plot with proportional axis
    axis equal
    title('All of the Trajectories plotted at once')
    xlabel('X Coordinate') 
    ylabel('Y Coordinate')
    zlabel('Z Coordinate')
end

% Exercise 5: Plot the tuning curve of individual neural units by plotting 
% the firing rate averaged across time and trials as a function of movement direction
% ***********************************************************************************
function get_tuning_curves(trials)
    firing_rates = zeros(98,8)
    % For each direction
    for direction=1:size(trials, 2)
        % For each Trial in the direction
        for trial=1:size(trials, 1)
            current_trial = trials(trial, direction);
            spikes = current_trial.spikes;
            % For each neuron
            for neuron=1:size(spikes, 1)
                firing_rates(neuron, direction) = firing_rates(neuron, direction) + sum(spikes(neuron,:));
            end
        end
    end
    % Average out the firing rates
    firing_rates = firing_rates./(672*100);
    
    % Plot the averaged firing rate of each neuron vs. the direction for
    % each direction
    for n=1:size(trials, 2)
        figure(n)
        bar(firing_rates(:,n))
        title(sprintf('Tuning Curve for Neurons in Direction %i', round(n)))
        xlabel('Neuron number (array indedx)') 
        ylabel('Average Activation Across Trials')
        grid
    end
end





