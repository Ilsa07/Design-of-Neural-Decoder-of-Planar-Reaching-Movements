clear all; close all;
clear all; close all; clc;
load('monkeydata0.mat');

%%

[dim1,dim2]=size(trial);
change_pos=zeros(1,449,dim1);
change_pos_avg=zeros(1,449,dim2);
spikes_pos_total=zeros(1,450,dim1);
spikes_pos_avg=zeros(1,450,dim2);
for q=1:dim2
    for i=1:dim1
        %obtain the change of position
        change_pos(:,:,i)=sum(diff(trial(i,q).handPos(:,101:550)'),2);
        %sum over all the neurons to get the overall brain activity in a
        %trial
        spikes_pos_total(:,:,i)=sum(trial(i,q).spikes(:,101:550),1);
    end
%median brain activity and change of position for an angle
change_pos_avg(:,:,q)=median(change_pos,3);
spikes_pos_avg(:,:,q)=median(spikes_pos_total,3);
%you can also try the mean value
figure;
subplot(211)
plot(change_pos_avg(:,:,q))
subplot(212)
plot(spikes_pos_avg(:,:,q))
end
