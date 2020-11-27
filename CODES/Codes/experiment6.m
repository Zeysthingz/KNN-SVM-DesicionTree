clear all; close; clc;

%TÝME CALCULATÝNG FOR FÝSHERÝRÝS
load  fisheriris;

X = meas(:,:);
Y = species;


[m,n] = size(X);
P = 0.80;
idx = transpose(randperm(m));            %Spilitting data 
Training_X = X(idx(1:round(P*m)),:);
Training_Y= Y(idx(1:round(P*m)),:);
Testing_X = X(idx(round(P*m)+1:end),:);
Testing_Y = Y(idx(round(P*m)+1:end),:);
y=numel(unique(Training_Y));

%Timimg for SVM 

%TicTocValue = 0;
tic;

%training tine
temp = templateSVM('KernelFunction', 'linear');
model = fitcecoc(Training_X, Training_Y, 'Learner', temp);
Training_Time1 = toc;
%disp(['Training time: ', num2str(TicTocValue),'sec']);

%prediction time 
tic;
y_pred = predict(model, Testing_X);
Prediction_Time1 =toc;
%disp(['Prediction time: ', num2str(TicTocValue),'sec']);

result_of_matrice= confusionmat(Testing_Y, y_pred);
[Metric] = metrics(result_of_matrice,y);
F1_score=Metric{{'Average'},'F1'};


%Timing for KNN 
tic;
model = ClassificationKNN.fit(Training_X,Training_Y,'NumNeighbors',13,'Distance','minkowski');
TicTocValue3 =toc;

%Prediction time
tic;
predicted = predict(model, Testing_X);
PredictionTime3 = toc;

result_of_matrice= confusionmat(Testing_Y, predicted);
[Metric] = metrics(result_of_matrice,y);     %Taking F1
F1=Metric{{'Average'},'F1'};

%Timing for desicion tree


tic;
model = fitctree(Training_X,Training_Y); 
TicTocValue5 = toc;

%prediction time
tic;
predict2 = predict(model,Testing_X);
PredictionTime6 = toc;

result_of_matrice= confusionmat(Testing_Y, predict2);
[Metric] = metrics(result_of_matrice,y);
F1_score3=Metric{{'Average'},'F1'};


%Creating table

Table1=table(F1_score',Training_Time1',Prediction_Time1','VariableNames',{'F1_Score ','Training_Time_sec','Prediction_Time_sec'},'RowNames',{'SVM Method'});
Table2=table(F1',TicTocValue3',PredictionTime3','VariableNames',{'F1_Score ','Training_Time_sec','Prediction_Time_sec'},'RowNames',{'KNN Method'});
Table3=table(F1_score3',TicTocValue5',PredictionTime6','VariableNames',{'F1_Score ','Training_Time_sec','Prediction_Time_sec'},'RowNames',{'Desicion Tree Method'});

Table=[Table1;Table2;Table3];
fprintf('Results Table for Fisheriris Dataset:\n\n');
disp(Table);


%Plotting
plot(Prediction_Time1,F1_score,'*','MarkerSize',9); xlabel('Prediction Time'); ylabel('F1 Scorre');
hold on
plot(PredictionTime3,F1,'o','MarkerSize',9);
plot(PredictionTime6,F1_score3,'-.dr','MarkerSize',9);

title('Prediction Time -F1 Score Graphic');legend('SVM Met','KNN Met','Desicion Tree Met');

grid minor;


%%
% TÝME CALCULATÝNG FOR IONESPHERE DATASET 

clear all;close;clc;

load ionosphere

[m,n] = size(X);
P = 0.80;
idx = transpose(randperm(m));            %Spilitting data 
Training_X = X(idx(1:round(P*m)),:);
Training_Y= Y(idx(1:round(P*m)),:);
Testing_X = X(idx(round(P*m)+1:end),:);
Testing_Y = Y(idx(round(P*m)+1:end),:);
y=numel(unique(Training_Y));



%Using SVM

tic;
temp = templateSVM('KernelFunction', 'linear');       %Timimg for training
model = fitcecoc(Training_X, Training_Y, 'Learner', temp);
training_time1 = toc;

tic;
y_pred = predict(model, Testing_X);
prediction_time1 =toc;

result_of_matrice= confusionmat(Testing_Y, y_pred);  %imimg for prediction
[Metric] = metrics(result_of_matrice,y);
F1_score1=Metric{{'Average'},'F1'};

%Using KNN 
tic;
model = ClassificationKNN.fit(Training_X,Training_Y,'NumNeighbors',13,'Distance','minkowski');
training_time2 =toc;

tic;

tic;
predicted = predict(model, Testing_X);
prediction_time2 = toc;

result_of_matrice= confusionmat(Testing_Y, predicted);
[Metric] = metrics(result_of_matrice,y);     %Taking F1
F1_score2=Metric{{'Average'},'F1'};

%Using Desicion Tree

tic;
model = fitctree(Training_X,Training_Y);  %Training time
training_time3 = toc;


tic;
predict3 = predict(model,Testing_X); %Prediction time
prediction_time3 = toc;

result_of_matrice= confusionmat(Testing_Y, predict3);
[Metric] = metrics(result_of_matrice,y);     %Taking F1
F1_score3=Metric{{'Average'},'F1'};


%Creating table

Table1=table(F1_score1',training_time1',prediction_time1','VariableNames',{'F1_Score ','Training_Time_sec','Prediction_Time_sec'},'RowNames',{'SVM Method'});
Table2=table(F1_score2',training_time2',prediction_time2','VariableNames',{'F1_Score ','Training_Time_sec','Prediction_Time_sec'},'RowNames',{'KNN Method'});
Table3=table(F1_score3',training_time3',prediction_time3','VariableNames',{'F1_Score ','Training_Time_sec','Prediction_Time_sec'},'RowNames',{'Desicion Tree Method'});

Table=[Table1;Table2;Table3];
fprintf('Results Table for Ionesphere Dataset:\n\n');
disp(Table);

plot(prediction_time1,F1_score1,'*','MarkerSize',9);xlabel('Prediction Time');ylabel('F1 Score');
grid minor;
hold on ;
plot(prediction_time2,F1_score2,'o','MarkerSize',9);
plot(prediction_time3,F1_score3,'-.dr','MarkerSize',9);
title('Ionesphere Dataset F1 Score Prediction Time Graphic');
legend('SVM Method', 'KNN Method', 'Desicion Tree Method');
