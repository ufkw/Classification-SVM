function  Function_fitcecoc_01(X_in,Y_in,coding,training_to_test_ratio, name) %meas,species)
%Function_fitcecoc_01 
%   X_in attributes
%   Y_in values
%   coding coding = {'onevsone' 'onevsall'};
%   training_to_test_ratio  this determines how data in X_in and Y_in is
%                           split into training set and test sets.  0.1/0.9,
%                           0.2/0.8 etc

[n_dataset,p] = size(X_in);

for jj = 1:length(coding)  % loop through analysis using one vs one and one vs all
    
    for ii = 1:length(training_to_test_ratio)  % loop through different size training (and test) sets
        rng(1);  % this ensures that same random sampling is used if code ran more than once
        n_trainingset=round(training_to_test_ratio(ii)*n_dataset);  % number of samples in trainng set
        idx_trainingset=sort(randsample(n_dataset,n_trainingset,0))'; %index of samples in training set
        idx_testset=setdiff(1:n_dataset,idx_trainingset); % index of samples in test set.
        X=X_in(idx_trainingset,:);  % training X's
        Y=Y_in(idx_trainingset);    % training Y's
        temp=templateSVM();
        
        Mdl = fitcecoc(X,Y,'Coding',coding{jj},'learners',temp);  % training
        
        label_trainset=predict(Mdl,X_in(idx_trainingset,:));  % predicted labels (or classes) with traiing set as input
        [C]=confusionmat(Y_in(idx_trainingset),label_trainset); % confusion matrix.  this can be plotted with plotconfusion
        classification_error_trainingset=(sum(sum(C))-trace(C))/sum(sum(C)); % classification error on training set
        label_testset=predict(Mdl,X_in(idx_testset,:));  % predicted labels with test set as input
        [C]=confusionmat(Y_in(idx_testset),label_testset);  % confusion matrix.  this can be plotted with plotconfusion
        figure;
        plotconfusion(categorical(Y_in(idx_testset)),categorical(label_testset), [name '.' num2str(ii) '.SVM']);
        
        disp([ coding{jj} '  Training to test ratio = ' num2str(training_to_test_ratio(ii)) '   Classification error test set = ' num2str((sum(sum(C))-trace(C))/sum(sum(C))) ' error training=' num2str(classification_error_trainingset)])   
        
    end
    disp(' ')
end


%cross validation

lambda = 0:1:10;
for p = 1:length(lambda)
    rng(3);
    c = cvpartition(size(X_in, 1), 'KFold', 10);

    for k=1:1:10
     
        idx_validation_set = test(c,k);
        idx_trainingset=training(c,k);
        X=X_in(idx_trainingset,:);  % training X's
        Y=Y_in(idx_trainingset);    % training Y's
        
        Mdl = fitcecoc(X,Y,'Coding',coding{jj},'learners',temp);  % training
        class_label=predict(Mdl,X_in(idx_validation_set, :));  % predicted labels (or classes) with training set as input
        [C]=confusionmat(Y_in(idx_validation_set),class_label); % confusion matrix.  this can be plotted with plotconfusion
        CE(k)=(sum(sum(C))-trace(C))/sum(sum(C));
    end
    CE_lambda(p) = mean(CE);
end
[CE_optimum_lambda, i_CE] = min(CE_lambda);
optimum_lambda=lambda(i_CE);

disp('Average classification error from cross validation; ')
disp(CE_optimum_lambda)
disp('Optimum lambda from cross validation: ')
disp(optimum_lambda)


end
