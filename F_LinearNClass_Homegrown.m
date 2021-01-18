function F_LinearNClass_Homegrown(X_in,Y_in,training_to_test_ratio, name)
%is one vs all

A=unique(Y_in);
A1=categorical(Y_in);
A2(1,1:length(Y_in))=0;
if iscell(A)
    for ii=1:length(A)
        A2(A1==A{ii})=ii;   % converts categories to numerical 1, 2, 3, ...
    end
    Y_in=A2';
end
nClasses=length(A);


[n_dataset,p] = size(X_in);
for i_t = 1:length(training_to_test_ratio)  % loop through different size training (and test) sets
    rng(1);  % this ensures that same random sampling is used if code ran more than once
    n_trainingset=round(training_to_test_ratio(i_t)*n_dataset);  % number of samples in training set
    idx_trainingset=sort(randsample(n_dataset,n_trainingset,0))'; %index of samples in training set
    idx_testset=setdiff(1:n_dataset,idx_trainingset); % index of samples in test set.
    X=X_in(idx_trainingset,:);  % training X's
    Y=Y_in(idx_trainingset);    % training Y's
    X=[X ones(length(X), 1)*1];
    
    rng(1);  % this ensures that same random sampling is used if code ran more than once
    n_part=round(.8*length(X));  % number of samples in training set, the SVM does an 80/20 ratio so we will imitate
    idx_firstpart=sort(randsample(length(X),n_part,0))'; %index of samples in training set
    idx_secondpart=setdiff(1:length(X),idx_firstpart); % index of samples in test set.
    X1=X(idx_firstpart,:);
Y1=Y(idx_firstpart);
    X2=X(idx_secondpart,:);
    Y2=Y(idx_secondpart);
    
    
    lm=0:1:10;
    for ii=1:length(lm)
        lambda=ones(nClasses,1)*0+lm(ii);
        [classification_error(ii),values,W]=F_LinearNClassifier(lambda,X1,Y1,X2,Y2,nClasses);
    end
    [CE,i_CE]=min(classification_error);
    lambda=ones(nClasses,1)*0+lm(i_CE);
    [CE,values,W]=F_LinearNClassifier(lambda,X1,Y1,X2,Y2,nClasses);
    
    [label_trainset]=F_DetermineLabel(X_in(idx_trainingset,:),W);
    [C]=confusionmat(Y_in(idx_trainingset),label_trainset); % confusion matrix.  this can be plotted with plotconfusion
    classification_error_trainingset=(sum(sum(C))-trace(C))/sum(sum(C)); % classification error on training set
    
    [label_testset]=F_DetermineLabel(X_in(idx_testset,:),W);
    figure;
    [C]=confusionmat(Y_in(idx_testset),label_testset);  % confusion matrix.  this can be plotted with plotconfusion
    plotconfusion(categorical(Y_in(idx_testset))',categorical(label_testset), [name '.' num2str(i_t) '.Homegrown']);
    
    
    disp([  '  Training to test ratio = ' num2str(training_to_test_ratio(i_t)) '   Classification error test set = ' num2str((sum(sum(C))-trace(C))/sum(sum(C))) ' error training=' num2str(classification_error_trainingset) ' lm=' num2str(lambda(1))])
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
        X=[X ones(length(X), 1)*1];
        
        [~,~,W] = Function_createW(X,Y,lambda, nClasses);
        [class_label]=F_DetermineLabel(X_in(idx_validation_set,:),W);
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

    function [class_label]=F_DetermineLabel(X1,W1)
        [~,class_label]=max([[X1 ones(length(X1),1)]*W1']');
    end

    function [CE,values,W]=F_LinearNClassifier(lambda,X,Y,X2,Y2,nClasses)
        for i_classes=1:nClasses
            y(1:length(Y))=0;y(Y==i_classes)=1;
            W(i_classes,:)=(X'*X+lambda(i_classes))\(X'*y');
        end
        [~,values]=max([X2*W']');
        CE=sum(Y2~=values')/length(Y2);
    end
end

