%
% random forest as a preference model
% 

metadata = importdata('/home/xudong/Codes/PrefLearnLib_Generator/PrefLearnLib/UCI/CarEvaluation/domain_description.txt');

num_features = size(metadata, 1);
features_names = cellfun(@(x) x(1:find(x==':')-1), metadata, 'uni', 0);
feature_values = cellfun(@(x) strsplit(x(find(x==':')+1:end), ','), metadata, 'uni', 0);

mapping_backward = cell(num_features,1);
mapping_forward = cell(num_features,1);
for ix = 1:num_features
  mapping_backward{ix} = containers.Map;
  mapping_forward{ix} = containers.Map;
  for jx = 1:numel(feature_values{ix})
    mapping_forward{ix}(feature_values{ix}{jx}) = jx;
    mapping_backward{ix}(num2str(jx)) = feature_values{ix}{jx};
  end
end

% dataset of cars
car_data = importdata('/home/xudong/Codes/PrefLearnLib_Generator/PrefLearnLib/UCI/CarEvaluation/outcomes.csv');
car_data = car_data(2:end, :);
num_cars = size(car_data,1);
car_data = cellfun(@(x) strsplit(x(find(x==',')+1:end),','), car_data, 'uni', 0);

new_car_data = cell(num_cars,1);
for ix = 1:num_cars
  for jx = 1:num_features
    new_car_data{ix}(jx) = mapping_forward{jx}(car_data{ix}{jx});
  end
end
new_car_data = cat(1, new_car_data{:});

% preferences, first column is preferred
preferences = importdata('/home/xudong/Codes/PrefLearnLib_Generator/PrefLearnLib/UCI/CarEvaluation/strict_examples.csv');
preferences = preferences.data; preferences = preferences(:, 2:end);
num_samples = size(preferences, 1);

% convert preferences into features/labels
features = nan(num_samples, num_features*2); labels = nan(num_samples, 1);
for ix = 1:num_samples
  % turn cars into features
  car1_feat = new_car_data(preferences(ix, 1), :);
  car2_feat = new_car_data(preferences(ix, 2), :);
  % randomly order them and assign label
  labels(ix) = round(rand);
  if labels(ix) == 1
    features(ix, :) = [car1_feat car2_feat];
  else
    features(ix, :) = [car2_feat car1_feat];
  end
end

%% set sample_sizes_arry for experiments

num_sample_sizes = 34;
sample_sizes_array = zeros(1,num_sample_sizes);
for ix = 1:num_sample_sizes
    if (ix >= 1) && (ix < 10)
        sample_sizes_array(ix) = ix;
    else
        sample_sizes_array(ix) = 10 + (ix-10) * 10;
    end
end


%% load and convert examples

rep = 20;
C1 = {'SampleSize', 'DT-Training%', 'DT-Testing%'};
for ix = 1:numel(sample_sizes_array)
    C2 = cell(1,3);
    sum_accuracy_train = 0;
    sum_accuracy_test = 0;
    for jx = 1:rep
        % randomly split
        rp = randperm(num_samples);
        % train_inds = rp(1:round(num_samples*.7));
        % test_inds = rp(round(num_samples*.7)+1:end);
        train_inds = rp(1:sample_sizes_array(ix));
        test_inds = rp(sample_sizes_array(ix)+1:end);
        
        train_data = features(train_inds, :);
        train_labels = labels(train_inds);
        test_data = features(test_inds, :);
        test_labels = labels(test_inds);
        
        % train_data = train_data(:, 1:6)-train_data(:,7:end);
        % test_data = test_data(:, 1:6)-test_data(:,7:end);
        
        % train_data_flipped = [train_data(:, 7:end) train_data(:, 1:6)];
        % train_labels_flipped = ~train_labels;
        % train_data_doubled = [train_data; train_data_flipped];
        % train_labels_doubled = [train_labels; train_labels_flipped];
        
        num_trees = 1;
        B = TreeBagger(num_trees, train_data, train_labels,'OOBPred','On');
        
        preds_train = predict(B, train_data);
        preds_train = cellfun(@(x) str2double(x), preds_train);
        preds_test = predict(B, test_data);
        preds_test = cellfun(@(x) str2double(x), preds_test);
        
        sum_accuracy_train = sum_accuracy_train + sum(preds_train == train_labels)/numel(train_labels);
        sum_accuracy_test = sum_accuracy_test + sum(preds_test == test_labels)/numel(test_labels);
        
        % view(B.Trees{1}, 'mode', 'graph')
    end
    C2{1,1} = sample_sizes_array(ix);
    C2{1,2} = sum_accuracy_train/rep;
    C2{1,3} = sum_accuracy_test/rep;
    C1 = [C1;C2];
%     celldisp(sum_accuracy_testing/rep)
end

% write results to file
fid = fopen('./sum.csv', 'w') ;
fprintf(fid, '%s,', C1{1,1:end-1}) ;
fprintf(fid, '%s\n', C1{1,end}) ;
fclose(fid) ;
dlmwrite('./sum.csv', C1(2:end,:), '-append') ;

