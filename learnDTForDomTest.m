%
% random forest as a preference model
% 

%% put the names of datasets in a cell array of strings
datasets_names = {
    'BreastCancerWisconsinDownsampled' 
    'CarEvaluation'
    'CreditApprovalDownsampledFurther' 
    'GermanCreditDownsampledFurther'
    'IonosphereDownsampledFurther' 
    'MammographicMassDownsampled'
    'MushroomDownsampled' 
    'NurseryDownsampledFurther'
    'SpectHeartDownsampledFurther' 
    'TicTacToe'
    'VehicleDownsampledFurther' 
    'WineDownsampled'
};
% datasets_names = {
%     'TicTacToe'
% };

forest_sizes = [1:1:9 10:10:90 100:100:1000];
rules = {
    'TopCluster'
    'Top2Clusters'
    'Top3Clusters'
};

%%
for kx = 1:numel(datasets_names)
    metadata_filename =  '/Users/n01237497/Codes/PrefLearnLibGenerator/PrefLearnLib/UCI/';
    metadata_filename = strcat(metadata_filename,datasets_names(kx));
    metadata_filename = strcat(metadata_filename,'/domain_description.txt');
    
    %     metadata = importdata(char(metadata_filename));
    %     num_features = size(metadata, 1);
    %     features_names_cell = cellfun(@(x) x(1:find(x==':')-1), metadata, 'uni', 0);
    %     feature_values_cell = cellfun(@(x) strsplit(x(find(x==':')+1:end), ','), metadata, 'uni', 0);
    
    num_features = getNumLinesInFile(char(metadata_filename));
    features_names_cell = cell(num_features,1);
    feature_values_cell = cell(num_features,1);
    fid = fopen(char(metadata_filename));
    
    n = 1;
    feature_domain_string = fgetl(fid);
    while ischar(feature_domain_string)
        %     disp(feature_domain_string)
        feature_domain_cell = strsplit(feature_domain_string,':');
        features_names_cell{n} = feature_domain_cell{1};
        domain_cell = strsplit(feature_domain_cell{1,2}, ',');
        feature_values_cell{n} = domain_cell;
        
        n = n + 1;
        feature_domain_string = fgetl(fid);
    end
    
    fclose(fid);
    
    mapping_backward = cell(num_features,1);
    mapping_forward = cell(num_features,1);
    for ix = 1:num_features
        mapping_backward{ix} = containers.Map;
        mapping_forward{ix} = containers.Map;
        for jx = 1:numel(feature_values_cell{ix})
            mapping_forward{ix}(feature_values_cell{ix}{jx}) = jx;
            mapping_backward{ix}(num2str(jx)) = feature_values_cell{ix}{jx};
        end
    end
    
    % dataset
    data_filename =  '/Users/n01237497/Codes/PrefLearnLibGenerator/PrefLearnLib/UCI/';
    data_filename = strcat(data_filename,datasets_names(kx));
    data_filename = strcat(data_filename,'/outcomes.csv');
%     data_strings_cell = importdata(char(data_filename));
    num_data = getNumLinesInFile(char(data_filename));
    data_strings_cell = cell(num_data,1);
    fid = fopen(char(data_filename));
    
    n = 1;
    data_string = fgetl(fid);
    while ischar(data_string)
        data_strings_cell{n} = data_string;
        
        n = n + 1;
        data_string = fgetl(fid);
    end
    
    fclose(fid);

    data_strings_cell = data_strings_cell(2:end, :);
    num_cars = size(data_strings_cell,1);
    data_strings_cell = cellfun(@(x) strsplit(x(find(x==',')+1:end),','), data_strings_cell, 'uni', 0);
    
    new_data = cell(num_cars,1);
    for ix = 1:num_cars
        for jx = 1:num_features
            new_data{ix}(jx) = mapping_forward{jx}(data_strings_cell{ix}{jx});
        end
    end
    new_data = cat(1, new_data{:});
    
    % preferences, first column is preferred
    preferences_filename = '/Users/n01237497/Codes/PrefLearnLibGenerator/PrefLearnLib/UCI/';
    preferences_filename = strcat(preferences_filename,datasets_names(kx));
    preferences_filename = strcat(preferences_filename,'/strict_examples.csv');
    preferences = importdata(char(preferences_filename));
    preferences = preferences.data;
    preferences = preferences(:, 2:end);
    num_samples = size(preferences, 1);
    
    % convert preferences into features/labels
    features = nan(num_samples, num_features*2); 
    labels = nan(num_samples, 1);
    for ix = 1:num_samples
        % turn cars into features
        car1_feat = new_data(preferences(ix, 1), :);
        car2_feat = new_data(preferences(ix, 2), :);
        % randomly order them and assign label
        labels(ix) = round(rand);
        if labels(ix) == 1
            features(ix, :) = [car1_feat car2_feat];
        else
            features(ix, :) = [car2_feat car1_feat];
        end
    end
    
    % learn a decision tree
    train_data = features;
    train_labels = labels;
    tc = fitctree(train_data, train_labels);
    preds_train = predict(tc, train_data);
    accuracy_train = sum(preds_train == train_labels)/numel(train_labels);
    %             view(tc,'Mode','Graph')
    
    for p = 1:numel(rules)
        rep = 10;
        C1 = {'ForestSize', 'DT-Training%', 'DT-DomTesting%'};
        for i = 1:numel(forest_sizes)
            C2 = cell(1,3);
            sum_accuracy_test = 0;
            for j = 0:rep-1
                % build dom_test_preferences for testing
                dom_test_preferences_filename =  '/Users/n01237497/QtProjects/PLPTree2WPM/Exp-AAAI18/CICPForests/';
                dom_test_preferences_filename = strcat(dom_test_preferences_filename, datasets_names(kx));
                dom_test_preferences_filename = strcat(dom_test_preferences_filename, '/');
                dom_test_preferences_filename = strcat(dom_test_preferences_filename, rules(p));
                dom_test_preferences_filename = strcat(dom_test_preferences_filename, '/50samplePerTree/optTestSet');
                dom_test_preferences_filename = strcat(dom_test_preferences_filename, num2str(forest_sizes(i)));
                dom_test_preferences_filename = strcat(dom_test_preferences_filename, '_');
                dom_test_preferences_filename = strcat(dom_test_preferences_filename, num2str(j));
                dom_test_preferences_filename = strcat(dom_test_preferences_filename, '.csv');
                
                dom_test_preferences = importdata(char(dom_test_preferences_filename));
                dom_test_labels = ones(1000,1);
                
                test_data = dom_test_preferences;
                test_labels = dom_test_labels;
                preds_test = predict(tc, test_data);
                sum_accuracy_test = sum_accuracy_test + sum(preds_test == test_labels)/numel(test_labels);
            end
            C2{1,1} = forest_sizes(i);
            C2{1,2} = accuracy_train;
            C2{1,3} = sum_accuracy_test/rep;
            C1 = [C1;C2];
        end

        % write results to file
        result_filename = '/Users/n01237497/Codes/DecisionTreeLearning/';
        result_filename = strcat(result_filename,datasets_names(kx));
        result_filename = strcat(result_filename,'/');
        result_filename = strcat(result_filename,rules(p));
        
        if ~exist(result_filename{1}, 'dir')
            mkdir(result_filename{1});
        end
        
        result_filename = strcat(result_filename,'/sumDT.txt');
        fid = fopen(char(result_filename), 'w') ;
        fprintf(fid, '%s,', C1{1,1:end-1}) ;
        fprintf(fid, '%s\n', C1{1,end}) ;
        fclose(fid) ;
        dlmwrite(char(result_filename), C1(2:end,:), '-append') ;
    end
end
