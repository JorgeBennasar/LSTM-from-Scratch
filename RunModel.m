function [final_param, cost_train, subsets] = RunModel(data, param_model)

% Training and testing data selection
if param_model.mode == 1
    m_train = 8*param_model.train_samples_per_target;
    time = size(data.X,3);
    
    x_train = zeros(size(data.X,1),m_train,time);
    y_train = zeros(2,m_train,time);
    index_train = zeros(1,m_train);
    x_test = zeros(size(data.X,1),size(data.X,2)-m_train,time);
    y_test = zeros(2,size(data.X,2)-m_train,time);
    index_test = zeros(1,size(data.X,2)-m_train);
    
    target_counter = zeros(1,8);
    counter_train = 0;
    counter_test = 0;
    for i = 1:size(data.X,2)
        for j = 1:8
            if j == data.targets(i) 
                if target_counter(j) < param_model.train_samples_per_target
                    target_counter(j) = target_counter(j) + 1;
                    counter_train = counter_train + 1;
                    x_train(:,counter_train,:) = data.X(:,i,:);
                    for k = 1:2
                        for l = 1:time
                            y_train(k,counter_train,l) = data.Y(k,i,l);
                        end
                    end
                    index_train(counter_train) = i;
                else
                    counter_test = counter_test + 1;
                    x_test(:,counter_test,:) = data.X(:,i,:);
                    for k = 1:2
                        for l = 1:time
                            y_test(k,counter_test,l) = data.Y(k,i,l);
                        end
                    end
                    index_test(counter_test) = i;
                end
            end
        end
    end
elseif param_model.mode == 2
    x_train = data.x_train;
    y_train = data.y_train;
end

% Training
[final_param, cost_train] = LSTM_train(x_train, y_train, ...
    param_model.mini_batch_size, param_model.num_epochs, ...
    param_model.n_hidden, param_model.beta_1, param_model.beta_2, ...
    param_model.epsilon, param_model.learning_rate, ...
    param_model.optimization, param_model.lambda, ...
    param_model.stop_condition, param_model.connectivity, ...
    param_model.network_model,param_model.links, ...
    param_model.correlation_reg);

% Subsets:

if param_model.mode == 1
    subsets.train.X = x_train;
    subsets.train.Y = y_train;
    subsets.train.index = index_train;
    subsets.test.X = x_test;
    subsets.test.Y = y_test;
    subsets.test.index = index_test;
elseif param_model.mode == 2
    subsets = [];
end

end