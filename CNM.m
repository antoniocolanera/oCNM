classdef (ConstructOnLoad) CNM < dynamicprops
    % CNM  Cluster-based Network Modelling for Reduced-Order Dynamics
    %
    %   CNM implements a data-driven reduced-order model (ROM) using
    %   clustering and variable-order network models. It is designed to
    %   learn and predict the evolution of reduced-order physical systems
    %   from a finite set of state snapshots.
    %
    %   Features:
    %     - Clustering of reduced-state snapshots (via K-means++)
    %     - Variable-order network transition modeling
    %     - Probabilistic time-aware state evolution
    %     - Prediction via stochastic sampling of cluster sequences
    %     - State reconstruction through centroid interpolation
    %
    %   USAGE:
    %     model = CNM(reduced_state, dt, n_clusters, model_order)
    %     model = CNM(reduced_state, dt, n_clusters, model_order, interp_method)
    %
    %   INPUTS:
    %     reduced_state : (nq x Nt) matrix of reduced state snapshots
    %     dt            : scalar, time step between snapshots
    %     n_clusters    : integer, number of clusters
    %     model_order   : integer, order of the Markov chain
    %     interp_method : (optional) string, interpolation method
    %                     (e.g., 'spline', 'linear'). Default: 'spline'
    %
    %   EXAMPLE:
    %     cnm = CNM(u_snapshots, 0.01, 10, 2, 'spline');
    %     prediction = cnm.predict(initial_state, 1.0, 0.01);
    %
    %   -----------------------------------------------------------------------
    %   Author: Antonio Colanera
    %   Created: March 2023
    %   -----------------------------------------------------------------------
    properties
        dt                          % Time step size
        nq                          % Dimensionality of reduced state
        n_clusters                  % Number of clusters
        model_order                 % Order of the Markov model
        clusters                    % Cluster centroids
        visited_clusters            % Cluster labels assigned to training trajectory
        Distan                      % (unused) placeholder for distances
        tree                        % KD-tree for fast nearest-neighbor search
        sequence                    % Cluster sequence without self-transitions
        transition_prob             % Dictionary of transition probabilities
        transition_time             % Dictionary of transition times
        t                           % Time vector for training data
        times                       % Time vector for predicted trajectory
        prediction_visited_clusters % Cluster sequence visited during prediction
        interp                      % Interpolation method for final trajectory
        Qij                         % Transition matrix (for first-order model)
        Tij                         % (unused) placeholder for transition times matrix
    end

    methods (Static)
        function pippo = remove_sequential_duplicates(sequence)
            % Removes consecutive duplicate entries in a sequence
            is_different = diff(sequence(:)) ~= 0;
            pippo = sequence([true; is_different]);
        end

        function closest_cluster = find_closest_cluster(obj, query, kappa)
            % Finds the k-th nearest cluster for the input state vector
            if size(query, 2) ~= obj.nq
                query = query';
            end
            if nargin > 2
                [ind, ~] = knnsearch(obj.tree, query, 'k', kappa);
            else
                [ind, ~] = knnsearch(obj.tree, query);
            end
            closest_cluster = ind(end);
        end
        function [timeChange,numChange,timeKStay,timeJStay,rKOut,rJIn] = transitionParameterJK(K,J,idx)
            % this function provides the information from the given transition from cluster K to cluster J
            % input: cluster K(starting), cluster J(destination)
            % output: transition time, boundrary snapshots between transition.
            %
            % HouC, 05/17/2023

            % Copyright: 2023 HouC (houchangqaz@qq.com)
            % CC-BY-SA

            %%
            countSnapshots = 1;
            flag = 0;
            numChange = 0;

            for i = 2:length(idx)
                if (idx(i) == idx(i-1)) && (idx(i-1) == K)
                    countSnapshots = countSnapshots+1;
                elseif idx(i-1) == K
                    if idx(i) == J
                        countK = countSnapshots;
                        countSnapshots = countSnapshots+1;
                        flag = 1;
                        outK = i-1;
                        inJ = i;
                    else
                        countSnapshots = 1;
                        countK = 0;
                        countJ = 0;
                        outK = 0;
                        inJ = 0;
                        flag = 0;
                    end
                elseif (idx(i) == idx(i-1)) && (idx(i-1) == J) && (flag == 1) && (i ~= length(idx))
                    countSnapshots = countSnapshots+1;
                elseif (idx(i-1) == J) && (idx(i) ~= J) && flag == 1
                    numChange = numChange + 1;
                    countJ = countSnapshots - countK;
                    timeChangeKVector(numChange) = countK;
                    timePointKOutVector(numChange) = outK;
                    timeChangeJVector(numChange) = countJ;
                    timePointJInVector(numChange) = inJ;
                    timeChangeTotalVector(numChange) = countSnapshots;
                    countSnapshots = 1;
                    countK = 0;
                    countJ = 0;
                    outK = 0;
                    inJ = 0;
                    flag = 0;
                elseif  (idx(i) == idx(i-1)) && (idx(i-1) == J) && (flag == 1) && (i == length(idx))
                    numChange = numChange+1;
                    countJ = countSnapshots + 1 - countK;
                    timeChangeKVector(numChange) = countK;
                    timePointKOutVector(numChange) = outK;
                    timeChangeJVector(numChange) = countJ;
                    timePointJInVector(numChange) = inJ;
                    timeChangeTotalVector(numChange) = countSnapshots+1;
                    countSnapshots = 1;
                    countK = 0;
                    countJ = 0;
                    outK = 0;
                    inJ = 0;
                    flag = 0;
                end
            end

            if numChange == 0
                timeChange = 0;
                timeKStay = 0;
                timeJStay = 0;
                rKOut = 0;
                rJIn = 0;
            else
                timeChange = timeChangeTotalVector;
                timeKStay = timeChangeKVector;
                rKOut = timePointKOutVector;
                timeJStay = timeChangeJVector;
                rJIn = timePointJInVector;
            end

        end
    end

    methods
        function obj = CNM(reduced_state, dt, n_clusters, model_order, interp, varie)
            % Constructor for CNM model

            if nargin == 4
                obj.interp = 'spline';
            elseif nargin > 4
                obj.interp = interp;
            end

            % Basic initialization
            obj.dt = dt;
            obj.n_clusters = n_clusters;
            obj.model_order = model_order;
            obj.nq = size(reduced_state, 1);

            % Clustering with k-means++
            bb = waitbar(0, 'K-means++');
            [IDX, C] = kmeans(reduced_state', n_clusters, ...
                'MaxIter', 100000, 'Replicates', 50);
            obj.visited_clusters = IDX;
            obj.clusters = C;

            % Reordering clusters based on transition probabilities
            obj = reorder_clusters(obj);

            waitbar(1/3, bb, 'Building KD-tree...');
            obj.tree = KDTreeSearcher(obj.clusters);

            % Transition modeling
            obj.sequence = CNM.remove_sequential_duplicates(obj.visited_clusters);
            waitbar(2/3, bb, 'Computing transitions...');
            obj.transition_prob = obj.compute_transition_prob();
            obj.transition_time = obj.compute_transition_time();

            obj.t = (0:size(reduced_state,2)-1) * dt;

            if obj.model_order == 1
                obj.computeQij();
            end

            close(bb);
        end

        function prob = compute_transition_prob(obj)
            % Computes empirical transition probabilities (order L)
            if obj.model_order >= length(obj.sequence)
                error('Model order is too high for available sequence length');
            end

            prob = containers.Map('KeyType', 'char', 'ValueType', 'any');
            visited = obj.sequence(1:obj.model_order);
            next_seq = obj.sequence(obj.model_order+1:end);

            for i = 1:length(next_seq)
                key = sprintf('%d,', visited); key(end) = [];  % Create key
                if ~isKey(prob, key)
                    prob(key) = [];
                end
                prob(key) = [prob(key); next_seq(i)];
                visited = [visited(2:end); next_seq(i)];
            end

            % Normalize to probabilities
            keys = prob.keys;
            for i = 1:length(keys)
                key = keys{i};
                vals = prob(key);
                [counts, uniq] = groupcounts(vals);
                prob(key) = [uniq, counts / sum(counts)];
            end
        end

        function transition_time = compute_transition_time(obj)
            % Estimate average time spent before each transition
            seq_dup = zeros(size(obj.sequence));
            k = 1;
            for i = 1:length(obj.visited_clusters)-1
                if obj.visited_clusters(i+1) ~= obj.visited_clusters(i)
                    seq_dup(k) = seq_dup(k) + 1;
                    k = k + 1;
                else
                    seq_dup(k) = seq_dup(k) + 1;
                end
            end
            seq_dup(k) = seq_dup(k) + 1;

            transition = containers.Map;
            for i = 1:(length(obj.sequence)-obj.model_order)
                idx = i + obj.model_order;
                key = sprintf('%d,', obj.sequence(i:idx)); key(end) = [];
                if ~isKey(transition, key)
                    transition(key) = [];
                end
                transition(key) = [transition(key); 0.5 * obj.dt * sum(seq_dup(idx-1:idx))];
            end

            % Average transition time
            transition_time = containers.Map;
            for key = keys(transition)
                transition_time(key{1}) = mean(transition(key{1}));
            end
        end

        function alt_history = find_history(obj, history)
            % Given a partial history, find a matching history in the learned transitions
            history = history(:);
            for count = 1:obj.model_order
                test_hist = sprintf('%d,', history(count:end));
                test_hist(end) = [];

                keys = obj.transition_prob.keys;
                matches = keys(endsWith(keys, test_hist));

                if ~isempty(matches)
                    chosen = matches{randi(numel(matches))};
                    tokens = split(chosen, ',');
                    alt_history = str2double(tokens)';
                    return;
                end
            end
        end

        function history = find_initial_history(obj, init_state)
            % Determine cluster history for a new initial state
            if size(init_state, 2) ~= obj.nq
                init_state = init_state';
            end
            labels = knnsearch(obj.tree, init_state);
            history = CNM.remove_sequential_duplicates(labels);

            if length(history) < obj.model_order
                history = obj.find_history(history);
            end
        end

        function [history, ttime] = sample_next_cluster(obj, history)
            % Samples the next cluster based on current history
            key = join(string(history(:))', ',');

            if ~isKey(obj.transition_prob, key)
                last = obj.clusters(history(end), :);
                history(end) = obj.find_closest_cluster(obj, last, 2);
                history = obj.find_history(history);
                key = join(string(history), ',');
            end

            probs = obj.transition_prob(key);
            idx = randsample(length(probs(:, 1)), 1, true, probs(:, 2));
            next_cluster = probs(idx, 1);

            key = strcat(key, ',', num2str(next_cluster));
            history = [history; next_cluster];
            ttime = obj.transition_time(key);
        end

        function pred = interpolate_trajectory(obj, step_size)
            % Interpolates full-state trajectory from visited clusters
            if length(obj.times) < 2
                error('Prediction needs at least two time points');
            end
            tnew = obj.t; % Typically obj.times(1):step_size:obj.times(end)
            Xreco = obj.clusters(obj.prediction_visited_clusters, :)';
            pred = zeros(obj.nq, length(tnew));
            for ix = 1:obj.nq
                pred(ix, :) = interp1(obj.times, Xreco(ix, :), tnew, obj.interp);
            end
        end

        function prediction = predict(obj, initial_state, end_time, step_size)
            % Predicts future reduced state trajectory
            history = obj.find_initial_history(initial_state);
            obj.prediction_visited_clusters = history(:)';
            obj.times = obj.t(1:obj.model_order);

            bb = waitbar(0, 'Predicting...');
            while obj.times(end) < end_time
                [history, dt_next] = obj.sample_next_cluster(history);
                history = history(end - obj.model_order + 1:end);
                obj.times(end + 1) = obj.times(end) + dt_next;
                obj.prediction_visited_clusters(end + 1) = history(end);
                waitbar(obj.times(end) / end_time, bb);
            end
            close(bb);
            prediction = obj.interpolate_trajectory(step_size);
        end

        function Qij = computeQij(obj)
            % Builds global first-order transition matrix Q(i,j)
            keys = obj.transition_prob.keys;
            Q = zeros(obj.n_clusters, obj.n_clusters);
            for i = 1:length(keys)
                row = str2double(split(keys{i}, ','));
                vals = obj.transition_prob(keys{i});
                Q(vals(:, 1), row(end)) = vals(:, 2);
            end
            obj.Qij = Q;
            Qij = Q;
        end

        function obj = reorder_clusters(obj)
            % Reorders clusters based on transition flow and probabilities
            idx_map = zeros(obj.n_clusters, 1);
            trans_count = zeros(obj.n_clusters, 1);

            freq = tabulate(obj.visited_clusters);
            [~, max_idx] = max(freq(:, 2));
            idx_map(1) = freq(max_idx, 1);

            for i = 1:(obj.n_clusters - 1)
                for j = 1:obj.n_clusters
                    if ~ismember(j, idx_map)
                        [~, count] = CNM.transitionParameterJK(idx_map(i), j, obj.visited_clusters);
                        trans_count(j) = count;
                    else
                        trans_count(j) = 0;
                    end
                end

                if all(trans_count == 0)
                    sorted = sortrows(freq, -3);
                    for k = 1:length(sorted)
                        if ~ismember(sorted(k, 1), idx_map)
                            idx_map(i + 1) = sorted(k, 1);
                            break;
                        end
                    end
                else
                    [~, j_max] = max(trans_count);
                    idx_map(i + 1) = j_max;
                end
                trans_count(:) = 0;
            end

            % Apply new ordering
            new_idx = arrayfun(@(x) find(idx_map == x), obj.visited_clusters);
            new_centroids = obj.clusters(idx_map, :);

            obj.visited_clusters = new_idx;
            obj.clusters = new_centroids;
        end



    end
end
