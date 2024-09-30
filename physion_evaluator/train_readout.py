#!/usr/bin/env python3
import argparse
import csv
import json
import os.path
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_hdf5(file_path, key='features'):
    with h5py.File(file_path, 'r') as file:
        dataset = file[key][...]

    if key != 'label':
        dataset = dataset.astype(np.float16)

    return dataset


def all_scenario_eval(args, model, data, target, scenario, result, indices, results):
    with open(args.test_scenario_map, 'r') as f:
        stimulus_ = json.load(f)
        stimulus = {v: k for k, v in stimulus_.items()}

    accuracy = model.score(data, target)
    result[scenario + '_test'] = accuracy
    print(f"Accuracy on %s test data (%d data points): {accuracy:.4f}" % (scenario, target.shape[0]))
    # breakpoint()
    probs = model.predict_proba(data)
    preds = model.predict(data)
    for i in range(target.shape[0]):
        stimulus_name = stimulus[indices[i]].split('/')[-1][:-5]
        entry = ['all', args.model_name, scenario, float(result['train']), float(result[scenario + '_test']),
                 "features", float(probs[i][0]), float(probs[i][1]), int(preds[i]),
                 int(target[i]), stimulus_name]
        results.append(entry)
    return result, results


def test_model(model, test_data, test_label, args, result):
    with open(args.test_scenario_indices, 'r') as f:
        test_data = test_data.reshape(test_data.shape[0], -1)  # [:, :10]

        scenarios_indices = json.load(f)

        results = [['Readout Train Data', 'Model', 'Readout Test Data', 'Train Accuracy',
                    'Test Accuracy', 'Readout Type', 'Predicted Prob_false',
                    'Predicted Prob_true', 'Predicted Outcome', 'Actual Outcome',
                    'Stimulus Name']]

        # get per scenario accuracy
        for sc in scenarios_indices.keys():
            ind = sorted(scenarios_indices[sc])

            data, target = test_data[ind], test_label[ind]
            result, results = all_scenario_eval(args, model, data, target,
                                                sc, result, ind,
                                                results)
        all_acc_sce = []
        for sc in scenarios_indices.keys():
            all_acc_sce.append(result[sc + '_test'])

        print(f"Accuracy on full test data avg scenario: {np.mean(all_acc_sce):.4f}")
        result['full_test_sce_acc'] = np.mean(all_acc_sce)

        # save to csv
        filename = args.save_path + '/' + args.model_name + '_results.csv'
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(results)
    return result


def get_indices(scenarios_indices):
    indices = []
    for k in scenarios_indices.keys():
        indices += scenarios_indices[k]
    return indices


def train(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    X = load_hdf5(args.train_path, 'features')
    X = X.reshape(X.shape[0], -1)

    y = load_hdf5(args.train_path, 'label')

    param_grid = {'clf__C': np.array(args.clf_C), 'clf__penalty': ['l2']}

    model = LogisticRegression(max_iter=20000)

    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])
    print("pipeline created", pipeline)

    # Perform grid search with stratified k-fold cross-validation
    print('Grid Search with KFold')
    import time
    t = time.time()
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, verbose=3)
    grid_search.fit(X, y)
    print(time.time() - t)
    # Print the best hyperparameters found by the grid search
    print(f"Best hyperparameters: {grid_search.best_params_}")
    result = grid_search.best_params_

    # Evaluate the model on the full data
    accuracy = grid_search.score(X, y)
    print(f"Accuracy on train data: {accuracy:.4f}")
    result['train'] = accuracy

    #
    test_data = load_hdf5(args.test_path, 'features')
    test_label = load_hdf5(args.test_path, 'label')

    result = test_model(grid_search, test_data, test_label, args, result)

    filename = args.save_path + '/' + args.model_name + '_results.json'

    with open(filename, 'w') as f:
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a logistic regression model')

    parser.add_argument('--clf_C', type=float, nargs='+', default=[1e-6, 1e-5, 0.01, 0.1, 1, 5, 10, 20], help='Values for clf__C parameter grid')

    parser.add_argument('--random-state', type=int, default=42, help='random seed for reproducibility')

    # data params
    parser.add_argument('--train-path', type=str, help='Train data hdf5 path')
    parser.add_argument('--test-path', type=str, help='Test data hdf5 path')

    parser.add_argument('--model-name', type=str, help='Name of the model that is being evaluated')

    parser.add_argument('--train-scenario-indices', type=str, required=True,
                        help='train indices json path')
    parser.add_argument('--test-scenario-indices', type=str, required=True,
                        help='test indices json path')
    parser.add_argument('--test-scenario-map', type=str, required=True,
                        help='test scenario json path')

    parser.add_argument('--save_path', type=str, help='where to save the results')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
