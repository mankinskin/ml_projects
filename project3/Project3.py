#!/usr/bin/env python
# coding: utf-8

# # Project 3: Real world data
# - Explore the data.
# - Train at least three different regression models and cross-validate them with MSE and R2-score.
# - Try out pre-processing techniques.
# - Choose two types of feature selection and implement at least one method for each.
# Experimentally evaluate your feature selections.
# - For the best performing combination of model and pre-processing steps, implement and plot the so-called learning curve
# on your own.

# In[126]:


# global imports
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display
import math
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
import os


# ## Data exploration

# In[127]:


# generic function to extract named properties from the data set items
# fs is a list of tuples, first a name, then a function mapping an item to a property
notebook_dir = os.path.dirname(os.path.abspath("Project3.ipynb"))
def property_table(name, fs, ds):
    keys = list(ds.keys()) # row names
    prop_names = fs.keys() # column names
    funcs = fs.values() # property functions
    items = list(map(lambda t: t[1], ds.items())) # data items
    mapped = zip(*list(map(lambda f: map(f, items), funcs))) # each prop for each item
    df = pd.DataFrame(mapped, index=keys, columns=prop_names) # use pandas.DataFrame to print a table
    tex_table = df.to_latex()
    write_str_to_file(name, tex_table)
    display(df)
    return df
def feature_info(name, dict):
    return property_table(
        f"{name} feature info",
        {
        'Min': lambda item: np.round(np.amin(item, axis=0)),
        'Max': lambda item: np.round(np.amax(item, axis=0)),
        'Mean': lambda item: np.round(np.mean(item, axis=0)),
        'Std': lambda item: np.round(np.std(item, axis=0)),
    }, dict)
# uncomment to write plots and tables to file
def write_str_to_file(filename, str):
    #filename = filename.replace(" ", "_")
    #file = open(f"{notebook_dir}/plots/{filename}_table.tex", 'w')
    #file.write(str)
    #file.close()
    return
def save_figure(name, fig=None):
    #name = name.replace(" ", "_")
    #if fig==None:
    #    plt.savefig(f"{notebook_dir}/plots/{name}.eps", format="eps")
    #else:
    #    fig.savefig(f"{notebook_dir}/plots/{name}.eps", format="eps")
    return


# In[128]:


# TODO: load dataset 1
data_set = np.load('dataset.npz')
X = data_set['X']
y = data_set['y']
features = data_set['features']
property_table(
    "data overview",
    {
    'Count': lambda item: len(item),
    'Shape': lambda item: item.shape,
    'Min': lambda item: np.round(np.amin(item, axis=0)),
    'Max': lambda item: np.round(np.amax(item, axis=0)),
    'Mean': lambda item: np.round(np.mean(item, axis=0)),
    'Std': lambda item: np.round(np.std(item, axis=0)),
}, {
    "X": X,
    "y": y
})
plt.title("Y distribution")
plt.plot(sorted(y))
plt.tight_layout()
save_figure("y_distribution")
feature_dict = { name: np.choose(i, X.T) for (i, name) in enumerate(features)}
feature_info("y info", { "y": y })
feature_info("full", feature_dict)
feature_dict["y"] = y.ravel()
values = np.array(list(feature_dict.values()))
df = pd.DataFrame(values.T, columns=feature_dict.keys())
cor = df.corr()
plt.figure()
plt.title("Feature Correlation")
plt.pcolor(cor)
plt.yticks(np.arange(0.5, len(cor.index), 1), cor.index)
plt.xticks(np.arange(0.5, len(cor.columns), 1), cor.columns, rotation=45, rotation_mode="anchor", ha="right")
plt.colorbar()
plt.tight_layout()
save_figure("full_feature_correlation")
plt.show()
cor = cor.drop(columns=list(feature_dict.keys())[:-1], index=list(feature_dict.keys())[-1])
cor = cor.transpose()
display(cor)
write_str_to_file("feature_y_correlation", cor.to_latex())


# ## Training & Evaluation

# In[129]:


CROSS_VAL_FOLDS=10
# train and score model using cross validation
def score_model(model, data_pair, scoring):
    (x, y) = data_pair
    return cross_val_score(model, x, y.ravel(), cv=CROSS_VAL_FOLDS, scoring=scoring)
# score a dictionary of models
def score_models(models, data_pair, scoring):
    #print(f'Scoring: {scoring}')
    #print(f'Scoring {name} model')
    return { name: score_model(model, data_pair, scoring) for name, model in models.items() }
# test multiple models on an input
def score_models_on_data(regs, name, data_pair, scorings):
    #print(f'Scoring models on {name} data')
    #print(f'{name} = {X}')
    return { scoring: score_models(regs, data_pair, scoring) for scoring in scorings.keys() }
# test multiple models on multiple data scalings using multiple scorings
def score_all_models_on_all_data_using_all_scorings(regs, data, scorings):
    # data_scores is a 4D dictionary "tensor" containing
    # scoring_scores for each data scaling
    # scoring_scores are 3D dictionaries containing
    # model_scores for each scoring
    # model_scores are 2D dictionaries containing
    # scores for each model
    # scores are 1D arrays of scores with length CROSS_VAL_FOLDS
    data_scores = {
        name: score_models_on_data(regs, name, data_pair, scorings)
        for name, data_pair in data.items()
    }
    for (i, (name, scoring_scores)) in enumerate(data_scores.items()):
        plot_model_scores_for_data(name, scoring_scores)
    plot_avg_model_scores_by_data_scaling(data_scores)
    plot_avg_data_scaling_scores_by_model(data_scores)
def plot_model_scores_for_data(data_name, scoring_scores):
    plt.figure(figsize=(20,6))
    plt.gcf().suptitle(data_name)
    n = len(scoring_scores)
    for (i, (scoring, model_scores)) in enumerate(scoring_scores.items()):
        plt.subplot(1,n,i+1)
        plt.title(scoring)
        plot_model_scores(model_scores)
    save_figure(f"{data_name}_model_scores")
def plot_model_scores(scores):
    plt.ylabel("Scores")
    for name, score in scores.items():
        plot_model_score(name, score)
    plt.legend()
def plot_model_score(name, scores):
    #print(f"Plotting scores for model {name}")
    #print(f"Scores {scores.shape}")
    avg_score = np.mean(scores, axis=0)
    c = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.plot(scores, color=c, label=name, alpha=1.0)
    plt.axhline(avg_score, color=c, linestyle="--", label=f"{name}(average)")
def scale(data, scaler):
    return scaler.fit_transform(data)
# transpose first two levels of dictionary
def sub_keys(d):
    return { key for sub in d.values() for key in sub.keys() }
# transpose first two levels of dictionary
def transpose_dict(d):
    keys = sub_keys(d)
    return {
        key: { k:d[k][key] for k in d if key in d[k] } for key in keys
    }
# plot each scaling's scores averaged across all models
def plot_avg_model_scores_by_data_scaling(data_scores):
    # average scores across models
    avg_data_scores = {
        scaling: {
            scoring: np.mean(np.array(list(model_scores.values())), axis=(0,1))
            for scoring, model_scores in scoring_scores.items()
        } for scaling, scoring_scores in data_scores.items()
    }
    # transpose so that scoring dictionary is top level
    avg_data_scores_transposed = transpose_dict(avg_data_scores)
    #print(avg_data_scores_transposed)
    plot_bar_charts_for_dicts(
        "Average model scores by data scaling",
        "Avg Model score",
        avg_data_scores_transposed
    )
# plot each models scores averaged across all data scalings
def plot_avg_data_scaling_scores_by_model(data_scores):
    transposed = transpose_dict(data_scores)
    transposed_again = {
        scoring: transpose_dict(data_scores) for scoring, data_scores in transposed.items()
    }
    avg_model_scores_transposed = {
        scoring: {
            model: np.mean(np.array(list(data_scores.values())), axis=(0,1))
            for model, data_scores in model_scores.items()
        } for scoring, model_scores in transposed_again.items()
    }
    plot_bar_charts_for_dicts(
        "Average scaling scores by model",
        "Avg Scaling score",
        avg_model_scores_transposed
    )
# plot one titled, labelled bar chart for each nested dictonary in dicts
def plot_bar_charts_for_dicts(title, ylabel, dicts):
    plt.figure(figsize=(20,6))
    plt.suptitle(title)
    n = len(dicts)
    for (i, (name, data)) in enumerate(dicts.items()):
        plt.subplot(1,n,i+1)
        plot_bar_chart(ylabel, name, {k: v for k, v in sorted(data.items(), key=lambda t: t[1])})
    save_figure(f"{title}_barcharts")
def plot_bar_chart(ylabel, name, data, width=0.25):
    # barchart from https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html
    values = data.values()
    ind = np.arange(len(values))
    low = min(values)
    high = max(values)
    plt.gca().set_ylim([low - 0.5000*(high-low), high + 0.5000*(high-low)])
    plt.gca().bar(ind, values, width)
    plt.gca().set_title(name)
    plt.gca().set_ylabel(ylabel)
    plt.gca().set_xticks(ind)
    plt.gca().set_xticklabels(data.keys(), rotation=45, rotation_mode="anchor", ha="right")
    plt.gcf().tight_layout()

def data_randomized(x, y, seed=None):
    p = np.random.RandomState(seed=seed).permutation(len(x))
    return (x[p], y[p])


# In[130]:


# Todo: evaluate different techniques of feature selection
# embedded feature selections
scorings = {
    "neg_mean_squared_error": lambda y_true, y_pred: -mean_squared_error(y_true, y_pred),
    "r2": r2_score,
}
regs = {
    "linreg": LinearRegression(),
    "ridge": Ridge(),
    "huber": HuberRegressor(),
    "random_forest(mean squared error)": RandomForestRegressor(n_estimators=10),
    "random_forest(mean absolute error)": RandomForestRegressor(n_estimators=10, criterion="mae"),
}
# filter feature selections
data_raw = (X, y)
# randomize data multiple times
p = np.random.RandomState(seed=42).permutation(len(X))
data_randomized_42 = (X[p], y[p])
p = np.random.RandomState(seed=37).permutation(len(X))
data_randomized_37 = (X[p], y[p])
p = np.random.RandomState(seed=231).permutation(len(X))
(X, y) = (X[p], y[p])
var_filter = VarianceThreshold(threshold=0.3)
var_filtered = var_filter.fit_transform(data_randomized_42[0])
std_scaled_data = (scale(var_filtered, StandardScaler()), data_randomized_42[1])
var_filtered_data = (scale(var_filtered, StandardScaler()), data_randomized_42[1])
data = {
    "raw": data_raw,
    "Randomized(seed=42)": data_randomized_42,
    "Randomized(seed=37)": data_randomized_37,
    "robust scaled": (scale(var_filtered, RobustScaler()), var_filtered_data[1]),
    "standard scaled": std_scaled_data,
    "L2 Normalized": (scale(var_filtered, Normalizer()), var_filtered_data[1]),
}

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    score_all_models_on_all_data_using_all_scorings(regs, data, scorings)


# In[131]:


remaining_features = features[var_filter.get_support()]
print(f"Remaining features: {len(remaining_features)} (of {len(features)})")
feature_dict = { name: np.choose(i, var_filtered_data[0].T) for (i, name) in enumerate(remaining_features)}
feature_info("variance_filtered", feature_dict)

sel = SelectFromModel(RandomForestRegressor(n_estimators=10))
sel.fit(std_scaled_data[0], std_scaled_data[1].ravel())
remaining_features = remaining_features[sel.get_support()]
print(f"Remaining features: {len(remaining_features)} (of {len(features)})")
feature_dict = { name: np.choose(i, data_raw[0].T) for (i, name) in enumerate(remaining_features)}
feature_info("random_forest", feature_dict)
feature_dict["y"] = y.ravel()
values = np.array(list(feature_dict.values()))
print(f"{values.shape}")
df = pd.DataFrame(values.T, columns=feature_dict.keys())
cor = df.corr()
cor = cor.drop(columns=list(feature_dict.keys())[:-1], index=list(feature_dict.keys())[-1])
display(cor)


# ## Learning curves

# In[136]:


# learning curve for model, data and scoring
def learning_curve_scores(model, x_train, y_train, x_test, y_test, score_fn, step):
    train_scores = []
    test_scores = []
    n = len(x_train)
    steps = math.floor(n/step)
    for i in range(1, steps+1):
        l = min(i*step, n)
        model.fit(x_train[0:l], y_train[0:l].ravel())
        y_pred_train = model.predict(x_train[0:l])
        y_pred_test = model.predict(x_test[0:l])
        train_scores.append(score_fn(y_train[0:l], y_pred_train))
        test_scores.append(score_fn(y_test[0:l], y_pred_test))
    return np.array(train_scores), np.array(test_scores)
# calc average cross validated learning curve for model, data and scoring
def cv_learning_curve_scores(model, data, score_fn, step):
    (x_data, y_data) = data
    all_train_scores = []
    all_test_scores = []
    for (i, (train_index, test_index)) in enumerate(KFold(n_splits=CROSS_VAL_FOLDS).split(x_data)):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        train_scores, test_scores = learning_curve_scores(model, x_train, y_train, x_test, y_test, score_fn, step)
        all_train_scores.append(train_scores)
        all_test_scores.append(test_scores)
    avg_train_scores = np.mean(all_train_scores, axis=0)
    avg_test_scores = np.mean(all_test_scores, axis=0)
    return (avg_train_scores, avg_test_scores)
# plot average learning curves for each model with all data scalings and scorings
def plot_all_learning_curves(models, datas, scorings, step=1):
    data_count = len(datas)
    scoring_count = len(scorings)
    model_count = len(models)
    plot_width = 6
    plot_height = 4
    all_scores = []
    for model in models.values():
        for data in datas.values():
            for scoring in scorings.values():
                all_scores.append(cv_learning_curve_scores(model, data, scoring, step))


    plt.subplots(1, scoring_count, figsize=(plot_width*scoring_count,plot_height))
    for (k, (model_name, model)) in enumerate(models.items()):
        plt.suptitle(f"Average Learning curves (step={step}) for Model: {model_name}")
        for (i, (data_name, data)) in enumerate(datas.items()):
            plt.suptitle(f"{model_name} with scaling: {data_name}")
            for (j, (scoring_name, scoring)) in enumerate(scorings.items()):
                index = k*(data_count*scoring_count) + i*scoring_count + j
                plt.subplot(data_count, scoring_count, j+1)
                plt.title(f"Score: {scoring_name}")
                avg_train_scores, avg_test_scores = all_scores[index]
                plt.plot(avg_train_scores, label=f"{model_name} train score")
                plt.plot(avg_test_scores, label=f"{model_name} test score")
                plt.xlabel("steps")
                plt.ylabel("score")
                plt.legend()
        plt.tight_layout()
        plt.subplots_adjust(left=0.15, top=0.9)
        plt.show()
        save_figure(f"{model_name}_learning_curves")


# In[ ]:


scorings = {
    "neg_mean_squared_error": lambda y_true, y_pred: -mean_squared_error(y_true, y_pred),
    "r2": r2_score,
}
regs = {
    #"linreg": LinearRegression(),
    "random_forest(mean squared error)": RandomForestRegressor(n_estimators=10),
    #"random_forest(mean absolute error)": RandomForestRegressor(n_estimators=10, criterion="mae"),
}
data = {
    #"raw": data_raw,
    #"Randomized(seed=42)": data_randomized_42,
    #"robust scaled": (scale(var_filtered, RobustScaler()), var_filtered_data[1]),
    "standard scaled": std_scaled_data,
    #"L2 Normalized": (scale(var_filtered, Normalizer()), var_filtered_data[1]),
}
plot_all_learning_curves(regs, data, scorings, step=20)

