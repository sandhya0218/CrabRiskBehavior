# -*- coding: utf-8 -*-
"""
Created on 12 DEC 2022
Explore and perform statistical tests on drugged crab data
@author: Ryan Dancoes, Sandhya Sangappa, Sydney Gallo
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import warnings


def read_data():
    """
    Create DataFrame from csv file
    @return: DataFrame of all data
    """
    path = 'data'  # "/content/drive/My Drive/MA 346 Project 2/"
    file = 'crab_behavior.csv'
    data = os.path.join(path, file)

    df = pd.read_csv(data)
    return df


def inspect(df, fname):
    """
    Show metadata, summary statistics, and histogram of DataFrame
    @param df: DataFrame with >=1 numerical dtype column
    @param fname: File name of figure to save
    """
    print(df.shape, '\n')
    print(df.head(), '\n')
    df.info()

    print('\n', df.describe().T)  # Transpose
    df.hist(figsize=(10, 8))
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join('figures', fname))


def clean(df):
    """
    Remove inital unnecessary columns
    @param df: DataFrame
    @return: Clean DataFrame
    """
    # Unnecessary columns
    df = df.drop(columns=['Trial_type', 'Pred_Escape', 'Pred_Kill'])

    # Identify and drop rows with null values
    df = df.dropna()
    drop_indices = []
    for i, val in df.iterrows():
        if val.Active == ' ':
            drop_indices.append(i)
    df = df.drop(index=drop_indices).reset_index()
    return df


def engineer(df):
    """
    Build new columns and transform existing ones for regression
    @param df: DataFrame
    @return: Engineered DataFrame
    """
    # Create new aggregate column
    df.Active = df.Active.astype('int')
    df['Score'] = df.Active + df.Agonistic + df.Foraging + df.Social - df.Still

    df.drop(columns=['index', 'Active', 'Agonistic', 'Foraging', 'Social', 'Still'], inplace=True)

    # Create WeekNum column
    week_count = 0
    week_new = False
    for i, val in df.iterrows():
        if i != 0:
            if df.loc[i-1, 'Time'] == 'Night' and val.Time == 'Day':
                week_new = True
            if week_new:
                week_count += 1
                week_new = False
            df.loc[i, 'WeekNum'] = week_count
        else:
            df.loc[i, 'WeekNum'] = week_count

    # Transform Object columns to binary
    df.Sex = df.Sex.map({'F': 0, 'M': 1})
    df.Status = df.Status.map({'Sub': 0, 'Dom': 1})
    df.Time = df.Time.map({'Day': 0, 'Night': 1})

    # Transform Treatment column to categorical Yes/No
    df['Treatment_cat'] = df.Treatment.map({'Control': 'No', '3': 'Yes', '30': 'Yes'})

    # Create Risk column subdividing Score into three groups
    df['Risk'] = np.zeros(len(df))
    df.loc[df.Score < -3, 'Risk'] = -1
    df.loc[df.Score > 3, 'Risk'] = 1

    return df


def explore(df):
    """
    Visualize data
    @param df: DataFrame
    """
    # Countplot of each Treatment observation
    fig, ax = plt.subplots()  # NOQA
    _ = sns.countplot(data=df, x='Treatment', order=['Control', '3', '30'])
    plt.title('Count of each Treatment Observation')
    # plt.show()
    plt.savefig(os.path.join('figures', 'countplot.png'))

    # Simple histogram of Score
    df.hist('Score')
    # plt.show()
    plt.savefig(os.path.join('figures', 'histogram_score_simple.png'))

    # Split histograms of Score by Treatment
    fig, ax = plt.subplots(1, 3, figsize=(10, 8))
    colors = ['r', 'g', 'b']
    labels = ['Control', '3', '30']
    for i in range(len(labels)):
        df.loc[df.Treatment == labels[i], 'Score'].hist(ax=ax[i], label=labels[i], color=colors[i])
        ax[i].set_title(f'Score dist of treatment: {labels[i]}')
        ax[i].set_ylim(0, 370)
    # plt.show()
    plt.savefig(os.path.join('figures', 'histogram_score_split.png'))

    # Bar plot of Score by Treatment over time
    fig, ax = plt.subplots()  # NOQA
    df_treat_week = df.groupby(['Treatment', 'WeekNum'])['Score'].agg('mean').reset_index()
    hue_order = ['Control', '3', '30']
    ax = sns.barplot(data=df_treat_week, x='WeekNum', y='Score', hue='Treatment', hue_order=hue_order)
    ax.set_title('Score by Treatment over Week Number')
    # plt.show()
    plt.savefig(os.path.join('figures', 'barplot_score_treatment_week.png'))

    # Bar plot of Score by Treatment and time of day over time
    order = ['Control', '3', '30']
    g = sns.FacetGrid(data=df, row="Time", col="Treatment", col_order=order, margin_titles=True)
    g.map(sns.barplot, "WeekNum", "Score", errorbar=('ci', False))
    g.fig.suptitle('Score by Treatment and Time of Day over Weeks')
    g.fig.tight_layout()
    fig.subplots_adjust(top=.90)
    # plt.show()
    plt.savefig(os.path.join('figures', 'barplot_score_treatment_time_week.png'))


def bootandswarmplot(df):
    """
    Swarmplot of Score by Treatment
    @param df: DataFrame
    """
    def boot(data, cat, boot_num=70):
        """
        Bootstrap function for swarmplot
        @param data: DataFrame
        @param cat: Treatment category
        @param boot_num: Number of bootstrap samples
        @return: DataFrame of bootstrapped data
        """
        means = [np.random.choice(data.loc[data['Treatment_ord'] == cat, 'Score'], len(data.loc[data['Treatment_ord'] == cat, 'Score']), replace=True).mean() for _ in range(boot_num)]  # Bootstrapping using mean
        arr = ([cat]*len(means), means)  # array of duplicate Treatment category
        means = pd.DataFrame({'cats': arr[0], 'means': arr[1]})
        return means

    df_means = pd.DataFrame({'cats': [], 'means': []})
    for i in range(3):
        df2 = boot(df, i)
        df_means = pd.concat([df_means, df2], axis=0)

    fig, ax = plt.subplots()  # NOQA
    ax = sns.swarmplot(data=df_means, x='cats', y='means', hue='cats')
    ax.set_xticks(range(3), ['Control', '3', '30'])
    ax.legend(['Control', '3', '30'])
    ax.set_title('Swarmplot of Bootstrapped Scores by Treatment')
    # plt.show()
    plt.savefig(os.path.join('figures', 'swarmplot_score_treatment.png'))


def regression(df, cols):
    """
    Run regression on data
    @param df: DataFrame
    @param cols: list of columns to use in regression
    @return: Regression results
    """
    X = df[cols]
    X = sm.add_constant(X)
    y = df['Score']
    print(X)
    modfit = sm.OLS(y, X).fit()
    print(modfit.summary())
    return modfit


def residuals(df, mod, cols, fname):
    """
    Plot residuals of regression
    @param df: DataFrame
    @param mod: Regression results
    @param cols: list of columns used in regression
    @param fname: filename to save plot
    """
    def prepare_resids(data, category):
        """
        Prepare residuals for plotting
        @param data: DataFrame
        @param category: binary or ordinal category
        """
        x = data[cols]
        x = sm.add_constant(x, has_constant='add')  # Add constant to data if not already present
        pred_x = mod.predict(x)
        y = data['Score']
        arr = ([str(category)]*len(y), y, pred_x)  # array of duplicate Treatment category, Score, and predicted Score
        return pd.DataFrame({'cats': arr[0], 'y': arr[1], 'pred_x': arr[2]})

    pred_var = cols[0]
    prep_df = pd.DataFrame({'cats': [], 'y': [], 'pred_x': []})  # Initialize DataFrame for residuals

    series = df[cols[0]].unique()  # Get unique values of predictor variable
    for i in series:
        df_ = df.loc[df[pred_var] == i]  # Get subset of data for each unique value of predictor variable
        df_ = prepare_resids(df_, i)
        prep_df = pd.concat([prep_df, df_], axis=0)  # Concatenate DataFrames

    prep_df['resid'] = prep_df['y'] - prep_df['pred_x']  # Calculate residuals
    series = [str(i) for i in sorted(series)]  # Convert series to string
    fig, ax = plt.subplots()  # NOQA
    sns.violinplot(data=prep_df, x='cats', y='resid', order=series)
    plt.title(f'Residuals of {pred_var} on Score')
    # plt.show()
    plt.savefig(os.path.join('figures', fname))

    # fig = sm.graphics.plot_regress_exog(mod, pred_var, fig=plt.figure(figsize=(8, 8)))  # NOQA, Plot residuals
    # plt.show()  # Look at Residuals versus Treatment_ord graph


def chi_independence_test(data, male=False, dose_given=False, overall=False):  # doseGive: True= 3 vs 30, False=Yes vs No dose
    """
    Chi-squared test of independence
    @param data: DataFrame
    @param male: Boolean of isMale
    @param dose_given: Boolean of dose given
    @param overall: Boolean of overall
    """
    labels = {'Sex': '', 'Dose': '', 'Overall': ''}

    # Get data for tests
    if not overall:
        labels['Overall'] = 'Sex Subset'
        if male:
            data = data.loc[data['Sex'] == 1, :]
            labels['Sex'] = 'Male'
        else:
            data = data.loc[data['Sex'] == 0, :]
            labels['Sex'] = 'Female'
    else:
        labels['Overall'] = 'Overall'

    if dose_given:
        data = data[data['Treatment_yes_no'] == 1]
        treatType = data.Treatment
        labels['Dose'] = '3vs30'
    else:
        treatType = data.Treatment_yes_no
        labels['Dose'] = 'YesVsNo'

    print(f"Chi^2 Ind Test: {labels['Overall']} {labels['Sex']} {labels['Dose']}\n")

    # Table of counts
    ct = pd.crosstab(treatType, data.Risk, margins=True)
    print(ct)

    row_sum = ct.iloc[0:2, 3].values

    # Array of expected counts
    exp = []
    for j in range(2):
        for val in ct.iloc[2, 0:3].values:
            exp.append(val * row_sum[j] / ct.loc['All', 'All'])  # Expected count = (row total * column total) / grand total
    # print(exp)
    # (len(row_sum)-1)*(len(ct.iloc[2,0:3].values)-1)  # Degrees of freedom

    # Array of observed counts
    obs = np.array([ct.iloc[0][0:3].values,
                    ct.iloc[1][0:3].values])

    # Checksum: ((obs.ravel() - exp)**2/exp).sum()

    # Obtain chi^2 and p-value and Degrees of Freedom
    summary = stats.chi2_contingency(obs)[0:3]
    if summary[1] < 0.05:
        print(f'Dependent relationship between Risk Score and Treatment Type: p-val={summary[1]:.2e}')
    else:
        print(f'Independent relationship between Risk Score and Treatment Type: p-val={summary[1]:.2e}')
    print('\n')


def main():
    # Prepare for analysis
    np.random.seed(12345)
    warnings.filterwarnings('ignore')

    # Read in data and inspect
    crab = read_data()
    inspect(crab, 'histograms_before.png')

    # Clean, transform, and inspect data
    crab = clean(crab)
    crab = engineer(crab)
    inspect(crab, 'histograms_after.png')

    # Plot data
    explore(crab)

    # Ordinal transformation of Treatment
    crab['Treatment_ord'] = crab.Treatment.map({'Control': 0, '3': 1, '30': 2})

    # Regression with ordinal Treatment
    ord_col = ['Treatment_ord', 'Sex', 'Status', 'Time']
    model = regression(crab, ord_col)
    residuals(crab, model, ord_col, 'residuals_ord.png')

    # Categorial transformation of Treatment
    crab['Treatment_yes_no'] = crab.Treatment.map({'Control': 0, '3': 1, '30': 1})

    # Regression with categorial Treatment (Yes/No)
    cat_col = ['Treatment_yes_no', 'Time', 'Status', 'Sex']
    model = regression(crab, cat_col)
    residuals(crab, model, cat_col, 'residuals_yes_no.png')

    # Create DataFrame of only crabs that received a dose
    crab_treated = crab[crab['Treatment_yes_no'] == 1]

    # Differentiate between crabs that received 3 and 30
    crab_treated['Treatment_three_thirty'] = crab_treated['Treatment'].map({'3': 0, '30': 1})

    # Regression with categorial Treatment (3/30)
    treat_col = ['Treatment_three_thirty', 'Sex', 'Status', 'Time']
    model = regression(crab_treated, treat_col)
    residuals(crab_treated, model, treat_col, 'residuals_three_thirty.png')

    bootandswarmplot(crab)

    # Chi-squared test of independence for various combinations
    for dose in [False, True]:
        chi_independence_test(crab, dose_given=dose, overall=True)
        for sex in [False, True]:
            chi_independence_test(crab, dose_given=dose, overall=False, male=sex)


if __name__ == '__main__':
    main()