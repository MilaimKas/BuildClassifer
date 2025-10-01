"""
Contains functions helpers to analyze data
"""

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr, chi2_contingency, pointbiserialr, f_oneway
from sklearn.preprocessing import LabelEncoder


# classify categorical variable accorinding to cardinality and skweness
# -----------------------------------------------------------------------------------------------------------------

def gini_coefficient(category_counts):
    """Compute the Gini coefficient for category frequencies."""
    p = category_counts / category_counts.sum()
    return 1 - np.sum(p**2)

# Define threshold functions for cardinality and gini
def categorize_cardinality(val, lower_thresh=50, upper_thresh=200):
    if val <= lower_thresh:
        return "Low"
    elif val <= upper_thresh:
        return "Medium"
    else:
        return "High"

def categorize_gini(val, lower_thresh=0.2, upper_thresh=0.6):
    if val < lower_thresh:
        return "Low"
    elif val < upper_thresh:
        return "Medium"
    else:
        return "High"

# Assign smoothing class based on table mapping
def assign_smoothing_group(card_class, gini_class):
    if card_class == "Low" and gini_class == "Low":
        return "A"
    elif card_class == "Medium" and gini_class == "Low":
        return "B"
    elif card_class == "Low" and gini_class == "Medium":
        return "B"
    elif card_class == "Medium" and gini_class == "Medium":
        return "B"
    elif card_class == "High" and gini_class == "Low":
        return "B"
    elif card_class == "Low" and gini_class == "High":
        return "B"
    elif card_class == "Medium" and gini_class == "High":
        return "C"
    elif card_class == "High" and gini_class == "Medium":
        return "C"
    elif card_class == "High" and gini_class == "High":
        return "C"

def assign_smoothing_class(df:pd.DataFrame, cat_col_list:list, gini_thresh=[0.2, 0.6], card_thresh=[50, 200]):
    """

    Function that assigns categorical variablles into 3 groups according to their skweness (gini) and cardinality.

    Args:
        df (pd.DataFrame): Data Frame containing the data and categorical columns.
        cat_col_list (list): list of categorical columns to group.

    Raises:
        ValueError: categorical column not present in data frame

    Returns:
        pd.DataFrame: data frame with column name as index, cardinality, gini coefficient and assigned group.
    """
    for col in cat_col_list:
        if col not in df.columns:
            raise ValueError(f"{col} not found in dataframe") 
    
    card = []
    gini_c = []
    for col in cat_col_list:
        category_counts = df[col].value_counts()
        card.append(len(category_counts))
        gini_c.append(gini_coefficient(category_counts))

    df_tmp = pd.DataFrame(data={"cardinality":card, "gini_coef":gini_c}, index=cat_col_list)
    df_tmp["cardinality_class"] = df_tmp["cardinality"].apply(categorize_cardinality, args=gini_thresh)
    df_tmp["gini_class"] = df_tmp["gini_coef"].apply(categorize_gini, args=card_thresh)
    df_tmp["group"] = df_tmp.apply(lambda row: assign_smoothing_group(row["cardinality_class"], row["gini_class"]), axis=1)

    return df_tmp

def plot_smoothing_class(df):
    """ Plot result of 'assign_smoothing_class' function """
    plt.bar(cat_class["card"].keys(), cat_class["card"]/cat_class["card"].max(), label="card")
    cat_class["gini"].plot(kind="bar", color="red", alpha=0.4, label="gini")
    plt.legend()
    plt.xticks(rotation=90)

# General correlation analysis
# -----------------------------------------------------------------------------------------------------------------


def stat(data1, data2, n_bootstraps=1000, statistics=np.mean, alpha=0.05, alternative="two-sided"):
    """
    Function that returns statistic on the difference between two data sets:
        - bootstrapping confidence intervals and p-value
        - rank based mannwhitneyu p-value
    """

    # remove nan
    data1_proc = data1[~np.isnan(data1)]
    data2_proc = data2[~np.isnan(data2)]

    # observed difference
    observed_stat = statistics(data1_proc) - statistics(data2_proc)

    bootstrapped_diffs = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        sample1 = np.random.choice(data1_proc, size=len(data1_proc), replace=True)
        sample2 = np.random.choice(data2_proc, size=len(data2_proc), replace=True)
        bootstrapped_diffs[i] = statistics(sample1) - statistics(sample2)

    # confidence interval assuming normality
    lower = np.percentile(bootstrapped_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrapped_diffs, 100 * (1 - alpha / 2))

    # Calculate one-tailed p-value
    extreme_counts = sum(np.abs(bootstrapped_diffs-observed_stat) > np.abs(observed_stat))/len(bootstrapped_diffs)
    if alternative == "two-sided":
        extreme_counts *= 2 
    extreme_counts = sum(np.abs(bootstrapped_diffs-observed_stat) > np.abs(observed_stat))/len(bootstrapped_diffs)
    pval_boot = extreme_counts / n_bootstraps

    # rank test
    st, pval_rank = mannwhitneyu(data1_proc, data2_proc, alternative=alternative)

    return lower, upper, pval_boot, bootstrapped_diffs, pval_rank, observed_stat

def cramers_v(cat1, cat2):
    """
    Calculate Cramér's V for categorical-categorical association strength.

    Args:
        cat1 (array-like): array with categorical variable 1
        cat2 (array-like): array with categorical variable 2

    Returns:
        float: Cramer coefficient of association
    """
    confusion_matrix = pd.crosstab(cat1, cat2)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

def eta_squared(num_col, cat_col):
    """
    Compute Normalized Eta-Squared (η²) for Num-Cat correlation and one way ANOVA pval.

    Args:
        num_col (array-like): array with numerical variable.
        cat_col (array-like): array with categorical variable.

    Returns:
        float, float: Eta-Squared, p-value
    """
    # create independant groups
    categories = np.unique(cat_col)
    groups = [num_col[cat_col == cat] for cat in categories]
    # perform one way ANOVA
    _, p_value = f_oneway(*groups)
    ss_between = sum(len(group) * (group.mean() - num_col.mean()) ** 2 for group in groups)
    ss_total = sum((num_col - num_col.mean()) ** 2)
    eta_sq = ss_between / ss_total if ss_total != 0 else 0  # Normalize to [0,1]
    return np.sqrt(eta_sq), p_value  # Taking sqrt to make it more comparable to correlation [-1,1]

def correlation_with_pvalues(df):
    """
    Calculates the correlation coefficients for mixed types data (num and cat)
    - pearson for num-num
    - cramer V and chi square test for cat-cat
    - eta squared or Point-Biserial and ANOVA for cat-num

    As well as the pvalues.

    Args:
        df (pd.DataFrame): pandas dataframe with features as columns

    Returns:
        np.array, np.array: correlation matrix, pvalue matrix
    """
    cols = df.columns
    corr_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    pvalue_matrix = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i >= j:  # Avoid duplicate computations
                continue
            if (col1 == "scam_flag") or (col2 == "scam_flag"):
                continue

            is_cat1 = df[col1].dtype in ['bool', 'object', 'category'] #or df[col1].nunique() < 10
            is_cat2 = df[col2].dtype in ['bool', 'object', 'category'] #or df[col2].nunique() < 10

            try:
                # num-num case
                if not is_cat1 and not is_cat2: 
                    corr_value, p_value = pearsonr(df[col1], df[col2])

                # cat-cat case
                elif is_cat1 and is_cat2:
                    corr_value = cramers_v(df[col1], df[col2])
                    _, p_value, _, _ = chi2_contingency(pd.crosstab(df[col1], df[col2]))

                # num-cat case
                else:  
                    num_col, cat_col = (col1, col2) if not is_cat1 else (col2, col1)  # Ensure num is first
                    categories = df[cat_col].unique()
                    
                    if len(categories) == 2:  # Binary category -> Use Point-Biserial Correlation
                        labels = LabelEncoder().fit_transform(df[cat_col])  # Encode categorical
                        corr_value, p_value = pointbiserialr(df[num_col], labels)
                    else:  # Multi-category -> Use Eta-Squared (η²)
                        corr_value, p_value = eta_squared(df[num_col], df[cat_col])
            
            except:
                print(f"Problem with {col1},  <---> , {col2}")
                corr_value = None
                p_value = None
            finally:
                corr_matrix.loc[col1, col2] = corr_matrix.loc[col2, col1] = corr_value
                pvalue_matrix.loc[col1, col2] = pvalue_matrix.loc[col2, col1] = p_value

    return corr_matrix, pvalue_matrix

def plot_masked_corr_matrix(corr_matrix, pvalue_matrix, pval_thresh=0.05, effect_size_thresh=0.2):
    
    # Mask non-significant and small correlation values
    significance_mask = pvalue_matrix > pval_thresh
    effect_size_mask = np.abs(corr_matrix) < effect_size_thresh
    upper_triangle_mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    total_mask = significance_mask | upper_triangle_mask | effect_size_mask

    # Plot heatmap of correlation coefficients
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.abs(corr_matrix), annot=False, fmt=".2f", cmap="coolwarm", mask=total_mask, vmin=0.2, vmax=1)
    plt.title("Correlation Coefficients with Significance Masked")

def stat_cat(cat1, cat2):

    # Create a contingency table
    contingency_table = pd.crosstab(cat1, cat2)

    # Perform the Chi-Square Test
    chi2, pval, dof, expected = chi2_contingency(contingency_table)

    # Calculate Cramér's V
    cramers_v_coeff = association(contingency_table, method="cramer")

    return pval, cramers_v_coeff