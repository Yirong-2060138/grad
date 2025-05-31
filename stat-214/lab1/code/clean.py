import pandas as pd
import numpy as np
import re
import os

def print_missing_value(df):
    """
    Prints missing values for each column in the given DataFrame.
    If no missing values are found, it prints a message indicating so.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The original DataFrame (unchanged).
    """
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("no missing value")
    else:
        print("missing value for columns:")
        print(missing_values)
    return df

def find_columns_with_multiple_90_91_92(df):
    """
    Identifies columns that contain more than one of the values {90, 91, 92}.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        list: A list of column names where multiple values from {90, 91, 92} are present.
    """
    target_values = {90, 91, 92} 
    invalid_cols = [] 

    for col in df.columns:
        unique_values = set(df[col].dropna().unique()) 
        matched_values = unique_values.intersection(target_values)
        if len(matched_values) > 1:
            invalid_cols.append(col)

    return invalid_cols

def check_age_consistency(df):
    """
    Checks if 'age_in_years' and 'age_in_month' are consistent and ensures ages are below 18.
    The rule: age_in_years <= (age_in_month / 12) < age_in_years + 1
    and both values should be less than 18.

    Parameters:
        df (pd.DataFrame): The dataset to check.

    Prints:
        - Indexes of rows where 'age_in_years' and 'age_in_month' are inconsistent.
        - Indexes of rows where age is 18 or above.
    """
    # Condition 1: Check age consistency
    inconsistent_mask = ~((df["age_in_years"] <= df["age_in_month"] / 12) & 
                          (df["age_in_month"] / 12 < df["age_in_years"] + 1))

    # Condition 2: Check if age is below 18
    age_limit_mask = (df["age_in_years"] >= 18) | (df["age_in_month"] >= 18 * 12)

    # Get inconsistent rows
    inconsistent_rows = df[inconsistent_mask]
    age_violation_rows = df[age_limit_mask]

    # Print results
    if not inconsistent_rows.empty:
        print("Rows with inconsistent 'age_in_years' and 'age_in_month':")
        print(inconsistent_rows.index.tolist())
    else:
        print("All 'age_in_years' and 'age_in_month' values are consistent.")

    if not age_violation_rows.empty:
        print("Rows where age is 18 or above:")
        print(age_violation_rows.index.tolist())
    else:
        print("All ages are below 18.")

def convert_columns_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all numeric columns in the dataframe to an integer type that supports NaN values.
    Uses 'pd.Int64Dtype()' to ensure compatibility with NaN while keeping integer representation.

    Parameters:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: A dataframe where all numeric columns are converted to nullable integer types.
    """
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = df[col].astype("Int64")  # Convert to nullable Int64 (supports NaN)
    
    return df

def convert_to_snake_case(col_name):
    """
    Converts a given column name to snake_case format.

    Parameters:
        col_name (str): The original column name.
    
    Returns:
        str: The formatted column name in snake_case.
    """
    col_name = re.sub(r'(?<=[a-z])([A-Z])', r'_\1', col_name)
    col_name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', col_name)
    col_name = col_name.lower()

    return col_name

def standardize_column_names(df):
    """
    Converts all column names in the DataFrame to snake_case.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with standardized column names.
    """
    new_columns = {col: convert_to_snake_case(col) for col in df.columns}
    df = df.rename(columns=new_columns)
    
    return df

def drop_na_in_column(df, column):
    """
    Drops rows with NaN values in a specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column in which NaN values should be dropped.
    
    Returns:
        pd.DataFrame: The DataFrame with NaN values removed from the specified column.
    """
    if not isinstance(column, str):
        print("column must be str")
        return df  

    if column not in df.columns:
        print(f"colunb '{column}' not in DataFrame")
        return df

    original_rows = len(df)

    df_cleaned = df.dropna(subset=[column])

    deleted_rows = original_rows - len(df_cleaned)

    #print(f"删除了 {deleted_rows} 行，因为这些行在 '{column}' 列中包含 NaN")

    return df_cleaned

def fill_missing_based_on_condition(df, target_col, condition_cols, operator_types, condition_values, fill_value):
    """
    Fills missing values in the target column based on specified conditions.
    The conditions are applied using specified operator types (0: ==, 1: <, 2: >).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The column to fill missing values.
        condition_cols (list or str): Columns used to define the condition.
        operator_types (list or int): Operators defining the condition (0: ==, 1: <, 2: >).
        condition_values (list or int/float): Values used for comparison.
        fill_value (any): Value to fill missing entries.

    Returns:
        pd.DataFrame: The DataFrame with missing values in `target_col` filled based on conditions.
    """
    if isinstance(condition_cols, str):
        condition_cols = [condition_cols]
    if isinstance(operator_types, int):
        operator_types = [operator_types]
    if isinstance(condition_values, (int, float)):
        condition_values = [condition_values]

    if not (len(condition_cols) == len(operator_types) == len(condition_values)):
        #print("`condition_cols`, `operator_types` 和 `condition_values` 必须长度一致！")
        return df

    missing_before = df[target_col].isnull().sum()

    condition = None
    condition_descriptions = []  
    for i, col in enumerate(condition_cols):
        op_type = operator_types[i]
        val = condition_values[i]

        if op_type == 0:
            new_condition = df[col] == val
            condition_descriptions.append(f"{col} == {val}")
        elif op_type == 1:
            new_condition = df[col] < val
            condition_descriptions.append(f"{col} < {val}")
        elif op_type == 2:
            new_condition = df[col] > val
            condition_descriptions.append(f"{col} > {val}")
        else:
            #print(f"无效的 operator_type `{op_type}`，必须是 0, 1, 2 之间的整数")
            return df

        condition = new_condition if condition is None else (condition & new_condition)

    df.loc[df[target_col].isnull() & condition, target_col] = fill_value

    missing_after = df[target_col].isnull().sum()
    rows_filled = missing_before - missing_after  

    condition_string = " AND ".join(condition_descriptions)

    #print(f"变量 '{target_col}' 缺失值已填充 {fill_value}, {rows_filled} 行 (条件: {condition_string})")
    return df

def fill_based_on_condition(df, target_col, condition_cols, operator_types, condition_values, fill_value):
    """
    Fills values in the target column based on specified conditions.
    Unlike `fill_missing_based_on_condition`, this function modifies all rows meeting the condition.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The column to fill values.
        condition_cols (list or str): Columns used to define the condition.
        operator_types (list or int): Operators defining the condition (0: ==, 1: <, 2: >).
        condition_values (list or int/float): Values used for comparison.
        fill_value (any): Value to assign where condition is met.

    Returns:
        pd.DataFrame: The DataFrame with values in `target_col` modified based on conditions.
    """
    if isinstance(condition_cols, str):
        condition_cols = [condition_cols]
    if isinstance(operator_types, int):
        operator_types = [operator_types]
    if isinstance(condition_values, (int, float)):
        condition_values = [condition_values]

    if not (len(condition_cols) == len(operator_types) == len(condition_values)):
        print("`condition_cols`, `operator_types` 和 `condition_values` 必须长度一致！")
        return df

    condition = None
    condition_descriptions = []  
    for i, col in enumerate(condition_cols):
        op_type = operator_types[i]
        val = condition_values[i]

        if op_type == 0:
            new_condition = df[col] == val
            condition_descriptions.append(f"{col} == {val}")
        elif op_type == 1:
            new_condition = df[col] < val
            condition_descriptions.append(f"{col} < {val}")
        elif op_type == 2:
            new_condition = df[col] > val
            condition_descriptions.append(f"{col} > {val}")
        else:
            #print(f"无效的 operator_type `{op_type}`，必须是 0, 1, 2 之间的整数")
            return df

        condition = new_condition if condition is None else (condition & new_condition)

    rows_to_modify = df[condition].shape[0]

    df.loc[condition, target_col] = fill_value

    condition_string = " AND ".join(condition_descriptions)

    return df

def fill_preverbal_nonverbal(df, target_col):
    """
    Fills the target column with value 91 based on preverbal or nonverbal conditions.
    Conditions include:
    - Age in months < 24
    - GCS verbal score in {1, 2, 3}
    - Intubated status is 1

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The column to be filled based on conditions.
    
    Returns:
        pd.DataFrame: The modified DataFrame with updated values in `target_col`.
    """
    modified_count = 0
    age_condition = df["age_in_month"] < 24
    modified_count += age_condition.sum()
    df.loc[age_condition, target_col] = 91

    gcs_condition = df["gcs_verbal"].isin([1, 2, 3])
    modified_count += gcs_condition.sum()
    df.loc[gcs_condition, target_col] = 91

    intubated_condition = df["intubated"] == 1
    modified_count += intubated_condition.sum()
    df.loc[intubated_condition, target_col] = 91

    #print(f"变量 '{target_col}' 已更新，总共修改了 {modified_count} 行（符合 Preverbal / Nonverbal 条件）")
    
    return df

def extract_base_col(col, prefix_map):
    """
    Extracts the base column name from a prefixed column name based on a mapping dictionary.

    Parameters:
        col (str): The column name to process.
        prefix_map (dict): Dictionary mapping prefixes to base column names.
    
    Returns:
        str: The corresponding base column name if found, otherwise None.
    """
    for prefix, base in prefix_map.items():
        if col.startswith(prefix):
            return base
    return None 

def fill_not_applicable(df):
    """
    Fills columns with '92' where corresponding base columns have a value of 0 or NaN.
    Handles special cases where additional rules apply.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The modified DataFrame with '92' assigned to applicable columns.
    """
    prefix_map = {
        "ct_sed": "ct_form1",
        "loc_len": "loc_separate",
        "seiz_": "seiz",
        "ha_": "ha",
        "vomit_": "vomit",
        "ams_": "ams",
        "s_fx_palp_": "s_fx_palp",
        "s_fx_bas_": "s_fx_bas",
        "hema_": "hema",
        "clav_": "clav",
        "neuro_d_": "neuro_d",
        "osi_": "osi",
        "ct_sed_": "ct_sed",
        "ind_": "ct_done",
        "finding_": "ct_done",
        "edct": "ct_done",
        "pos_ct": "edct"
    }

    modified_count = 0 


    for col in df.columns:
        base_col = extract_base_col(col, prefix_map)  
        if base_col and base_col in df.columns:  
            condition = (df[base_col] == 0) | (df[base_col].isna())  
            rows_to_modify = df[condition].shape[0]

            df.loc[condition, col] = 92
            modified_count += rows_to_modify
            #print(f"变量 '{col}' 修改 {rows_to_modify} 行 (基于 {base_col} == 0 或 NaN)")

            if col.startswith("ha_"):
                ha_condition = (df["ha"] == 91)
                ha_rows_to_modify = df[ha_condition].shape[0]
                df.loc[ha_condition, col] = 92
                modified_count += ha_rows_to_modify
                #print(f"变量 '{col}' 额外修改 {ha_rows_to_modify} 行 (基于 ha == 91)")

    if "ct_form1" in df.columns and "ct_sed" in df.columns:
        ct_condition = (df["ct_form1"] == 0)
        ct_rows_to_modify = df[ct_condition].shape[0]
        df.loc[ct_condition, "ct_sed"] = 92
        modified_count += ct_rows_to_modify
        #print(f"特例处理: 'ct_sed' 修改 {ct_rows_to_modify} 行 (基于 ct_form1 == 0)")

    #print(f"总计修改 {modified_count} 个数据点，所有符合规则的值都已替换为 92")
    
    return df

def clean_data(df):
    """
    Cleans the dataset by filtering data, standardizing column names, filling missing values,
    and performing consistency checks.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # focus on patients who had a total GCS of 14 or 15
    df = df[df["GCSGroup"] == 2]
    df = standardize_column_names(df)
    df = df.rename(columns={"ha_verb": "ha"})
    df = df.rename(columns={"agein_years": "age_in_years"})
    df = fill_missing_based_on_condition(df, "amnesia_verb", "s_fx_bas", 0, 0, 0)
    df = fill_missing_based_on_condition(df, "amnesia_verb", "high_impact_inj_sev", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "amnesia_verb", "gcs_total", 0, 15, 0)
    df = fill_missing_based_on_condition(df, "loc_separate", "high_impact_inj_sev", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "loc_separate", "act_norm", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "loc_separate", "gcs_total", 0, 15, 0)
    df = fill_missing_based_on_condition(df, "seiz", "gcs_total", 0, 15, 0)
    df = fill_missing_based_on_condition(df, "seiz", "finding4", 0, 0, 0)
    df = fill_missing_based_on_condition(df, "seiz", "high_impact_inj_sev", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "seiz", "s_fx_bas", 0, 0, 0)
    df = fill_based_on_condition(df, "ams", ["gcs_total", "ams_agitated", "ams_sleep", "ams_slow", "ams_repeat", "ams_oth"],
                                            [0, 0, 0, 0, 0, 0], 
                                            [15, 0, 0, 0, 0, 0], 0)
    df = fill_preverbal_nonverbal(df, "amnesia_verb")
    df = fill_preverbal_nonverbal(df, "ha")
    df = fill_not_applicable(df)
    df = fill_missing_based_on_condition(df, "amnesia_verb", "s_fx_bas", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "loc_separate", "high_impact_inj_sev", 0, 3, 1)
    df = fill_missing_based_on_condition(df, "loc_separate", "finding4", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "loc_separate", "s_fx_bas", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "seiz", "loc_len", 0, 4, 1)
    df = fill_missing_based_on_condition(df, "seiz", "finding20", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "seiz", "s_fx_bas", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "seiz", "finding4", 0, 1, 1)
    df = fill_based_on_condition(df, "ams", "gcs_total", 0, 14, 1)
    df = drop_na_in_column(df, "injury_mech")
    df = drop_na_in_column(df, "empl_type")
    df = drop_na_in_column(df, "amnesia_verb")
    df = drop_na_in_column(df, "loc_separate")
    df = drop_na_in_column(df, "ha")
    df = drop_na_in_column(df, "vomit")
    df = drop_na_in_column(df, "intubated")
    df = drop_na_in_column(df, "paralyzed")
    df = drop_na_in_column(df, "sedated")
    df = drop_na_in_column(df, "s_fx_palp")
    df = drop_na_in_column(df, "font_bulg")
    df = drop_na_in_column(df, "s_fx_bas")
    df = drop_na_in_column(df, "hema")
    df = drop_na_in_column(df, "clav")
    df = drop_na_in_column(df, "neuro_d")
    df = drop_na_in_column(df, "osi")
    df = drop_na_in_column(df, "ct_form1")
    df = drop_na_in_column(df, "gender")
    df = drop_na_in_column(df, "ed_disposition")
    df = drop_na_in_column(df, "death_tbi")
    df = drop_na_in_column(df, "hosp_head")
    df = drop_na_in_column(df, "intub24head")
    df = drop_na_in_column(df, "pos_int_final")
    df = drop_na_in_column(df, "ams")
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 1, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 2, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 3, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 8, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 12, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 6, 1)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 7, 1)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 4, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 5, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 9, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 10, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 11, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 12, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 90, 2)

    finding_cols = [f"finding{i}" for i in list(range(1, 15)) + list(range(20, 24))]
    df.loc[df[finding_cols].eq(1).any(axis=1), "pos_ct"] = 1
    check_age_consistency(df)
    invalid_cols = find_columns_with_multiple_90_91_92(df)
    print("columns with mutiple 90 91 92:", invalid_cols)
    df['citbi'] = ((df['death_tbi'] == 1) | 
                   (df['neurosurgery'] == 1) | 
                   (df['hosp_head'] == 1) | 
                   (df['intub24head'] == 1)).astype(int)

    df = convert_columns_to_int(df)
    df.info()
    return df

def stab_data(df):
    """
    Cleans the dataset by filtering data, standardizing column names, filling missing values in inccorect way,
    and performing consistency checks.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame for stability check.
    """
    df = df[df["GCSGroup"] == 2]
    df = standardize_column_names(df)
    df = df.rename(columns={"ha_verb": "ha"})
    df = df.rename(columns={"agein_years": "age_in_years"})
    df = fill_missing_based_on_condition(df, "amnesia_verb", "s_fx_bas", 0, 0, 1)
    df = fill_missing_based_on_condition(df, "amnesia_verb", "high_impact_inj_sev", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "amnesia_verb", "gcs_total", 0, 15, 1)
    df = fill_missing_based_on_condition(df, "loc_separate", "high_impact_inj_sev", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "loc_separate", "act_norm", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "loc_separate", "gcs_total", 0, 15, 1)
    df = fill_missing_based_on_condition(df, "seiz", "gcs_total", 0, 15, 1)
    df = fill_missing_based_on_condition(df, "seiz", "finding4", 0, 0, 1)
    df = fill_missing_based_on_condition(df, "seiz", "high_impact_inj_sev", 0, 1, 1)
    df = fill_missing_based_on_condition(df, "seiz", "s_fx_bas", 0, 0, 1)
    df = fill_based_on_condition(df, "ams", ["gcs_total", "ams_agitated", "ams_sleep", "ams_slow", "ams_repeat", "ams_oth"],
                                            [0, 0, 0, 0, 0, 0], 
                                            [15, 0, 0, 0, 0, 0], 1)
    df = fill_preverbal_nonverbal(df, "amnesia_verb")
    df = fill_preverbal_nonverbal(df, "ha")
    df = fill_not_applicable(df)
    df = fill_missing_based_on_condition(df, "amnesia_verb", "s_fx_bas", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "loc_separate", "high_impact_inj_sev", 0, 3, 0)
    df = fill_missing_based_on_condition(df, "loc_separate", "finding4", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "loc_separate", "s_fx_bas", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "seiz", "loc_len", 0, 4, 0)
    df = fill_missing_based_on_condition(df, "seiz", "finding20", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "seiz", "s_fx_bas", 0, 1, 0)
    df = fill_missing_based_on_condition(df, "seiz", "finding4", 0, 1, 0)
    df = fill_based_on_condition(df, "ams", "gcs_total", 0, 14, 0)
    df = drop_na_in_column(df, "injury_mech")
    df = drop_na_in_column(df, "empl_type")
    df = drop_na_in_column(df, "amnesia_verb")
    df = drop_na_in_column(df, "loc_separate")
    df = drop_na_in_column(df, "ha")
    df = drop_na_in_column(df, "vomit")
    df = drop_na_in_column(df, "intubated")
    df = drop_na_in_column(df, "paralyzed")
    df = drop_na_in_column(df, "sedated")
    df = drop_na_in_column(df, "s_fx_palp")
    df = drop_na_in_column(df, "font_bulg")
    df = drop_na_in_column(df, "s_fx_bas")
    df = drop_na_in_column(df, "hema")
    df = drop_na_in_column(df, "clav")
    df = drop_na_in_column(df, "neuro_d")
    df = drop_na_in_column(df, "osi")
    df = drop_na_in_column(df, "ct_form1")
    df = drop_na_in_column(df, "gender")
    df = drop_na_in_column(df, "ed_disposition")
    df = drop_na_in_column(df, "death_tbi")
    df = drop_na_in_column(df, "hosp_head")
    df = drop_na_in_column(df, "intub24head")
    df = drop_na_in_column(df, "pos_int_final")
    df = drop_na_in_column(df, "ams")
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 1, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 2, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 3, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 8, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 12, 3)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 6, 1)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 7, 1)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 4, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 5, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 9, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 10, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 11, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 12, 2)
    df = fill_based_on_condition(df, "high_impact_inj_sev", "injury_mech", 0, 90, 2)

    finding_cols = [f"finding{i}" for i in list(range(1, 15)) + list(range(20, 24))]
    df.loc[df[finding_cols].eq(1).any(axis=1), "pos_ct"] = 1
    check_age_consistency(df)
    invalid_cols = find_columns_with_multiple_90_91_92(df)
    print("columns with mutiple 90 91 92:", invalid_cols)
    df['citbi'] = ((df['death_tbi'] == 1) | 
                   (df['neurosurgery'] == 1) | 
                   (df['hosp_head'] == 1) | 
                   (df['intub24head'] == 1)).astype(int)

    df = convert_columns_to_int(df)
    df.info()
    return df


if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "TBI PUD 10-08-2013.csv")
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "TBI_cleaned.csv")
    output_path2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "TBI_stability.csv")

    df = pd.read_csv(data_path)
    df_cleaned = clean_data(df)
    df_stab = stab_data(df)
    df_cleaned.to_csv(output_path, index=False)
    df_stab.to_csv(output_path2, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Cleaned data for stability check saved to {output_path2}")