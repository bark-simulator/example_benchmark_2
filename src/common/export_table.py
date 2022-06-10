# Copyright (c) 2019 The diadem authors 
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================


import pandas as pd
import os
import numpy as np
import re

_pmsign = "(pm)"


def _convertDataToStringDict(data):
    new_dict_list = []
    for dict in data:
        new_dict = {}
        for k, v in dict.items():
            new_dict[k] = _convertToString(v)
        new_dict_list.append(new_dict)
    return new_dict_list


def _convertToString(v, mark_bold=False, precision=2):
    if v is None or str(v) == 'nan':
        return '-'
    elif isinstance(v, int):
        return 'i%d' % v
    elif isinstance(v, float):
        float_prec = 1
        # if 'float_prec' in kwargs:
        #	float_prec = kwargs["float_prec"]
        if mark_bold:
            return "bold{1:,.{0}f}".format(precision,v)
        else:
            return "f{1:,.{0}f}".format(precision,v)
    elif isinstance(v, str):
        return v
    elif isinstance(v, dict):
        return _convertDataToStringDict(v)

def _convertDataToStringWithStd(data_mean, data_std, **kwargs):
    st = _convertToString(data_mean) + _pmsign + _convertToString(data_std)
    return st


def _formatData(v):
    if '-' in v:
        return '-'
    if 'i' in v:
        v = v.replace('i', '')
    if 'f' in v:
        v = v.replace('f', '')
    if _pmsign in v:
        v = v.replace(_pmsign, '$\pm$')
    if "bold" in v:
        v = "\textbf{{}}".format(v)
    return v


def export_table(filename, **kwargs):

    if "mark_lowest_columns" in kwargs:
        mark_lowest_columns = kwargs["mark_lowest_columns"]
    else:
        mark_lowest_columns = None

    if "mark_highest_columns" in kwargs:
        mark_highest_columns = kwargs["mark_highest_columns"]
    else:
        mark_highest_columns = None

    if "mark_group" in kwargs:
        mark_group = kwargs["mark_group"]
    else:
        mark_group = None

    if "precision" in kwargs:
        precision_param = kwargs["precision"]
    else:
        precision_param = 2


    # todo: add row suppoert

    if "index_tuples" in kwargs and "index_names" in kwargs and "dict_list" in kwargs: # a tuple list, and index names as well as data given
        index_tuples, index_names, dict_list = kwargs["index_tuples"], kwargs["index_names"], kwargs["dict_list"]
        data = _convertDataToStringDict(dict_list)
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
        data_frame = pd.DataFrame(data, index=multi_index)
    elif "df_mean" in kwargs and "df_std" in kwargs: # two data frames with same shape given, one for mean, one for std
        df_mean, df_std = kwargs["df_mean"], kwargs["df_std"]
        if not df_mean.shape == df_std.shape:
            print("Error: Different shapes for mean and std.")
            return
        for idx,row in df_mean.iterrows():
            for col in row.index:
                df_mean.loc[idx, col] = _convertDataToStringWithStd(df_mean.loc[idx,col], df_std.loc[idx,col])
        data_frame = df_mean
    elif "df" in kwargs: # just a data frame given
        df = kwargs["df"].copy()
        df_orig = kwargs["df"].copy()

        def mark_data_frame(df):
            for idx,row in df.iterrows():
                for col in row.index:
                   mark_bold = False
                   if mark_highest_columns and col in mark_highest_columns:
                       if df_orig[col].idxmax() == idx:
                           mark_bold = True
                   if mark_lowest_columns and col in mark_lowest_columns:
                       if df_orig[col].idxmin() == idx:
                           mark_bold = True
                   precision = precision_param
                   if isinstance(precision_param, dict) and col in precision_param:
                       precision = precision_param[col]

                   df.loc[idx, col] = _convertToString(df.loc[idx,col],mark_bold, precision)
            return df

        if mark_group:
            df.groupby(mark_group).apply(mark_data_frame)

        else:
            df = mark_data_frame(df)

        data_frame = df

    if "replace_dict" in kwargs:
        replace_dict = kwargs["replace_dict"]
        if "replace_exact" in kwargs:
            replace_exact = kwargs["replace_exact"]
        else:
            replace_exact = False
        if replace_exact:
            data_frame = data_frame.replace(to_replace=replace_dict)
        else:
            for idx, row in data_frame.iterrows():
                for col in row.index:
                    for k, v in replace_dict.items():
                        if k in str(data_frame.loc[idx, col]):
                            data_frame.loc[idx, col] = v


    if "IdxAsRow" in kwargs:
        data_frame = data_frame.transpose()

    if "replace_dict" in kwargs:
        replace_dict = kwargs["replace_dict"]
        data_frame = data_frame.rename(replace_dict)
        data_frame = data_frame.rename(index=str, columns=replace_dict)
        for i,level in list(enumerate(data_frame.index.levels)):
            name = level.name
            if name in replace_dict:
                data_frame.index.set_names(replace_dict[name], level=i,inplace=True)

    if "index" in kwargs:
        index = kwargs["index"]
    else:
        index = False
    tex_string = data_frame.to_latex(multirow=True, multicolumn=True, formatters=[_formatData] * data_frame.columns.size, index = index,escape=False)

    if "latex_column_def" in kwargs:
        latex_column_def = kwargs["latex_column_def"]
        m = re.search('{l*}',tex_string)
        if m is not None:
            first_half = tex_string[:m.span()[0]+1]
            second_half = tex_string[m.span()[1]-1:]
            tex_string = "{}{}{}".format(first_half, latex_column_def, second_half)


    if "spacing_midrule" in kwargs:
        spacing_midrule = kwargs["spacing_midrule"]
    else:
        spacing_midrule = True

    if spacing_midrule:
        tex_string = tex_string.replace("\cline","\cmidrule")

    if not ".tex" in filename:
        filename += ".tex"
    with open(filename, "w") as text_file:
        text_file.write(tex_string)