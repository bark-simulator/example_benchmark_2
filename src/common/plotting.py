import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

THESIS_TEXTWIDTH = 418.13095


def matplotlib_latex_conf(pgf_backend=True, thesis=False):
    if pgf_backend:
      matplotlib.use("pgf")
    tex_preamble = [
            r"\input{path}".replace("path", os.path.abspath("doc/common/packages")),
            r"\input{path}".replace("path", os.path.abspath("doc/common/pgfexternalize")),
            r"\input{path}".replace("path", os.path.abspath("doc/common/commands")),
            r"\input{path}".replace("path", os.path.abspath("doc/common/math_commands")),
            r"\input{path}".replace("path", os.path.abspath("doc/common/acronyms")),
            r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
            r'\usepackage[OT1]{fontenc}',    # set the normal font here
            r'\usepackage{mathpazo}',
            r'\normalfont'
        ]

    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "lualatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
    #    "mathtext.fontset" : "cm",
       # "font.family": "sans-serif",
       # "font.serif": ['Computer Modern'],  # blank entries should cause plots to inherit fonts from the document
      #  "font.sans-serif": ["DejaVu Sans"],
       # "font.monospace": ["dejavuserif"],
        "pgf.preamble": tex_preamble,
        "text.latex.preamble": tex_preamble
    }
    pgf_with_latex_thesis = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "mathtext.fontset" : "cm",
        "font.family": "serif",
        "font.serif": ['Computer Modern'],  # blank entries should cause plots to inherit fonts from the document
     #   "font.sans-serif": ["DejaVu Sans"],
      #  "font.monospace": ["Computer Modern Typewriter"],
        "pgf.preamble": tex_preamble,
        "text.latex.preamble": tex_preamble
    }
    matplotlib.rcParams.update(pgf_with_latex if not thesis else pgf_with_latex_thesis)



def GetMplStyleConfigStandard():
  return {
        "axes.labelsize": 5,  # LaTeX default is 10pt font.
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "axes.titlesize" : 5,
        "lines.linewidth": 0.5,
        "lines.markeredgewidth": 0.05,  # the line width around the marker symbol
        "lines.markersize": 0.2,
        'legend.fontsize': 5,
        'legend.handlelength': 2,
  }

def GetMplStyleConfigStandardThesis():
  return {
        "axes.labelsize": 8,  # Thesis font size is 11t font.
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.titlesize" : 9,
        "lines.linewidth": 0.5,
        "lines.markeredgewidth": 0.05,  # the line width around the marker symbol
        "lines.markersize": 0.2,
        'legend.fontsize': 9,
        'legend.handlelength': 2,
        'legend.title_fontsize' : 9
  }

def GetMplStyleConfigTrafficPaper():
  return {
        "axes.labelsize": 4,  # LaTeX default is 10pt font.
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "axes.titlesize" : 7,
        "lines.linewidth": 1.5,
        "lines.markeredgewidth": 0.05,  # the line width around the marker symbol
        "lines.markersize": 0.2,
        'legend.fontsize': 5,
        'legend.handlelength': 2,
        'hatch.linewidth' : 0.001
  }

def GetMplStyleConfigTrafficThesisSmall():
  return {
        "axes.labelsize": 4,  # LaTeX default is 10pt font.
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "axes.titlesize" : 7,
        "lines.linewidth": 0.5,
        "lines.markeredgewidth": 0.01,  # the line width around the marker symbol
        "lines.markersize": 0.1,
        'legend.fontsize': 5,
        'legend.handlelength': 2,
        'hatch.linewidth' : 0.001
  }

def GetMplStyleConfigTrafficVideo():
  return {
        "axes.labelsize": 5,  # LaTeX default is 10pt font.
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "axes.titlesize" : 4,
        "lines.linewidth": 1.5,
        "lines.markeredgewidth": 0.05,  # the line width around the marker symbol
        "lines.markersize": 0.2,
        'legend.fontsize': 5,
        'legend.handlelength': 2,
  }

def GetMplStyleConfigPolicyVideo():
  return {
        "axes.labelsize": 5,  # LaTeX default is 10pt font.
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "axes.titlesize" : 4,
        "lines.linewidth": 0.5,
        "lines.markeredgewidth": 0.05,  # the line width around the marker symbol
        "lines.markersize": 0.2,
        'legend.fontsize': 5,
        'legend.handlelength': 2,
        'hatch.linewidth' : 0.5,
        "patch.linewidth" : 0.5
  }


def GetMplStyleConfigPolicyVideo():
  return {
        "axes.labelsize": 4,  # LaTeX default is 10pt font.
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "axes.titlesize" : 4,
        "lines.linewidth": 0.5,
        "lines.markeredgewidth": 0.05,  # the line width around the marker symbol
        "lines.markersize": 0.2,
        'legend.fontsize': 5,
        'legend.handlelength': 2,
        'hatch.linewidth' : 0.5,
        "patch.linewidth" : 0.5
  }

def GetMplStyleConfigPolicyPaper():
  return {
        "axes.labelsize": 4,  # LaTeX default is 10pt font.
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
        "axes.titlesize" : 4,
        "lines.linewidth": 0.2,
        "lines.markeredgewidth": 0.05,  # the line width around the marker symbol
        "lines.markersize": 0.2,
        'legend.fontsize': 5,
        'legend.handlelength': 2,
        'hatch.linewidth' : 0.3,
        "patch.linewidth" : 0.3
  }

MPL_STYLES = {}

def GetMplStyle(type):
  if not type in MPL_STYLES:
    raise ValueError("style type {} not defined!".format(type))
  return MPL_STYLES[type]

def SetMplStyle(type, style):
  MPL_STYLES[type] = style


def latex_to_mp_figsize(latex_width_pt=252.0, scale=1.0, height_scale=None):
    fig_width_pt = latex_width_pt  # columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch        # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width  # height in inches
    fig_size = [fig_width, fig_height]
    if height_scale is not None:
        fig_size[1] *= height_scale
    return fig_size

def change_width_seaborn_barplot_bar(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def space_equally_seaborn_barplot_bars(ax, bar_range, xtick, space, bar_width):
    total_space = (len(bar_range)-1)*space+len(bar_range)*bar_width

    xticks = ax.get_xticks()
    xposition = xticks[xtick]

    lines = ax.lines
    patches_list = ax.patches
    current_x = xposition - total_space/2.0
    for patch_idx in bar_range:
        patch = patches_list[patch_idx]
        patch_x_before = patch.get_x() + patch.get_width()/2.0
        patch.set_width(bar_width)
        patch.set_x(current_x)
        for line in lines:
            line_points = line._path.vertices
            all_x_equal = np.all(line_points[:,0] == line_points[0,0], axis = 0)
            line_center = line_points[0, 0]
            if all_x_equal and abs(line_center - patch_x_before) < 0.001:
                line._path.vertices[:,0] = current_x + bar_width/2.0
        current_x += (space + bar_width)


def stacked_bar_chart(pivoted_df,hue, stack_vals, level_values_field, chart_title, x_label, y_label, color_palette, hatches, axes=None):
    #
    # stacked_bar_chart: draws and saves a barchart figure to filename
    #
    # pivoted_df: dataframe which has been pivoted so columns correspond to the values to be plotted
    # stack_vals: the column names in pivoted_df to plot
    # level_values_field: column in the dataframe which has the values to be plotted along the x axis (typically time dimension)
    # chart_title: how to title chart
    # x_label: label for x axis
    # y_label: label for y axis
    # filename: full path filename to save file
    # color1: first color in spectrum for stacked bars
    # color2: last color in spectrum for stacked bars; routine will select colors from color1 to color2 evenly spaced
    #
    # Implementation: based on (http://randyzwitch.com/creating-stacked-bar-chart-seaborn/; https://gist.github.com/randyzwitch/b71d47e0d380a1a6bef9)
    # this routine draws overlapping rectangles, starting with a full bar reaching the highest point (sum of all values), and then the next shorter bar
    # and so on until the last bar is drawn.  These are drawn largest to smallest with overlap so the visual effect is that the last drawn bar is the
    # bottom of the stack and in effect the smallest rectangle drawn.
    #
    # Here "largest" and "smallest" refer to relationship to foreground, with largest in the back (and tallest) and smallest in front (and shortest).
    # This says nothing about which part of the bar appear large or small after overlap.
    if not axes:
        plt.figure()
        axes = plt.gca()
    #
    stack_total_column = 'Stack_subtotal_xyz'  # placeholder name which should not exist in pivoted_df
    bar_num = 0
    legend_rectangles = []
    legend_names = []
    legend_colors = []
    for idx, bar_part in enumerate(stack_vals):    # for every item in the stack we need to compute a rectangle
        #stack_color = color_spectrum[bar_num].get_hex_l()  # get_hex_l ensures full hex code of color
        sub_count = 0
        pivoted_df[stack_total_column] = 0
        stack_value = ""
        for stack_value in stack_vals:  # for every item in the stack we create a new subset [stack_total_column] of 1 to N of the sub values
            pivoted_df[stack_total_column] += pivoted_df[stack_value]  # sum up total
            sub_count += 1
            if sub_count >= len(stack_vals) - bar_num:  # we skip out after a certain number of stack values
                break
        # now we have set the subtotal and can plot the bar.  reminder: each bar is overalpped by smaller subsequent bars starting from y=0 axis
        g = sns.barplot(x=level_values_field, y=stack_total_column, hue=hue, data=pivoted_df,
                palette=color_palette, ax=axes, edgecolor="k", linewidth=0.5, hatch=hatches[idx])
        # build the average color for the legend
        for idx, lh in enumerate(g.legend_.legendHandles):
            pass
        #legend_rectangles.append(plt.Rectangle((0,0),1,1,fc=stack_color, edgecolor = 'none'))  
        #legend_names.append(stack_value)   # the "last" stack_value is the name of that part of the stack
        bar_num += 1

    g.set(xlabel=x_label, ylabel=y_label)
    plt.tight_layout()
    plt.title(chart_title)
    sns.despine(left=True)