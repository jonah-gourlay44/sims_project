from os.path import dirname, join
import argparse

from bokeh.plotting import figure, output_file, show
from bokeh.io import curdoc
from bokeh.layouts import column, layout
from bokeh.models import ColumnDataSource, Div, Select

from fem_1d_Electrostatic_Analysis_Ver_1_mesh_study import *

def select_plot(plot_dict, plot, plot_type):
    plot_val = plot.value
    which_plot = plot_type.value
    selected = plot_dict[plot_val][which_plot]

    return selected

def update(plot_dict, plot, plot_type, source, p):
    df = select_plot(plot_dict, plot, plot_type)
    #x_name = x_names[plot.value]
    #y_name = y_names[plot.value]

    which_plot = plot_type.value


    if which_plot == 'mesh':
        x_name = 'Position (cm)'
        y_name = ''
    elif which_plot == 'mat props':
        x_name = 'Position (cm)'
        y_name = 'eps_r'
    elif which_plot == 'electric potential':
        x_name = 'Position (cm)'
        y_name = 'Phi (V)'
    elif which_plot == 'electric field':
        x_name = 'Position (cm)'
        y_name = 'Ex (V/m)'
    elif which_plot == 'flux density':
        x_name = 'Position (cm)'
        y_name = 'Dx (C/m^2)'


    p.xaxis.axis_label = x_name
    p.yaxis.axis_label = y_name
    p.title.text = plot_type.value 
    source.data = dict(x=df[0], y=df[1])

def main():

    fem = fem_analysis()
    fem.main()

    plot_dict = fem.data   

    desc = Div(text=open(join(dirname(__file__), "description.html")).read(), sizing_mode='stretch_width')

    plot = Select(title='nodes', value='0', options=[str(i) for i in range(plot_dict['n1_'])])
    plot_type = Select(title='plot', value='mesh', options=['mesh', 'mat props', 'electric potential', 'electric field', 'flux density'])

    source = ColumnDataSource(data=dict(x=[], y=[]))

    p = figure(plot_height=600, plot_width=700, title="", sizing_mode="scale_both")
    p.circle(x="x", y="y", source=source, size=7)
    p.line(x="x", y="y", source=source, line_width=2, line_alpha=1)

    controls = [plot, plot_type]
    for control in controls:
        control.on_change('value', lambda attr, old, new: update(plot_dict, plot, plot_type, source, p))

    inputs = column(*controls, width=320, height=1000)
    inputs.sizing_mode = "fixed"
    l = layout([
        [desc],
        [inputs, p],
    ], sizing_mode='scale_both')

    update(plot_dict, plot, plot_type, source, p)

    curdoc().add_root(l)
    curdoc.title = 'plot'

main()
