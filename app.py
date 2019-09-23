import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.axes3d import Axes3D  # impt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn.metrics.pairwise import euclidean_distances
import json
from dash.exceptions import PreventUpdate

app = dash.Dash()

df1 = pd.read_csv('/heart/egm_2018_21_52_36.csv', engine='python')
df = np.array(df1)

scatter = pd.read_csv('/heart/3D_21_52_36.csv', engine='python')

app.layout = html.Div(
    [
        html.Div([
            dcc.Graph(
                id='heart-graph',
                figure={
                    'data': [
                        go.Scatter3d(
                            x=scatter['ptx'],
                            y=scatter['pty'],
                            z=scatter['ptz'],
                            mode='markers',
                            marker={
                                'size': 7,
                                'color': scatter['time'],
                                'colorscale': 'YlGnBu'})],

                    'layout': go.Layout(
                        margin={'l': 10, 'b': 35, 't': 35, 'r': 35},
                        hovermode='closest')

                }
            )],
            className="six columns",
         #style={'width': '80%', 'display': 'inline-block', 'padding': '0 20'}
        ),


        html.Div([
            dcc.Store(id='store', storage_type='local'),
            html.Button('Normal', id='normal', n_clicks_timestamp='0'),
            html.Button('Split', id='split', n_clicks_timestamp='0'),
            html.Button('Noise', id='noise', n_clicks_timestamp='0'),
            html.Button('Unclassified', id='unclassified', n_clicks_timestamp='0'),
            html.Div(id='output'), #style={'display': 'none'}


        ]),


        html.Div([
            dcc.Markdown(("""
                **Click Data**

                Click on points in the graph.
            """)),
            html.Pre(id='click-data'),
        ], className='three columns'),

        html.Div([
            dcc.Graph(id='egm')
        ], style={'display': 'inline-block', 'width': '45%'}),

        #html.Div([
        #    dcc.Graph(id='egm_nn1')
        #], style={'display': 'inline-block', 'width': '25%'}),

        html.Div([
            html.Div([dcc.Graph(id = 'left-graph')], className='six columns'),
            html.Div([dcc.Graph(id = 'right-top-graph')], className='six columns'),
            html.Div([dcc.Graph(id = 'right-bottom-graph')], className='six columns'),], className='row')

        ])


app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})


#############return idx of selected point
@app.callback(
    Output('click-data', 'children'),
    [Input('heart-graph', 'clickData')])
def display_clicked_point(clickData):
    global idx
    if clickData:
        idx = clickData['points'][0]['pointNumber']
    return idx

##################egm of selected point
@app.callback(
    Output('egm', 'figure'),
    [Input('heart-graph', 'clickData')])
def plot_egm(clickData):
    # trace_egm = None
    if clickData:
       # id = clickData['points'][0]['pointNumber']
        x = np.array([range(0, 2035)]).reshape(-1, 1)
        y = df[:,idx].reshape(-1,1)
        yx = np.column_stack((x, y))
    return {'data': [go.Scatter(
                  x=yx[:, 0],
                  y=yx[:, 1],
                  mode='lines'
    )],
        'layout': {
            'title': idx}

    }

#######################find nn
def nearest_neighbour(clickData, neighbour=4):
     if clickData:
        temp_egm = []
        idx = clickData['points'][0]['pointNumber']
        x = scatter['ptx'][idx]
        y = scatter['pty'][idx]
        z = scatter['ptz'][idx]
        coord = np.array([x, y, z]).reshape(1, -1)
        distance_mat = euclidean_distances(coord, scatter.iloc[:, [0, 1, 2]]).reshape(-1, 1)
        rank = sorted(range(len(distance_mat)), key=lambda i: distance_mat[i])[:neighbour]
       # return rank #return concatenated str
        for i in rank:
            temp_egm.append(df[:, i])
        egm_df = np.asanyarray(temp_egm).T
     return egm_df


####################plot top nn
@app.callback(
    Output('left-graph', 'figure'),
    [Input('heart-graph', 'clickData')])
def plot_nn(clickData):
    x = np.array([range(0, 2035)]).reshape(-1, 1)
    y = nearest_neighbour(clickData)[:,1].reshape(-1, 1)
    yx = np.column_stack((x, y))
    return {
            'data': [go.Scatter(x=yx[:,0],
                                y=yx[:,1],
                                mode='lines'
                                )],
            'layout': {
                'title': 'egm_nn_1'
            }
        }

########################plot 2nd nn
@app.callback(
    Output('right-top-graph', 'figure'),
    [Input('heart-graph', 'clickData')])
def plot_nn(clickData):
    x = np.array([range(0, 2035)]).reshape(-1, 1)
    y = nearest_neighbour(clickData)[:, 2].reshape(-1, 1)
    yx = np.column_stack((x, y))
    return {
            'data': [go.Scatter(x=yx[:,0],
                                y=yx[:,1,],
                                mode='lines'
                                )],
            'layout': {
                'title': 'egm_nn_2'
            }
        }

######################3plot 3rd nn
@app.callback(
    Output('right-bottom-graph', 'figure'),
    [Input('heart-graph', 'clickData')])
def plot_nn(clickData):
    x = np.array([range(0, 2035)]).reshape(-1, 1)
    y = nearest_neighbour(clickData)[:, 3].reshape(-1, 1)
    yx = np.column_stack((x, y))
    return {
            'data': [go.Scatter(x=yx[:,0],
                                y=yx[:,1],
                                mode='lines'
                                )],
            'layout': {
                'title': 'egm_nn_3'
            }
        }



#####################store label
@app.callback(
    Output('store', 'data'),
    [Input('normal', 'n_clicks_timestamp'),
     Input('split', 'n_clicks_timestamp'),
     Input('noise', 'n_clicks_timestamp'),
     Input('unclassified', 'n_clicks_timestamp')],
    [State('store', 'data')]
)
def change_that_value(normal, split, noise, unclassified, data):
    if int(normal) == max([int(normal), int(split), int(noise), int(unclassified)]):
        data = data or {}
        data[idx] = 'normal'
    elif int(split) == max([int(normal),int(split), int(noise), int(unclassified)]):
        data[idx] = 'split'
    elif int(noise) == max([int(normal),int(noise), int(split), int(unclassified)]):
        data[idx] = 'noise'
    else:
        data[idx] = 'unclassified'
    #  elif int(unclassified) == max([int(normal), int(split), int(noise)]):
   #     data[idx] = 'unclassified'
    return data



#######################################################3
@app.callback(
    Output('output', 'children'),
    [Input('store', 'modified_timestamp')],
    [State('store', 'data')]
)
def add_to_storage(timestamp, data):
    if timestamp is None:
        raise dash.exceptions.PreventUpdate

    print('current storage is {}'.format(data))
    print('modified_timestamp is {}'.format(timestamp))
    with open('/Users/Tytyty/Desktop/BII/heaRt/result.json', 'w') as outfile:
        outfile.write(json.dumps(data, ensure_ascii=False))
    return json.dumps(data)



if __name__ == '__main__':
    app.run_server(debug=True)
