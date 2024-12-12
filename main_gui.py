import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import os
import threading
import time

# Import training functions
from training.train_gan import train as train_gan
from training.train_vae import train as train_vae
from training.train_autoregressive import train as train_autoregressive

# Import inference functions
from inference.infer_gan import infer as infer_gan
from inference.infer_vae import infer as infer_vae
from inference.infer_autoregressive import infer as infer_autoregressive

# Initialize Dash app with multi-page support
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define models
models = ['3D-GAN', 'Voxel VAE', 'Autoregressive Model']

# Shared state for progress tracking
progress_data = {
    'status': '',
    'progress': 0,
    'active_task': None  # Can be 'train', 'generate', or 'dataset'
}
progress_lock = threading.Lock()

# Define Navbar for navigation
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Generate Dataset", href="/generate-dataset")),
        dbc.NavItem(dbc.NavLink("Train Models", href="/train-models")),
        dbc.NavItem(dbc.NavLink("Generate Designs", href="/generate-designs")),
    ],
    brand="Urban Planning Generative Models",
    color="primary",
    dark=True,
    fluid=True,
)

# Define the layout with Navbar and page content
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    dbc.Container(id='page-content', fluid=True, className='mt-4')
])

# Define page layouts

# Page 1: Generate Dataset
generate_dataset_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2('Generate Dataset'), className='mb-4')
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Start Dataset Generation', id='generate-dataset-button', color='info')
        ], width='auto')
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='dataset-status-label', className='mb-2'),
            dbc.Progress(id='dataset-progress-bar', value=0, striped=True, animated=True, style={'height': '20px'})
        ], width=12)
    ])
], fluid=True)

# Page 2: Train Models
train_models_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2('Train Generative Models'), className='mb-4')
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label('Select Model to Train'),
            dcc.Dropdown(
                id='train-model-select',
                options=[{'label': model, 'value': model} for model in models],
                value='3D-GAN'
            )
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Train Model', id='train-button', color='primary', className='mr-2'),
        ], width='auto')
    ], className='my-2'),
    dbc.Row([
        dbc.Col([
            html.Div(id='train-status-label', className='mb-2'),
            dbc.Progress(id='train-progress-bar', value=0, striped=True, animated=True, style={'height': '20px'})
        ], width=12)
    ])
], fluid=True)

# Page 3: Generate Designs/Inference
generate_designs_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2('Generate Designs'), className='mb-4')
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label('Select Model for Inference'),
            dcc.Dropdown(
                id='infer-model-select',
                options=[{'label': model, 'value': model} for model in models],
                value='3D-GAN'
            )
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Generate Designs', id='generate-button', color='secondary', className='mr-2'),
        ], width='auto')
    ], className='my-2'),
    dbc.Row([
        dbc.Col([
            html.Div(id='infer-status-label', className='mb-2'),
            dbc.Progress(id='infer-progress-bar', value=0, striped=True, animated=True, style={'height': '20px'})
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='designs-grid')
        ], width=12)
    ])
], fluid=True)

# Update page content based on URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/generate-dataset':
        return generate_dataset_layout
    elif pathname == '/train-models':
        return train_models_layout
    elif pathname == '/generate-designs':
        return generate_designs_layout
    else:
        return generate_dataset_layout  # Default page

# Function to update progress safely
def update_progress(progress, status_message, page):
    with progress_lock:
        if progress == -1:
            progress_data['status'] = f"{status_message} failed."
            progress_data['progress'] = 0
            progress_data['active_task'] = None
        elif progress == 100:
            progress_data['status'] = f"{status_message} completed."
            progress_data['progress'] = 100
            progress_data['active_task'] = None
        else:
            progress_data['status'] = f"{status_message}... {progress}%"
            progress_data['progress'] = progress
            progress_data['active_task'] = status_message.lower()

# Define functions to run in separate threads
def run_generate_dataset(task_name):
    def progress_callback(p):
        update_progress(p, task_name, 'dataset')
    try:
        # Assuming there is a function generate_dataset in training module
        from data.dataset_preparation import generate_synthetic_data
        generate_synthetic_data(progress_callback=progress_callback, num_samples=1000, voxel_size=32)
    except Exception as e:
        print(f"Error during dataset generation: {e}")
        update_progress(-1, task_name, 'dataset')

def run_train(model, task_name):
    def progress_callback(p):
        update_progress(p, task_name, 'train')
    try:
        if model == '3D-GAN':
            train_gan(progress_callback)
        elif model == 'Voxel VAE':
            train_vae(progress_callback)
        elif model == 'Autoregressive Model':
            train_autoregressive(progress_callback)
    except Exception as e:
        print(f"Error during training {model}: {e}")
        update_progress(-1, task_name, 'train')

def run_infer(model, task_name):
    def progress_callback(p):
        update_progress(p, task_name, 'infer')
    try:
        if model == '3D-GAN':
            infer_gan(progress_callback)
        elif model == 'Voxel VAE':
            infer_vae(progress_callback)
        elif model == 'Autoregressive Model':
            infer_autoregressive(progress_callback)
    except Exception as e:
        print(f"Error during inference {model}: {e}")
        update_progress(-1, task_name, 'infer')

# Callback for generating dataset
@app.callback(
    Output('dataset-status-label', 'children'),
    Output('dataset-progress-bar', 'value'),
    Output('dataset-progress-bar', 'label'),
    Input('generate-dataset-button', 'n_clicks'),
    prevent_initial_call=True
)
def handle_generate_dataset(n_clicks):
    if n_clicks:
        status = "Generating Dataset"
        progress = 0
        # Start dataset generation in a new thread
        thread = threading.Thread(target=run_generate_dataset, args=("Generating Dataset",))
        thread.start()
        return status, progress, ""
    else:
        return dash.no_update, dash.no_update, dash.no_update

# Callback for training models
@app.callback(
    Output('train-status-label', 'children'),
    Output('train-progress-bar', 'value'),
    Output('train-progress-bar', 'label'),
    Input('interval-component', 'n_intervals'),
)
def update_train_progress_bar(n):
    with progress_lock:
        status = progress_data['status']
        progress = progress_data['progress']
    if progress == 0:
        label = ""
    else:
        label = f"{progress}%"
    return status, progress, label

@app.callback(
    Output('train-status-label', 'children', allow_duplicate=True),
    Output('train-progress-bar', 'value', allow_duplicate=True),
    Output('train-progress-bar', 'label', allow_duplicate=True),
    Input('train-button', 'n_clicks'),
    State('train-model-select', 'value'),
    prevent_initial_call=True
)
def handle_train_button(n_clicks, model):
    if n_clicks:
        status = f"Training {model}"
        progress = 0
        label = ""
        # Start training in a new thread
        thread = threading.Thread(target=run_train, args=(model, "Training"))
        thread.start()
        return status, progress, label
    else:
        return dash.no_update, dash.no_update, dash.no_update

# Callback for generating designs
@app.callback(
    Output('infer-status-label', 'children'),
    Output('infer-progress-bar', 'value'),
    Output('infer-progress-bar', 'label'),
    Input('interval-component', 'n_intervals'),
)
def update_infer_progress_bar(n):
    with progress_lock:
        status = progress_data['status']
        progress = progress_data['progress']
    if progress == 0:
        label = ""
    else:
        label = f"{progress}%"
    return status, progress, label

@app.callback(
    Output('designs-grid', 'children'),
    Output('infer-status-label', 'children', allow_duplicate=True),
    Output('infer-progress-bar', 'value', allow_duplicate=True),
    Output('infer-progress-bar', 'label', allow_duplicate=True),
    Input('generate-button', 'n_clicks'),
    State('infer-model-select', 'value'),
    prevent_initial_call=True
)
def handle_generate_button(n_clicks, model):
    if n_clicks:
        status = f"Generating Designs using {model}"
        progress = 0
        # Start inference in a new thread
        thread = threading.Thread(target=run_infer, args=(model, "Generating Designs"))
        thread.start()
        return dash.no_update, status, progress, ""
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback to display designs grid after inference completes
@app.callback(
    Output('designs-grid', 'children', allow_duplicate=True),
    Input('infer-progress-bar', 'value'),
    State('infer-model-select', 'value'),
    prevent_initial_call=True
)
def display_designs(progress, model):
    if progress == 100:
        designs = []
        output_dir = 'outputs'
        for i in range(1, 101):
            filename = os.path.join(output_dir, f'generated_voxel_{model.lower()}_{i}.npy')
            if os.path.exists(filename):
                voxel = np.load(filename)
                x, y, z = voxel.nonzero()
                fig = go.Figure(data=[go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=z,
                        colorscale='Viridis',
                    )
                )])
                fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, t=0, b=0))
                designs.append(dcc.Graph(figure=fig, style={'width': '200px', 'height': '200px', 'display': 'inline-block'}))
            else:
                designs.append(html.Div("Voxel data not found.", style={'width': '200px', 'height': '200px', 'display': 'inline-block'}))
        grid = dbc.Row([dbc.Col(design, width='auto') for design in designs], justify='start', no_gutters=True)
        return grid
    else:
        return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
