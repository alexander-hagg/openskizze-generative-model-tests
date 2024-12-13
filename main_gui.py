import dash
from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import os
import threading

# Import dataset preparation functions
from data.dataset_preparation import generate_synthetic_data

# Import training functions
from training.train_gan import train as train_gan
from training.train_vae import train as train_vae
from training.train_autoregressive import train as train_autoregressive

# Import inference functions
from inference.infer_gan import infer as infer_gan
from inference.infer_vae import infer as infer_vae
from inference.infer_autoregressive import infer as infer_autoregressive

# Initialize Dash app with multi-page support
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Define models
models = ['3D-GAN', 'Voxel VAE', 'Autoregressive Model']

# Shared state for progress tracking
progress_data = {
    'dataset': {'status': '', 'progress': 0},
    'train': {'status': '', 'progress': 0},
    'infer': {'status': '', 'progress': 0},
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

# Define the layout with Navbar, page content, and separate Interval components
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    dbc.Container(id='page-content', fluid=True, className='mt-4'),
    # Separate Interval components for each task
    dcc.Interval(
        id='dataset-interval',
        interval=1*1000,  # 1 second
        n_intervals=0
    ),
    dcc.Interval(
        id='train-interval',
        interval=1*1000,  # 1 second
        n_intervals=0
    ),
    dcc.Interval(
        id='infer-interval',
        interval=1*1000,  # 1 second
        n_intervals=0
    ),
    # Store components to track if visualization has been done
    dcc.Store(id='dataset-visualization-done', data=False),
    dcc.Store(id='infer-visualization-done', data=False),  # If needed
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
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='dataset-visualization', className='mt-4')  # Div for visualization
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
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    prevent_initial_call=True,
    allow_duplicate=True
)
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
def update_progress(task, progress, status_message):
    with progress_lock:
        if progress == -1:
            progress_data[task]['status'] = f"{status_message} failed."
            progress_data[task]['progress'] = 0
        elif progress == 100:
            progress_data[task]['status'] = f"{status_message} completed."
            progress_data[task]['progress'] = 100
        else:
            progress_data[task]['status'] = f"{status_message}... {progress}%"
            progress_data[task]['progress'] = progress

# Helper function to create voxel mesh with correct normals
def create_voxel_mesh(voxel, color='blue', voxel_size=1):
    """
    Create a Mesh3d object representing the voxel grid with normals pointing outwards.
    
    Parameters:
    - voxel (numpy.ndarray): 3D binary numpy array indicating occupied voxels.
    - color (str): Color of the voxels.
    - voxel_size (int or float): Size of each voxel cube.
    
    Returns:
    - go.Mesh3d: Plotly Mesh3d object.
    - float: Maximum coordinate value for axis scaling.
    """
    vertices = []
    faces = []
    i, j, k = voxel.nonzero()
    vertex_map = {}
    vertex_count = 0

    # Define the 8 vertices of a unit cube
    cube_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    for idx in range(len(i)):
        x, y, z = i[idx], j[idx], k[idx]
        # Compute the absolute positions of the cube's vertices
        cube_abs_vertices = [[x * voxel_size + v[0] * voxel_size, 
                               y * voxel_size + v[1] * voxel_size, 
                               z * voxel_size + v[2] * voxel_size] for v in cube_vertices]
        
        # Add vertices to the list and keep track of indices
        current_indices = []
        for v in cube_abs_vertices:
            key = tuple(v)
            if key not in vertex_map:
                vertices.append(v)
                vertex_map[key] = vertex_count
                current_indices.append(vertex_count)
                vertex_count += 1
            else:
                current_indices.append(vertex_map[key])
        
        # Define faces with correct winding order (counter-clockwise)
        # Each face has two triangles
        # Bottom Face
        faces += [
            current_indices[2], current_indices[1], current_indices[0],
            current_indices[3], current_indices[2], current_indices[0],
        ]
        # Top Face
        faces += [
            current_indices[6], current_indices[5], current_indices[4],
            current_indices[7], current_indices[6], current_indices[4],
        ]
        # Front Face
        faces += [
            current_indices[5], current_indices[1], current_indices[0],
            current_indices[4], current_indices[5], current_indices[0],
        ]
        # Back Face
        faces += [
            current_indices[7], current_indices[3], current_indices[2],
            current_indices[6], current_indices[7], current_indices[2],
        ]
        # Left Face
        faces += [
            current_indices[7], current_indices[4], current_indices[0],
            current_indices[7], current_indices[0], current_indices[3],
        ]
        # Right Face
        faces += [
            current_indices[6], current_indices[2], current_indices[1],
            current_indices[6], current_indices[1], current_indices[5],
        ]
    
    vertices = np.array(vertices)
    faces = np.array(faces)

    # Determine the maximum coordinate value for axis scaling
    max_coord = voxel.shape[0] * voxel_size  # Assuming voxel is cubic
    
    # Create Mesh3d
    mesh = go.Mesh3d(
        x=vertices[:,0],
        y=vertices[:,1],
        z=vertices[:,2],
        i=faces[::3],
        j=faces[1::3],
        k=faces[2::3],
        color=color,
        opacity=0.9,
        flatshading=True,
        showscale=False,
        lighting=dict(
            ambient=0.5,
            diffuse=0.5,
            fresnel=0.1,
            specular=0.5,
            roughness=0.9,
            facenormalsepsilon=1e-6
        ),
        lightposition=dict(
            x=100,
            y=200,
            z=0
        )
    )
    
    return mesh, max_coord


# Define functions to run in separate threads
def run_generate_dataset():
    task_name = "dataset"
    status_message = "Generating Dataset"
    def progress_callback(p):
        update_progress(task_name, p, status_message)
    try:
        generate_synthetic_data(progress_callback=progress_callback, num_samples=1000, voxel_size=32, save_dir='data/processed/')
        update_progress(task_name, 100, status_message)
    except Exception as e:
        print(f"Error during dataset generation: {e}")
        update_progress(task_name, -1, status_message)

def run_train(model):
    task_name = "train"
    status_message = f"Training {model}"
    def progress_callback(p):
        update_progress(task_name, p, status_message)
    try:
        if model == '3D-GAN':
            train_gan(progress_callback)
        elif model == 'Voxel VAE':
            train_vae(progress_callback)
        elif model == 'Autoregressive Model':
            train_autoregressive(progress_callback)
        update_progress(task_name, 100, status_message)
    except Exception as e:
        print(f"Error during training {model}: {e}")
        update_progress(task_name, -1, status_message)

def run_infer(model):
    task_name = "infer"
    status_message = f"Generating Designs using {model}"
    def progress_callback(p):
        update_progress(task_name, p, status_message)
    try:
        if model == '3D-GAN':
            infer_gan(progress_callback)
        elif model == 'Voxel VAE':
            infer_vae(progress_callback)
        elif model == 'Autoregressive Model':
            infer_autoregressive(progress_callback)
        update_progress(task_name, 100, status_message)
    except Exception as e:
        print(f"Error during inference {model}: {e}")
        update_progress(task_name, -1, status_message)

# Callback for generating dataset
@app.callback(
    [
        Output('dataset-status-label', 'children'),
        Output('dataset-progress-bar', 'value'),
        Output('dataset-progress-bar', 'label'),
        Output('dataset-visualization-done', 'data'),  # Reset visualization flag
    ],
    Input('generate-dataset-button', 'n_clicks'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def handle_generate_dataset(n_clicks):
    if n_clicks:
        # Reset the visualization done flag
        visualization_done = False
        # Start dataset generation in a new thread
        thread = threading.Thread(target=run_generate_dataset)
        thread.start()
        return "Generating Dataset... 0%", 0, "0%", visualization_done
    else:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Callback for training models
@app.callback(
    [
        Output('train-status-label', 'children'),
        Output('train-progress-bar', 'value'),
        Output('train-progress-bar', 'label'),
    ],
    Input('train-button', 'n_clicks'),
    State('train-model-select', 'value'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def handle_train_button(n_clicks, model):
    if n_clicks:
        # Start training in a new thread
        thread = threading.Thread(target=run_train, args=(model,))
        thread.start()
        return f"Training {model}... 0%", 0, "0%"
    else:
        return dash.no_update, dash.no_update, dash.no_update

# Callback for generating designs
@app.callback(
    [
        Output('infer-status-label', 'children'),
        Output('infer-progress-bar', 'value'),
        Output('infer-progress-bar', 'label'),
    ],
    Input('generate-button', 'n_clicks'),
    State('infer-model-select', 'value'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def handle_generate_button(n_clicks, model):
    if n_clicks:
        # Start inference in a new thread
        thread = threading.Thread(target=run_infer, args=(model,))
        thread.start()
        return f"Generating Designs using {model}... 0%", 0, "0%"
    else:
        return dash.no_update, dash.no_update, dash.no_update

# Callback to update dataset progress bar
@app.callback(
    [
        Output('dataset-status-label', 'children', allow_duplicate=True),
        Output('dataset-progress-bar', 'value', allow_duplicate=True),
        Output('dataset-progress-bar', 'label', allow_duplicate=True),
    ],
    Input('dataset-interval', 'n_intervals'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def update_dataset_progress(n):
    with progress_lock:
        status = progress_data['dataset']['status']
        progress = progress_data['dataset']['progress']
    if progress > 0:
        label = f"{progress}%"
    else:
        label = ""
    return status, progress, label

# Callback to update training progress bar
@app.callback(
    [
        Output('train-status-label', 'children', allow_duplicate=True),
        Output('train-progress-bar', 'value', allow_duplicate=True),
        Output('train-progress-bar', 'label', allow_duplicate=True),
    ],
    Input('train-interval', 'n_intervals'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def update_train_progress(n):
    with progress_lock:
        status = progress_data['train']['status']
        progress = progress_data['train']['progress']
    if progress > 0:
        label = f"{progress}%"
    else:
        label = ""
    return status, progress, label

# Callback to update inference progress bar
@app.callback(
    [
        Output('infer-status-label', 'children', allow_duplicate=True),
        Output('infer-progress-bar', 'value', allow_duplicate=True),
        Output('infer-progress-bar', 'label', allow_duplicate=True),
    ],
    Input('infer-interval', 'n_intervals'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def update_infer_progress(n):
    with progress_lock:
        status = progress_data['infer']['status']
        progress = progress_data['infer']['progress']
    if progress > 0:
        label = f"{progress}%"
    else:
        label = ""
    return status, progress, label

# Callback to display dataset items after generation (only once)
@app.callback(
    [
        Output('dataset-visualization', 'children', allow_duplicate=True),
        Output('dataset-visualization-done', 'data', allow_duplicate=True),
    ],
    [
        Input('dataset-progress-bar', 'value')
    ],
    [
        State('dataset-visualization-done', 'data')
    ],
    prevent_initial_call=True
)
def display_dataset_items(progress, visualization_done):
    if progress == 100 and not visualization_done:
        print(f'Progress equals 100, displaying dataset items')
        samples_file = 'data/processed/all_samples.npy'  # Path to the samples file
        
        # Check if the file exists
        if not os.path.exists(samples_file):
            return html.Div(f"Processed data file {samples_file} does not exist.", style={'color': 'red'}), visualization_done
        
        try:
            # Load the voxel samples
            voxels = np.load(samples_file)
            print(f'Loaded voxel data from {samples_file}: shape={voxels.shape}')
            
            # Ensure there are at least 9 samples
            n_samples = voxels.shape[0]
            n_display = 9
            if n_samples < n_display:
                n_display = n_samples
                print(f"Only {n_samples} samples available. Displaying all available samples.")
            
            # Select the first 9 samples
            selected_voxels = voxels[:n_display]
            
            # Create a list to hold the graph components
            graphs = []
            
            for idx, voxel in enumerate(selected_voxels):
                # Create a Mesh3d for the voxel grid with correct normals
                mesh, max_coord = create_voxel_mesh(voxel, color='blue', voxel_size=1)

                fig = go.Figure(data=[mesh])
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(range=[0, max_coord], autorange=False),
                        yaxis=dict(range=[0, max_coord], autorange=False),
                        zaxis=dict(range=[0, max_coord], autorange=False),
                        aspectmode='cube'  # Ensures equal scaling on all axes
                    ),
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False
                )
                
                # Create a card for each voxel
                card = dbc.Card([
                    dbc.CardHeader(f"Voxel Design {idx + 1}"),
                    dbc.CardBody(
                        dcc.Graph(
                            figure=fig,
                            config={'displayModeBar': False},
                            style={'width': '400px', 'height': '400px'}
                        )
                    )
                ], className='m-2')
                
                graphs.append(dbc.Col(card, width='auto'))
            
            # Arrange the graphs in a 3x3 grid
            grid = dbc.Row(graphs, justify='start')
            
            return grid, True  # Set visualization_done to True after rendering
        
        except Exception as e:
            print(f"Error loading or visualizing voxel data: {e}")
            return html.Div(f"Error loading voxel data: {e}", style={'color': 'red'}), visualization_done
    else:
        return dash.no_update, dash.no_update

# Callback to display designs grid after inference completes
@app.callback(
    Output('designs-grid', 'children'),
    Input('infer-progress-bar', 'value'),
    State('infer-model-select', 'value'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def display_designs(progress, model):
    if progress == 100:
        designs = []
        output_dir = 'outputs'
        
        # Check if the directory exists
        if not os.path.exists(output_dir):
            return html.Div("Outputs directory does not exist.", style={'color': 'red'})
        
        # Define number of designs to display
        n_designs = 9
        count = 0
        
        # Iterate through the generated voxel files
        for i in range(1, 1001):  # Assuming up to 1000 designs
            if count >= n_designs:
                break
            filename = os.path.join(output_dir, f'generated_voxel_{model.lower()}_{i}.npy')
            if os.path.exists(filename):
                voxel = np.load(filename)
                
                # Create a Mesh3d for the voxel grid with correct normals
                mesh = create_voxel_mesh(voxel, color='red')
                
                # Create a card for each voxel
                card = dbc.Card([
                    dbc.CardHeader(f"Generated Voxel {i}"),
                    dbc.CardBody(
                        dcc.Graph(
                            figure=go.Figure(data=[mesh]),
                            config={'displayModeBar': False},
                            style={'width': '200px', 'height': '200px'}
                        )
                    )
                ], className='m-2')
                
                designs.append(dbc.Col(card, width='auto'))
                count += 1
            else:
                # If voxel data not found
                designs.append(
                    html.Div(
                        "Voxel data not found.",
                        style={
                            'width': '200px',
                            'height': '200px',
                            'display': 'inline-block',
                            'color': 'red'
                        },
                        className='m-2'
                    )
                )
                count += 1
        grid = dbc.Row(designs, justify='start', no_gutters=True)
        return grid
    else:
        return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
