import os
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator


def plot_interactive_mu_c(save_dir, mu_values, quantity='J_iso_meV', num_neighbors=3):
    all_data = {}
    unique_c = None
    unique_b = None
    dist_vector = None
    
    print("Loading data...")
    # --- 1. Load All Data ---
    for mu in mu_values:
        file_path = os.path.join(save_dir, f"data_mu_{mu}.npz")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping mu={mu}.")
            continue
            
        data = np.load(file_path)
        all_data[mu] = {
            'c_inter': data['c_inter'],
            'b_val': data['b_val'],
            'y_data': data[quantity],
            'distances': data['distances']
        }
        
        if unique_c is None:
            unique_c = np.unique(np.round(all_data[mu]['c_inter'], 5))
            unique_b = np.unique(np.round(all_data[mu]['b_val'], 5))
            dist_vector = all_data[mu]['distances'][0]
            rounded_dist = np.round(dist_vector, 4)
            unique_neigh = np.unique(rounded_dist)
            neigh_to_plot = unique_neigh[:num_neighbors]

    if not all_data:
        print("No data loaded. Please check your save_dir and mu_values.")
        return

    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    traces_dict = {}
    total_traces = 0
    N = len(unique_b)

    # --- 2. Build All Traces ---
    for m_idx, mu in enumerate(mu_values):
        if mu not in all_data: continue
        
        for k, c_val in enumerate(unique_c):
            traces_dict[(m_idx, k)] = []
            is_visible = (m_idx == 0 and k == 0) 
            
            for n_idx, n in enumerate(neigh_to_plot):
                color = colors[n_idx % len(colors)]
                indices = np.where(np.isclose(rounded_dist, n))[0]
                
                for j, i in enumerate(indices):
                    y_slice = all_data[mu]['y_data'][N*k : N*(k+1), i]
                    
                    # Scatter Points
                    fig.add_trace(go.Scatter(
                        x=unique_b, y=y_slice, mode='markers',
                        marker=dict(symbol='diamond', size=8, color=color),
                        name=f"dist={np.round(n,2)}",
                        legendgroup=f"dist_{np.round(n,2)}", 
                        showlegend=False, visible=is_visible
                    ))
                    traces_dict[(m_idx, k)].append(total_traces)
                    total_traces += 1

                    # Interpolated Line
                    xs = np.linspace(unique_b.min(), unique_b.max(), 400)
                    cs = PchipInterpolator(unique_b, y_slice)
                    fig.add_trace(go.Scatter(
                        x=xs, y=cs(xs), mode='lines',
                        line=dict(color=color, width=2),
                        name=f"dist={np.round(n,2)}",
                        legendgroup=f"dist_{np.round(n,2)}",
                        showlegend=(j == 0), hoverinfo='x+y+name', visible=is_visible
                    ))
                    traces_dict[(m_idx, k)].append(total_traces)
                    total_traces += 1

    # --- 3. Build UI Controls ---
    sliders_all = []
    dropdown_buttons = []

    for m_idx, mu in enumerate(mu_values):
        if mu not in all_data: continue
        
        steps = []
        for k, c_val in enumerate(unique_c):
            step_visibility = [False] * total_traces
            for trace_idx in traces_dict[(m_idx, k)]:
                step_visibility[trace_idx] = True
                
            steps.append(dict(
                method="update",
                args=[
                    {"visible": step_visibility},
                    {"title": f"Neighbors for μ = {mu}, c_inter = {np.round(c_val, 3)}"}
                ],
                label=str(np.round(c_val, 3))
            ))
            
        sliders_all.append(dict(
            active=0,
            currentvalue={"prefix": f"c_inter (μ={mu}): "},
            pad={"t": 50},
            steps=steps
        ))

        btn_visibility = [False] * total_traces
        for trace_idx in traces_dict[(m_idx, 0)]:
            btn_visibility[trace_idx] = True
            
        dropdown_buttons.append(dict(
            label=f"μ = {mu}",
            method="update",
            args=[
                {"visible": btn_visibility},
                {
                    "sliders": [sliders_all[m_idx]]
                }
            ]
        ))

    # --- 4. Layout Styling ---
    # Cleaned up the quantity string to prevent broken LaTeX formatting
    clean_y_title = "J_iso [meV]" if quantity == "J_iso_meV" else "D [meV]"

    fig.update_layout(
        xaxis_title="Zeeman splitting (b) [eV]",
        yaxis_title=clean_y_title,
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=120),  # Added generous top margin for the UI controls
        sliders=[sliders_all[0]], 
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.01, xanchor="left",
            y=1.20, yanchor="top" # Lifted the dropdown way above the title
        )],
        legend=dict(title="Distances", yanchor="top", y=1, xanchor="left", x=1.05),
        width=950, height=700
    )

    # Removed the fixed y-axis so Plotly auto-scales properly per slider/dropdown step
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    fig.add_annotation(
        text="Chemical Potential:",
        x=0.01, y=1.27, xref="paper", yref="paper", showarrow=False, xanchor="left" # Shifted annotation to match new dropdown position
    )

    fig.show()


def plot_interactive_mu_dist(save_dir, mu_values, quantity='J_iso_meV', num_neighbors=5):
    """
    Plots interactive data where the dropdown controls μ (Chemical Potential),
    the slider controls Neighbor Distance, and lines are colored by c_inter.
    """
    all_data = {}
    unique_c = None
    unique_b = None
    dist_vector = None
    
    print("Loading data...")
    # --- 1. Load All Data ---
    for mu in mu_values:
        file_path = os.path.join(save_dir, f"data_mu_{mu}.npz")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping mu={mu}.")
            continue
            
        data = np.load(file_path)
        all_data[mu] = {
            'c_inter': data['c_inter'],
            'b_val': data['b_val'],
            'y_data': data[quantity],
            'distances': data['distances']
        }
        
        # Extract unique axes and neighbor distances from the first valid file
        if unique_c is None:
            unique_c = np.unique(np.round(all_data[mu]['c_inter'], 5))
            unique_b = np.unique(np.round(all_data[mu]['b_val'], 5))
            dist_vector = all_data[mu]['distances'][0]
            rounded_dist = np.round(dist_vector, 4)
            unique_neigh = np.unique(rounded_dist)
            neigh_to_plot = unique_neigh[:num_neighbors]

    if not all_data:
        print("No data loaded. Please check your save_dir and mu_values.")
        return

    fig = go.Figure()
    
    # Expanded color palette (10 colors) to handle multiple c_inter values
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    traces_dict = {}
    total_traces = 0
    N = len(unique_b)

    # --- 2. Build All Traces (Hierarchy swapped!) ---
    for m_idx, mu in enumerate(mu_values):
        if mu not in all_data: continue
        
        # Loop through neighbor distances for the slider
        for n_idx, n in enumerate(neigh_to_plot):
            traces_dict[(m_idx, n_idx)] = []
            is_visible = (m_idx == 0 and n_idx == 0) # Only first combo visible
            
            indices = np.where(np.isclose(rounded_dist, n))[0]
            
            # Loop through c_inter values for the lines/colors
            for k, c_val in enumerate(unique_c):
                color = colors[k % len(colors)]
                c_label = np.round(c_val, 3)
                
                for j, i in enumerate(indices):
                    y_slice = all_data[mu]['y_data'][N*k : N*(k+1), i]
                    
                    # Scatter Points
                    fig.add_trace(go.Scatter(
                        x=unique_b, y=y_slice, mode='markers',
                        marker=dict(symbol='diamond', size=8, color=color),
                        name=f"c={c_label}",
                        legendgroup=f"c_{c_label}", 
                        showlegend=False, visible=is_visible
                    ))
                    traces_dict[(m_idx, n_idx)].append(total_traces)
                    total_traces += 1

                    # Interpolated Line
                    xs = np.linspace(unique_b.min(), unique_b.max(), 400)
                    cs = PchipInterpolator(unique_b, y_slice)
                    fig.add_trace(go.Scatter(
                        x=xs, y=cs(xs), mode='lines',
                        line=dict(color=color, width=2),
                        name=f"c={c_label}",
                        legendgroup=f"c_{c_label}",
                        showlegend=(j == 0), hoverinfo='x+y+name', visible=is_visible
                    ))
                    traces_dict[(m_idx, n_idx)].append(total_traces)
                    total_traces += 1

    # --- 3. Build UI Controls ---
    sliders_all = []
    dropdown_buttons = []

    for m_idx, mu in enumerate(mu_values):
        if mu not in all_data: continue
        
        # Slider steps (now based on neighbor distances)
        steps = []
        for n_idx, n in enumerate(neigh_to_plot):
            step_visibility = [False] * total_traces
            for trace_idx in traces_dict[(m_idx, n_idx)]:
                step_visibility[trace_idx] = True
                
            steps.append(dict(
                method="update",
                args=[
                    {"visible": step_visibility},
                    {"title": f"c_inter sweep for μ = {mu}, dist = {np.round(n, 3)}"}
                ],
                label=str(np.round(n, 3))
            ))
            
        sliders_all.append(dict(
            active=0,
            currentvalue={"prefix": f"Dist (μ={mu}): "},
            pad={"t": 50},
            steps=steps
        ))

        # Dropdown logic
        btn_visibility = [False] * total_traces
        for trace_idx in traces_dict[(m_idx, 0)]:
            btn_visibility[trace_idx] = True
            
        dropdown_buttons.append(dict(
            label=f"μ = {mu}",
            method="update",
            args=[
                {"visible": btn_visibility},
                {
                    "sliders": [sliders_all[m_idx]],
                    "title": f"c_inter sweep for μ = {mu}, dist = {np.round(neigh_to_plot[0], 3)}"
                }
            ]
        ))

    # --- 4. Layout Styling ---
    clean_y_title = "J_iso [meV]" if quantity == "J_iso_meV" else "D [meV]"

    fig.update_layout(
        title=dict(
            text=f"c_inter sweep for μ = {mu_values[0]}, dist = {np.round(neigh_to_plot[0], 3)}",
            y=0.98, x=0.5, xanchor='center', yanchor='top'
        ),
        xaxis_title="Zeeman splitting (b) [eV]",
        yaxis_title=clean_y_title,
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=120),  # Top margin to prevent squishing
        sliders=[sliders_all[0]], 
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.01, xanchor="left",
            y=1.20, yanchor="top"
        )],
        legend=dict(title="c_inter values", yanchor="top", y=1, xanchor="left", x=1.05),
        width=950, height=700
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    fig.add_annotation(
        text="Chemical Potential:",
        x=0.01, y=1.27, xref="paper", yref="paper", showarrow=False, xanchor="left"
    )

    fig.show()