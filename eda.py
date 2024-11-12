# import relevant packages
import pandas as pd
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os


# Create a function that creates a distribution for each column in the data set
def create_distributions(df, output_file="distributions.html", subplot_height = 1000, subplot_width= 1500):
    """
    Create interactive distribution plots for each numerical column in a DataFrame
    and save it as an HTML file.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - output_file (str): The name of the HTML file to save the plots to.

    Returns:
    - None. Saves an HTML file with interactive distributions for each column.
    """
    
    # Get list of numerical columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    num_columns = len(numeric_columns)
    
    # Create subplots layout
    fig = make_subplots(rows=num_columns, cols=1, subplot_titles=numeric_columns)
    
    for i, col in enumerate(numeric_columns, start=1):
        # Create a histogram for each column
        hist = go.Histogram(x=df[col], nbinsx=30, name=col, opacity=0.7)
        
        # Add histogram to subplot
        fig.add_trace(hist, row=i, col=1)
    
    # Update layout for better visualization
    fig.update_layout(
        height=subplot_height * num_columns,  # Adjust the height for each plot
        width = subplot_width,
        title_text="Distribution of Numerical Columns",
        showlegend=False,
        template="plotly_white"
    )
    # Save the interactive plot to the results subfolder
    full_path = os.path.join("./results/", output_file)
    # Save the interactive plot to an HTML file
    pio.write_html(fig, file=full_path, auto_open=True)

# create a function that generates scatter plots for two columns 


def create_scatter_plots(df, column_pairs, output_file="scatter_plots.html"):
    """
    Create a series of scatter plots for specified column pairs in a DataFrame and 
    save them into a single HTML file in the 'results' subfolder.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_pairs (list of tuples): A list of tuples where each tuple contains two column names for x and y axes.
    - output_file (str): The name of the HTML file to save the plots to.

    Returns:
    - None. Saves an HTML file with interactive scatter plots for each column pair.
    """
    
    # Determine the number of plots
    num_plots = len(column_pairs)
    
    # Set up subplots: create one row per scatter plot
    fig = make_subplots(rows=num_plots, cols=1, subplot_titles=[f"{x} vs {y}" for x, y in column_pairs])
    
    # Generate each scatter plot for the specified column pairs
    for i, (x_column, y_column) in enumerate(column_pairs, start=1):
        # Check if the columns exist in the DataFrame
        if x_column not in df.columns or y_column not in df.columns:
            raise ValueError(f"Columns '{x_column}' or '{y_column}' not found in the DataFrame.")
        
        # Create scatter plot for the current column pair
        scatter = go.Scatter(x=df[x_column], y=df[y_column], mode='markers', name=f"{x_column} vs {y_column}")
        
        # Add the scatter plot to the subplot layout
        fig.add_trace(scatter, row=i, col=1)
        
        # Set x and y axis titles for each subplot
        fig.update_xaxes(title_text=x_column, row=i, col=1)
        fig.update_yaxes(title_text=y_column, row=i, col=1)
    
    # Update layout for full page width and set template
    fig.update_layout(
        height=400 * num_plots,  # Set a height for each plot row
        width=1000,  # Full page width
        title_text="Scatter Plots of Specified Column Pairs",
        showlegend=False,
        template="plotly_white",
    )
    
    # Save the interactive plot to the results subfolder
    full_path = os.path.join("./results/", output_file)
    fig.write_html(full_path, auto_open=True)

# define a function that groups by the first column so we can see a more generalized relationship

def group_and_scatter(df, group_column, scatter_column, output_html="scatter_plot.html"):
    """
    Groups the DataFrame by the specified column, then creates a scatter plot
    between the group column and another specified column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - group_column (str): The column name to group by.
    - scatter_column (str): The column name to plot against the group column.
    - output_html (str): The name of the HTML file to save the scatter plot.

    Returns:
    - pd.DataFrame: The grouped and aggregated DataFrame.
    """
    
    # Check if the columns exist in the DataFrame
    if group_column not in df.columns or scatter_column not in df.columns:
        raise ValueError(f"Columns '{group_column}' or '{scatter_column}' not found in the DataFrame.")

    # Group by the group_column and calculate the mean for the scatter_column
    grouped_data = df.groupby(group_column)[scatter_column].mean().reset_index()

    # Create the scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=grouped_data[group_column],
        y=grouped_data[scatter_column],
        mode='markers+lines',  # Markers with connecting lines
        name=f"{group_column} vs {scatter_column}",
        text=grouped_data[scatter_column],  # Hover text
        marker=dict(size=8, opacity=0.7)
    ))

    # Create the layout for the scatter plot
    fig.update_layout(
        title=f"Scatter Plot of {scatter_column} vs {group_column}",
        xaxis_title=group_column,
        yaxis_title=f"Avg {scatter_column}",
        template="plotly_white",
        showlegend=True,
        autosize=True,
        margin=dict(t=40, b=40, l=40, r=40)  # Adjust margins to optimize space
    )

    # Save the interactive plot to the results subfolder
    scatter_plots_path = os.path.join("./results/", output_html)
    fig.write_html(scatter_plots_path, auto_open=True)  # Open the plot in the browser after saving

    # Return the grouped data with the average
    return grouped_data

# For binary variables scatter plots are of little use, so we use box plots instead
# define a function that creates box plots of variables of interest and returns an html
def generate_box_plots(df, variable_pairs, output_html="box_plots.html"):
    """
    Generates box plots for pairs of variables: one continuous and one binary.
    The resulting plot is saved as an interactive HTML file.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - variable_pairs (list of tuples): A list of tuples where each tuple contains
      two column names. The first is a binary variable, and the second is a continuous variable.
    - output_html (str): The name of the HTML file to save the box plot.
    
    Returns:
    - None (Saves the plot as an HTML file).
    """
    
    # Create a plotly figure
    fig = go.Figure()

    # Loop through each pair of variables
    for binary_column, continuous_column in variable_pairs:
        # Check if the columns exist in the DataFrame
        if binary_column not in df.columns or continuous_column not in df.columns:
            raise ValueError(f"Columns '{binary_column}' or '{continuous_column}' not found in the DataFrame.")

        # Create a box plot for each binary value (0 and 1)
        for value in df[binary_column].unique():
            group_data = df[df[binary_column] == value][continuous_column]
            fig.add_trace(go.Box(
                y=group_data,
                name=f"{binary_column} = {value}",
                boxmean="sd",  # Show the mean and standard deviation
                jitter=0.3,    # Add jitter for better visualization of individual points
                marker=dict(size=6),
                line=dict(width=1)
            ))

    # Update layout of the box plot
    fig.update_layout(
        title=f"Box Plots for {', '.join([f'{pair[1]} vs {pair[0]}' for pair in variable_pairs])}",
        xaxis_title="Binary Variable",
        yaxis_title="Continuous Variable",
        template="plotly_white",
        showlegend=True,
        autosize=True,
        height = 1500,
        margin=dict(t=40, b=40, l=40, r=40)  # Adjust margins for better space usage
    )

    # Save the interactive plot to the results subfolder as an HTML file
    box_plots_path = os.path.join("./results/", output_html)
    fig.write_html(box_plots_path, auto_open=True)  # Open in the browser after saving

    print(f"Box plots saved to {box_plots_path}")


# Introduce the data frame
df = pd.read_csv("./results/post_processed_data.csv")
#strip spaces
df.columns = df.columns.str.strip()

# apply the distribution funciton to a relevant subset of the data frame
columns_to_select = list(range(3, 8)) + [9, 18, 19, 27] + list(range(29, 41))
df_subset =  df.iloc[:, columns_to_select]
html = create_distributions(df_subset)

# apply the scatter plot function to a set of tuples
column_pairs = [('onset','duration'),('age','duration'),
                ('avg_x','duration'),('avg_y','duration'),('Order','duration'),('offset','duration')]

create_scatter_plots(df, column_pairs, output_file="scatter_plots.html")
# Create the grouped scatter plots
group_and_scatter(df,'age','duration',output_html="age_vs_duration_scatter.html")

# create the box plots html
variable_pairs = [('female','duration'),('Age_18-26', 'duration'),('Age_27-35', 'duration'),
                  ('Age_36-45', 'duration'),('Age_46-59', 'duration'),('airship','duration'),
                  ('diver_with_abs','duration'),('elefant_under_water','duration'),('female_diver','duration'),
                  ('man_wakeboarding','duration'),('mermaid','duration'),('octupus','duration'),
                  ('people_jumping_from_boat','duration'),('seal','duration'),('son_and_father_fishing','duration'),
                  ('space','duration'),('sunset','duration'),('turtle','duration'),('woman_on_surboard','duration')]

generate_box_plots(df, variable_pairs, output_html="box_plots_example.html")