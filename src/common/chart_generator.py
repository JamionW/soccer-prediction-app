import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import os
from datetime import datetime

"""
Generates interactive charts for simulation results that can be viewed locally
and easily ported to React frontend.
"""

class MLSNPChartGenerator:
    """
    Generates interactive Plotly charts for MLS Next Pro simulation results.
    
    Charts generated:
    1. Playoff Probability Bar Chart
    2. Final Rank Distribution Box Plot  
    3. Points Analysis Scatter Plot
    4. Shootout Win Impact Analysis
    """
    
    def __init__(self, output_dir: str = "output/charts"):
        """
        Initialize chart generator.
        
        Args:
            output_dir: Directory to save chart HTML files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # MLS Next Pro team colors (you can customize these)
        self.team_colors = {
            'default': '#1f77b4',
            'clinched': '#2ca02c',  # Green for clinched
            'eliminated': '#d62728',  # Red for eliminated
            'bubble': '#ff7f0e'  # Orange for bubble teams
        }
    
    def generate_all_charts(self, summary_df: pd.DataFrame, simulation_results: Dict, 
                          qualification_data: Dict, conference: str, 
                          n_simulations: int) -> Dict[str, str]:
        """
        Generate all charts for a simulation run.
        
        Args:
            summary_df: Summary DataFrame from simulation
            simulation_results: Raw simulation results (rank distributions)
            qualification_data: Qualification analysis data
            conference: Conference name
            n_simulations: Number of simulations run
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_files = {}
        
        # 1. Playoff Probability Chart
        playoff_chart = self.create_playoff_probability_chart(
            summary_df, conference, n_simulations
        )
        playoff_file = f"{self.output_dir}/playoff_probabilities_{conference}_{timestamp}.html"
        playoff_chart.write_html(playoff_file)
        chart_files['playoff_probabilities'] = playoff_file
        
        # 2. Rank Distribution Chart
        rank_chart = self.create_rank_distribution_chart(
            summary_df, simulation_results, conference
        )
        rank_file = f"{self.output_dir}/rank_distributions_{conference}_{timestamp}.html"
        rank_chart.write_html(rank_file)
        chart_files['rank_distributions'] = rank_file
        
        # 3. Points Analysis Chart
        points_chart = self.create_points_analysis_chart(
            summary_df, conference
        )
        points_file = f"{self.output_dir}/points_analysis_{conference}_{timestamp}.html"
        points_chart.write_html(points_file)
        chart_files['points_analysis'] = points_file
        
        # 4. Combined Dashboard
        dashboard = self.create_dashboard(
            summary_df, simulation_results, conference, n_simulations
        )
        dashboard_file = f"{self.output_dir}/dashboard_{conference}_{timestamp}.html"
        dashboard.write_html(dashboard_file)
        chart_files['dashboard'] = dashboard_file
        
        return chart_files
    
    def create_playoff_probability_chart(self, summary_df: pd.DataFrame, 
                                       conference: str, n_simulations: int) -> go.Figure:
        """
        Create horizontal bar chart of playoff probabilities.
        """
        # Sort by playoff probability
        df_sorted = summary_df.sort_values('Playoff Qualification %', ascending=True)
        
        # Assign colors based on playoff status
        colors = []
        for _, row in df_sorted.iterrows():
            prob = row['Playoff Qualification %']
            if prob >= 99.9:
                colors.append(self.team_colors['clinched'])
            elif prob <= 0.1:
                colors.append(self.team_colors['eliminated'])
            elif 25 <= prob <= 75:
                colors.append(self.team_colors['bubble'])
            else:
                colors.append(self.team_colors['default'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=df_sorted['Team'],
            x=df_sorted['Playoff Qualification %'],
            orientation='h',
            marker_color=colors,
            text=[f"{prob:.1f}%" for prob in df_sorted['Playoff Qualification %']],
            textposition='inside',
            textfont_color='white',
            hovertemplate='<b>%{y}</b><br>' +
                         'Playoff Probability: %{x:.1f}%<br>' +
                         'Current Points: %{customdata[0]}<br>' +
                         'Current Rank: %{customdata[1]}<extra></extra>',
            customdata=list(zip(df_sorted['Current Points'], df_sorted['Current Rank']))
        ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference Playoff Probabilities<br>' +
                       f'<sub>Based on {n_simulations:,} simulations</sub>',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Playoff Probability (%)',
            yaxis_title='Teams',
            height=600,
            margin=dict(l=150),
            template='plotly_white'
        )
        
        # Add vertical lines for reference
        fig.add_vline(x=50, line_dash="dash", line_color="gray", 
                     annotation_text="50% chance")
        fig.add_vline(x=75, line_dash="dot", line_color="green", 
                     annotation_text="75% chance")
        
        return fig
    
    def create_rank_distribution_chart(self, summary_df: pd.DataFrame, 
                                     simulation_results: Dict, 
                                     conference: str) -> go.Figure:
        """
        Create box plot showing rank distribution for each team.
        """
        fig = go.Figure()
        
        # Sort teams by average rank
        df_sorted = summary_df.sort_values('Average Final Rank')
        
        for _, row in df_sorted.iterrows():
            team_id = row['_team_id']
            team_name = row['Team']
            
            # Get rank distribution from simulation results
            rank_data = simulation_results.get(team_id, [])
            
            if rank_data:
                fig.add_trace(go.Box(
                    y=rank_data,
                    name=team_name,
                    boxmean='sd',  # Show mean and standard deviation
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Rank: %{y}<br>' +
                                 'Median: %{customdata[0]:.1f}<br>' +
                                 'Mean: %{customdata[1]:.1f}<extra></extra>',
                    customdata=[[row['Median Final Rank'], row['Average Final Rank']]] * len(rank_data)
                ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference Final Rank Distributions',
                'x': 0.5,
                'font': {'size': 20}
            },
            yaxis_title='Final Regular Season Rank',
            xaxis_title='Teams',
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        # Reverse y-axis so rank 1 is at the top
        fig.update_yaxes(autorange="reversed")
        
        # Add playoff line
        fig.add_hline(y=8.5, line_dash="dash", line_color="green", 
                     annotation_text="Playoff Line (Top 8)")
        
        return fig
    
    def create_points_analysis_chart(self, summary_df: pd.DataFrame, 
                                   conference: str) -> go.Figure:
        """
        Create scatter plot comparing current vs projected points.
        """
        fig = go.Figure()
        
        # Create bubble chart
        fig.add_trace(go.Scatter(
            x=summary_df['Current Points'],
            y=summary_df['Average Points'],
            mode='markers',
            marker=dict(
                size=summary_df['Playoff Qualification %'] / 3,  # Size based on playoff prob
                color=summary_df['Playoff Qualification %'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Playoff<br>Probability (%)")
            ),
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Current Points: %{x}<br>' +
                         'Projected Points: %{y:.1f}<br>' +
                         'Playoff Probability: %{marker.color:.1f}%<br>' +
                         'Current Rank: %{customdata}<extra></extra>',
            customdata=list(zip(summary_df['Team'], summary_df['Current Rank']))
        ))
        
        # Add diagonal line (y = x)
        min_points = min(summary_df['Current Points'].min(), summary_df['Average Points'].min())
        max_points = max(summary_df['Current Points'].max(), summary_df['Average Points'].max())
        
        fig.add_trace(go.Scatter(
            x=[min_points, max_points],
            y=[min_points, max_points],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Current = Projected',
            showlegend=True
        ))
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference: Current vs Projected Points' +
                        f'<sub>Bubble size = Playoff Probability | Hover for team names</sub>',
                'x': 0.5,
                'font': {'size': 20}
            },
            xaxis_title='Current Points',
            yaxis_title='Projected Final Points',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_dashboard(self, summary_df: pd.DataFrame, simulation_results: Dict,
                        conference: str, n_simulations: int) -> go.Figure:
        """
        Create a combined dashboard with multiple subplots.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Playoff Probabilities', 
                'Points: Current vs Projected',
                'Top 8 Teams - Rank Ranges',
                'Bottom 8 Teams - Rank Ranges'
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "box"}]]
        )
        
        # Sort for consistent ordering
        df_sorted_by_playoff = summary_df.sort_values('Playoff Qualification %', ascending=False)
        df_sorted_by_rank = summary_df.sort_values('Average Final Rank', ascending=True)
        
        # 1. Playoff Probabilities (top-left)
        fig.add_trace(
            go.Bar(
                x=df_sorted_by_playoff['Team'][:8],
                y=df_sorted_by_playoff['Playoff Qualification %'][:8],
                name='Playoff %',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. Points Analysis (top-right)
        fig.add_trace(
            go.Scatter(
                x=summary_df['Current Points'],
                y=summary_df['Average Points'],
                mode='markers',
                marker=dict(size=8, color='orange'),
                name='Teams',
                hovertemplate='<b>%{customdata}</b><br>' +
                     'Current Points: %{x}<br>' +
                     'Projected Points: %{y:.1f}<extra></extra>',
                customdata=summary_df['Team']
            ),
            row=1, col=2
        )
        
        # 3. Top 8 Teams Rank Ranges (bottom-left)
        top_8_teams = df_sorted_by_rank.head(8)
        for _, row in top_8_teams.iterrows():
            team_id = row['_team_id']
            rank_data = simulation_results.get(team_id, [])
            if rank_data:
                fig.add_trace(
                    go.Box(
                        y=rank_data,
                        name=row['Team'],
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Bottom 8 Teams Rank Ranges (bottom-right)
        bottom_8_teams = df_sorted_by_rank.tail(8)
        for _, row in bottom_8_teams.iterrows():
            team_id = row['_team_id']
            rank_data = simulation_results.get(team_id, [])
            if rank_data:
                fig.add_trace(
                    go.Box(
                        y=rank_data,
                        name=row['Team'],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title={
                'text': f'{conference.title()} Conference Simulation Dashboard<br>' +
                       f'<sub>{n_simulations:,} simulations</sub>',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=800,
            template='plotly_white'
        )
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        fig.update_yaxes(autorange="reversed", row=2, col=2)
        
        return fig
    
    def show_charts_summary(self, chart_files: Dict[str, str], conference: str):
        """
        Print a summary of generated charts.
        """
        print(f"\n{'='*60}")
        print(f"Generated Charts for {conference.title()} Conference")
        print(f"{'='*60}")
        
        for chart_name, file_path in chart_files.items():
            print(f" {chart_name.replace('_', ' ').title()}")
            print(f"      → {file_path}")
        
        print(f"\n Open any HTML file in your browser to view interactive charts!")
