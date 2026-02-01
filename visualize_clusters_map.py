#!/usr/bin/env python3
"""Visualize neighborhood clusters on maps - supports single and combined cities."""

import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import contextily as cx
from matplotlib.lines import Line2D
import sys

def to_web_mercator(lon, lat):
    """Convert WGS84 to Web Mercator."""
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    x = lon * 20037508.34 / 180
    y = np.log(np.tan((90 + lat) * np.pi / 360)) / (np.pi / 180)
    y = y * 20037508.34 / 180
    return x, y

def load_city_data(city_name):
    """Load data and metadata for a city."""
    df = pd.read_csv(f'data/{city_name}/rentals_with_stats.csv')
    with open(f'models/{city_name}/rental_price_model_metadata.json') as f:
        metadata = json.load(f)
    return df, metadata

def create_map(cities, output_name, title):
    """
    Create a map visualization for one or more cities.

    cities: list of city folder names (e.g., ['ann_arbor', 'ypsilanti'])
    output_name: base name for output files
    title: map title
    """
    # Extended color palette (20 colors)
    base_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]

    fig, ax = plt.subplots(figsize=(16, 12))

    all_data = []
    legend_elements = []
    color_idx = 0
    total_listings = 0

    for city in cities:
        df, metadata = load_city_data(city)
        centroids = metadata['cluster_centroids']
        city_display = metadata.get('city_display_name', city.replace('_', ' ').title())

        total_listings += len(df)

        # Plot each cluster
        for cluster_id, centroid_info in centroids.items():
            cluster_df = df[df['Cluster_ID'] == int(cluster_id)]
            if len(cluster_df) == 0:
                continue

            color = base_colors[color_idx % len(base_colors)]
            color_idx += 1

            # Get neighborhood name
            neighborhood = centroid_info.get('name', f'Cluster {cluster_id}')
            count = centroid_info['count']

            # For combined maps, prefix with city name
            if len(cities) > 1:
                label = f"{city_display}: {neighborhood} ({count})"
            else:
                label = f"{neighborhood} ({count})"

            # Convert coordinates
            xs, ys = to_web_mercator(cluster_df['Longitude'].values, cluster_df['Latitude'].values)

            # Plot individual listings
            ax.scatter(
                xs, ys,
                c=color,
                alpha=0.6,
                s=35,
                edgecolors='white',
                linewidths=0.3,
                zorder=3
            )

            # Plot centroid
            cx_c, cy_c = to_web_mercator(centroid_info['longitude'], centroid_info['latitude'])
            ax.scatter(
                cx_c, cy_c,
                c=color,
                s=250,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                zorder=4
            )

            # Store for legend
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=color, markersize=8, label=label))

            # Store centroid info for labels
            all_data.append({
                'name': neighborhood,
                'cx': cx_c,
                'cy': cy_c,
                'color': color,
                'count': count
            })

    # Add basemap
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)

    # Add labels with smart positioning
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]

    # Only label clusters with significant count
    min_count_for_label = 5 if total_listings > 100 else 2

    for data in all_data:
        if data['count'] >= min_count_for_label:
            offset_x = x_range * 0.015
            offset_y = y_range * 0.015

            ax.annotate(
                data['name'],
                (data['cx'], data['cy']),
                xytext=(data['cx'] + offset_x, data['cy'] + offset_y),
                fontsize=8,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85,
                         edgecolor=data['color'], linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color=data['color'], lw=1),
                zorder=5
            )

    # Title
    ax.set_title(f'{title}\n{total_listings:,} listings', fontsize=16, fontweight='bold', pad=15)

    # Legend - split into columns if many items
    ncol = 2 if len(legend_elements) > 10 else 1
    ax.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize=8,
        title='Neighborhood (listings)',
        title_fontsize=9,
        framealpha=0.9,
        ncol=ncol
    )

    ax.set_axis_off()
    plt.tight_layout()

    # Save both PNG and HTML
    plt.savefig(f'{output_name}.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved {output_name}.png")

    plt.close()


def create_interactive_map(cities, output_name, title):
    """Create an interactive folium map for one or more cities."""
    import folium

    base_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
    ]

    # Calculate center from all cities
    all_lats, all_lons = [], []
    for city in cities:
        df, _ = load_city_data(city)
        all_lats.extend(df['Latitude'].tolist())
        all_lons.extend(df['Longitude'].tolist())

    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='cartodbpositron')

    color_idx = 0
    legend_items = []
    total_listings = 0

    for city in cities:
        df, metadata = load_city_data(city)
        centroids = metadata['cluster_centroids']
        city_display = metadata.get('city_display_name', city.replace('_', ' ').title())
        total_listings += len(df)

        for cluster_id, centroid_info in centroids.items():
            cluster_df = df[df['Cluster_ID'] == int(cluster_id)]
            if len(cluster_df) == 0:
                continue

            color = base_colors[color_idx % len(base_colors)]
            color_idx += 1

            neighborhood = centroid_info.get('name', f'Cluster {cluster_id}')
            count = centroid_info['count']

            if len(cities) > 1:
                full_name = f"{city_display}: {neighborhood}"
            else:
                full_name = neighborhood

            legend_items.append((full_name, count, color))

            # Add listings
            for _, row in cluster_df.iterrows():
                # Handle potential NaN values
                price = int(row['Price']) if pd.notna(row['Price']) else 0
                beds = int(row['Beds']) if pd.notna(row['Beds']) else 0
                baths = row['Baths'] if pd.notna(row['Baths']) else 0

                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=f"${price:,}/mo<br>{beds} bed, {baths} bath<br>{full_name}",
                    tooltip=f"${price:,} - {neighborhood}"
                ).add_to(m)

            # Add centroid label
            folium.Marker(
                location=[centroid_info['latitude'], centroid_info['longitude']],
                icon=folium.DivIcon(
                    html=f'''<div style="background-color:{color};color:white;padding:3px 8px;
                            border-radius:10px;font-weight:bold;font-size:10px;white-space:nowrap;
                            border:2px solid white;box-shadow:1px 1px 3px rgba(0,0,0,0.3);">{neighborhood}</div>''',
                    icon_size=(120, 25),
                    icon_anchor=(60, 12)
                )
            ).add_to(m)

    # Add legend
    legend_html = f'''<div style="position:fixed;bottom:50px;left:50px;z-index:1000;
        background:white;padding:12px;border-radius:8px;box-shadow:2px 2px 6px rgba(0,0,0,0.3);
        font-family:Arial;font-size:11px;max-height:400px;overflow-y:auto;">
        <b style="font-size:13px;">{title}</b><br>
        <span style="color:#666;">{total_listings:,} listings</span><hr style="margin:6px 0;">'''

    for name, count, color in sorted(legend_items, key=lambda x: -x[1]):
        legend_html += f'''<div style="margin:3px 0;"><span style="background:{color};
            width:10px;height:10px;display:inline-block;border-radius:50%;margin-right:5px;"></span>
            {name} ({count})</div>'''
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(f'{output_name}.html')
    print(f"Saved {output_name}.html")


if __name__ == '__main__':
    # Detroit
    print("\n=== Creating Detroit map ===")
    create_map(['detroit'], 'detroit_clusters_map', 'Detroit Rental Neighborhoods')
    create_interactive_map(['detroit'], 'detroit_clusters_map', 'Detroit Neighborhoods')

    # Lansing + East Lansing
    print("\n=== Creating Lansing + East Lansing map ===")
    create_map(['lansing', 'east_lansing'], 'lansing_clusters_map', 'Lansing & East Lansing Rental Neighborhoods')
    create_interactive_map(['lansing', 'east_lansing'], 'lansing_clusters_map', 'Lansing & East Lansing')

    # Ypsilanti + Ann Arbor
    print("\n=== Creating Ypsilanti + Ann Arbor map ===")
    create_map(['ypsilanti', 'ann_arbor'], 'ypsi_aa_clusters_map', 'Ypsilanti & Ann Arbor Rental Neighborhoods')
    create_interactive_map(['ypsilanti', 'ann_arbor'], 'ypsi_aa_clusters_map', 'Ypsilanti & Ann Arbor')

    print("\nDone! Created 6 map files.")
