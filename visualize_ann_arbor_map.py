#!/usr/bin/env python3
"""Visualize Ann Arbor neighborhood clusters on an interactive map."""

import pandas as pd
import folium
from folium.plugins import MarkerCluster
import json

# Load data
df = pd.read_csv('data/ann_arbor/rentals_with_stats.csv')
with open('models/ann_arbor/rental_price_model_metadata.json') as f:
    metadata = json.load(f)

centroids = metadata['cluster_centroids']
city_center = metadata['city_center']

# Color palette for clusters
cluster_colors = {
    '0': '#1f77b4',  # blue
    '1': '#ff7f0e',  # orange
    '2': '#2ca02c',  # green
    '3': '#d62728',  # red
    '4': '#9467bd',  # purple
    '5': '#8c564b',  # brown
    '6': '#e377c2',  # pink
    '7': '#7f7f7f',  # gray
    '8': '#bcbd22',  # olive
    '9': '#17becf',  # cyan
}

# Create map centered on Ann Arbor
m = folium.Map(
    location=[city_center['lat'], city_center['lon']],
    zoom_start=12,
    tiles='cartodbpositron'
)

# Add each listing as a circle marker
for _, row in df.iterrows():
    cluster_id = str(int(row['Cluster_ID']))
    color = cluster_colors.get(cluster_id, '#333333')
    neighborhood = centroids[cluster_id]['name']

    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=f"${int(row['Price']):,}/mo<br>{int(row['Beds'])} bed, {row['Baths']} bath<br>{neighborhood}",
        tooltip=f"${int(row['Price']):,} - {neighborhood}"
    ).add_to(m)

# Add cluster centroids as larger markers with labels
for cluster_id, info in centroids.items():
    color = cluster_colors.get(cluster_id, '#333333')

    # Add centroid marker
    folium.Marker(
        location=[info['latitude'], info['longitude']],
        popup=f"<b>{info['name']}</b><br>{info['count']} listings",
        tooltip=f"{info['name']} ({info['count']} listings)",
        icon=folium.DivIcon(
            html=f'''
                <div style="
                    background-color: {color};
                    color: white;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-weight: bold;
                    font-size: 11px;
                    white-space: nowrap;
                    border: 2px solid white;
                    box-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                ">{info['name']}</div>
            ''',
            icon_size=(150, 30),
            icon_anchor=(75, 15)
        )
    ).add_to(m)

# Add legend
legend_html = '''
<div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    z-index: 1000;
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    font-family: Arial, sans-serif;
    font-size: 12px;
">
    <b style="font-size: 14px;">Ann Arbor Neighborhoods</b><br>
    <span style="color: #666;">938 listings, 10 clusters</span>
    <hr style="margin: 8px 0;">
'''
for cluster_id, info in sorted(centroids.items(), key=lambda x: -x[1]['count']):
    color = cluster_colors.get(cluster_id, '#333333')
    legend_html += f'''
    <div style="margin: 4px 0;">
        <span style="
            background-color: {color};
            width: 12px;
            height: 12px;
            display: inline-block;
            border-radius: 50%;
            margin-right: 6px;
        "></span>
        {info['name']} ({info['count']})
    </div>
    '''
legend_html += '</div>'
m.get_root().html.add_child(folium.Element(legend_html))

# Add title
title_html = '''
<div style="
    position: fixed;
    top: 10px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    background-color: white;
    padding: 10px 20px;
    border-radius: 8px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    font-family: Arial, sans-serif;
">
    <b style="font-size: 16px;">Ann Arbor Rental Neighborhoods</b>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# Save map
m.save('ann_arbor_clusters_map.html')
print("Saved interactive map to ann_arbor_clusters_map.html")
print("Open in browser to view")
