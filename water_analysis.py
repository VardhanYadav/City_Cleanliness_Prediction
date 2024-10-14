import streamlit as st
import pandas as pd
import os
import plotly.express as px


# Define acceptable ranges for water quality parameters
ACCEPTABLE_RANGES = {
    'ph': (6.5, 8.5),  # Ideal pH range for most aquatic life
    'do_mg_l': (5.0, float('inf')),  # Minimum DO level is 5 mg/L
    'temp': (0.0, 30.0)  # Acceptable temperature range, may vary
}

# Define improvement tips for out-of-range values
IMPROVEMENT_TIPS = {
    'ph': "Ensure proper waste disposal to prevent industrial runoff or chemicals affecting water pH.",
    'do_mg_l': "Increase aeration in water bodies by improving water flow or using aeration devices.",
    'temp': "Plant trees around water bodies to provide shade and regulate water temperature."
}




def preprocess_data(data):
    # Clean up column names and convert necessary columns to numeric
    data.columns = [col.strip().lower().replace(' ', '_').replace('.', '').replace('(', '').replace(')', '') for col in data.columns]
    data['temp'] = pd.to_numeric(data['temp'], errors='coerce')
    data['do_mg_l'] = pd.to_numeric(data['do_mg_l'], errors='coerce')
    data['ph'] = pd.to_numeric(data['ph'], errors='coerce')
    data = data.dropna(subset=['temp', 'do_mg_l', 'ph'])
    return data


def load_default_data(file_path):
    # Load data from a CSV file
    return pd.read_csv(file_path, encoding='ISO-8859-1')


def show_analysis(data, station_code1, station_code2):
    st.subheader(f"Comparison Between Station {station_code1} and {station_code2}")

    station1_data = data[data['station_code'] == station_code1]
    station2_data = data[data['station_code'] == station_code2]

    if station1_data.empty or station2_data.empty:
        st.error("Data for one or both stations is missing.")
        return

    # Extract parameter values
    station1_values = {
        'temp': station1_data['temp'].values[0],
        'do_mg_l': station1_data['do_mg_l'].values[0],
        'ph': station1_data['ph'].values[0]
    }
    station2_values = {
        'temp': station2_data['temp'].values[0],
        'do_mg_l': station2_data['do_mg_l'].values[0],
        'ph': station2_data['ph'].values[0]
    }

    comparison_data = pd.DataFrame({
        'Parameter': ['Temperature', 'D.O. (mg/l)', 'PH'],
        f'Station {station_code1}': [station1_values['temp'], station1_values['do_mg_l'], station1_values['ph']],
        f'Station {station_code2}': [station2_values['temp'], station2_values['do_mg_l'], station2_values['ph']]
    })

    st.table(comparison_data)

    # Flag to check if any imbalances are found
    imbalance_found = False

    # Check for imbalances and provide tips
    for parameter, (low, high) in ACCEPTABLE_RANGES.items():
        station1_value = station1_values[parameter]
        station2_value = station2_values[parameter]

        # Check for station 1
        if not (low <= station1_value <= high):
            st.warning(f"Station {station_code1} has an imbalanced {parameter.upper()}: {station1_value}.")
            st.info(f"Tip to improve {parameter.upper()}: {IMPROVEMENT_TIPS[parameter]}")
            imbalance_found = True

        # Check for station 2
        if not (low <= station2_value <= high):
            st.warning(f"Station {station_code2} has an imbalanced {parameter.upper()}: {station2_value}.")
            st.info(f"Tip to improve {parameter.upper()}: {IMPROVEMENT_TIPS[parameter]}")
            imbalance_found = True

    # If no imbalances are found, display a message
    if not imbalance_found:
        st.success("All parameters for both stations are within acceptable ranges.")

    # Line chart comparison
    fig = px.line(comparison_data, x='Parameter', y=[f'Station {station_code1}', f'Station {station_code2}'],
                  labels={'value': 'Values', 'variable': 'Station'},
                  title="Line Chart Comparison")
    st.plotly_chart(fig)

    # Bar chart with error bars
    bar_fig = px.bar(comparison_data, x='Parameter', y=[f'Station {station_code1}', f'Station {station_code2}'],
                     title="Bar Chart Comparison", barmode='group',
                     labels={'x': 'Parameters', 'y': 'Values'})
    st.plotly_chart(bar_fig)


def show_map(data, station_code1, station_code2):
    station1_data = data[data['station_code'] == station_code1]
    station2_data = data[data['station_code'] == station_code2]

    if 'lat' not in data.columns or 'lon' not in data.columns:
        st.error("Latitude and Longitude data are not available.")
        return

    station1_lat, station1_lon = station1_data['lat'].values[0], station1_data['lon'].values[0]
    station2_lat, station2_lon = station2_data['lat'].values[0], station2_data['lon'].values[0]

    map_data = pd.DataFrame({
        'Station': [station_code1, station_code2],
        'Latitude': [station1_lat, station2_lat],
        'Longitude': [station1_lon, station2_lon],
        'Temperature': [station1_data['temp'].values[0], station2_data['temp'].values[0]],
        'DO': [station1_data['do_mg_l'].values[0], station2_data['do_mg_l'].values[0]],
        'pH': [station1_data['ph'].values[0], station2_data['ph'].values[0]]
    })

    fig = px.scatter_mapbox(
        map_data,
        lat='Latitude',
        lon='Longitude',
        hover_name='Station',
        hover_data=['Temperature', 'DO', 'pH'],
        size=[15, 15],  # Size of the markers
        title="Water Quality Monitoring Stations Map"
    )

    # Update traces to include marker outlines
    fig.update_traces(
        marker=dict(
            size=14,  # Size of the marker
            color='blue',  # Color of the marker
        )
    )

    # Calculate the map center and zoom based on the locations
    center_lat = (station1_lat + station2_lat) / 2
    center_lon = (station1_lon + station2_lon) / 2

    # Calculate the max distance between the points to adjust zoom
    lat_diff = abs(station1_lat - station2_lat)
    lon_diff = abs(station1_lon - station2_lon)
    max_diff = max(lat_diff, lon_diff)

    # Adjust zoom level based on the max difference
    map_zoom = 6 if max_diff < 0.5 else 5 if max_diff < 1 else 4

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=map_zoom,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        width=800,
        height=600
    )

    st.plotly_chart(fig)


def add_sidebar_info_water_quality():
    st.sidebar.header("Additional Information on Water Quality Analysis", divider="blue")
    st.sidebar.write("""
    **Water Quality Analysis**:

    Water quality analysis assesses the suitability of water for various uses, including drinking, agriculture, and ecological support. It involves monitoring key physical and chemical parameters that influence the health of water bodies and the organisms dependent on them.

    ### Key Parameters:
    1. **Temperature**: Water temperature affects the solubility of gases like oxygen and the metabolic rates of aquatic organisms. Cooler water generally holds more oxygen, while higher temperatures can lead to lower dissolved oxygen levels and stress aquatic life.

    2. **Dissolved Oxygen (DO)**: This measures the amount of oxygen available in water. Adequate DO levels are crucial for the survival of fish and other aquatic organisms. Low DO can indicate pollution and poor water quality, potentially leading to dead zones where life cannot survive.

    3. **pH**: The pH scale measures how acidic or alkaline the water is. Most aquatic organisms prefer water with a pH between 6.5 and 8.5. Water that is too acidic or too basic can harm organisms and indicate the presence of industrial runoff or other pollutants.

    ### Analysis Process:
    1. **Sample Collection**: Water samples are taken from various sources such as rivers, lakes, and groundwater wells for analysis.
    2. **Lab Testing**: The key water parameters (Temperature, DO, pH) are measured using specialized equipment.
    3. **Data Interpretation**: Results are compared to standard water quality guidelines to assess the safety and environmental health of the water.
    4. **Reporting**: Any issues or potential risks are reported, such as high temperatures, low DO levels, or imbalanced pH.

    ### Benefits:
    - **Aquatic Life Protection**: Proper water quality helps maintain a healthy ecosystem for fish and other aquatic species.
    - **Pollution Detection**: Monitoring these key parameters allows for early detection of water pollution, leading to quicker corrective measures.
    - **Human Health**: Ensuring the water quality meets safe standards prevents harmful effects from pollutants in water consumed by people or used in agriculture.
    - **Ecosystem Balance**: Regular analysis supports a balanced ecosystem by identifying fluctuations in water quality that could disrupt the natural environment.

    ### Typical Standards:
    - **DO Levels**: Ideally above 5 mg/L for most aquatic life.
    - **pH Levels**: Should remain between 6.5 and 8.5 for safe and healthy water bodies.
    - **Temperature**: Varies based on the ecosystem, but typically, cooler temperatures support higher dissolved oxygen levels.
    """)


def main():
    st.header("Water Quality Analysis", divider="blue")

    # Sidebar for navigation
    st.sidebar.header("Navigation", divider="blue")
    page = st.sidebar.radio("Select a page:", ["Analysis", "Map"])

    add_sidebar_info_water_quality()

    # Specify the path to the default CSV file
    default_file_path = r"E:/PyCharm/Projects/pythonProject/water_dataX.csv"

    if os.path.exists(default_file_path):
        data = load_default_data(default_file_path)
        data = preprocess_data(data)

        data.dropna(inplace=True)

        # Create a unique mapping of station codes to locations
        unique_stations = data.drop_duplicates(subset=['station_code', 'locations'])

        # Prepare station options and mapping
        station_options = [f"{row['locations']} ({row['station_code']})" for _, row in unique_stations.iterrows()]
        station_mapping = {f"{row['locations']} ({row['station_code']})": row['station_code'] for _, row in
                           unique_stations.iterrows()}

        selected_area1_code = selected_area2_code = None

        if page == "Analysis":
            area1 = st.selectbox("Select Area 1", station_options)
            area2 = st.selectbox("Select Area 2", station_options)
            selected_area1_code = station_mapping[area1]
            selected_area2_code = station_mapping[area2]

            if st.button("Show Analysis"):
                show_analysis(data, selected_area1_code, selected_area2_code)

        elif page == "Map":
            area1 = st.selectbox("Select Area 1", station_options, key='map1')
            area2 = st.selectbox("Select Area 2", station_options, key='map2')
            selected_area1_code = station_mapping[area1]
            selected_area2_code = station_mapping[area2]
            if st.button("Show Map"):
                show_map(data, selected_area1_code, selected_area2_code)

    else:
        st.error("Default data file not found. Please upload a CSV file.")


if __name__ == "__main__":
    main()
