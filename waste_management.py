import streamlit as st
import pandas as pd
import plotly.graph_objs as go


THRESHOLDS = {
    'Community Compost Pits': {'assets_per_household': 1/100, 'tip': 'Increase the number of compost pits to at least 1 per 100 households.'},
    'Community Bio-Gas Plants': {'assets_per_household': 1/200, 'tip': 'Ensure at least 1 bio-gas plant for every 200 households to manage organic waste.'},
    'Vehicles for Collection & Transportation of Waste': {'assets_per_household': 1/500, 'tip': 'Increase the number of waste collection vehicles to at least 1 per 500 households.'},
    'Segregation Bins at Community Places': {'assets_per_household': 1/50, 'tip': 'Place more segregation bins, aiming for at least 1 bin per 50 households.'},
    'Waste Collection and Segregation Sheds': {'assets_per_household': 1/1000, 'tip': 'Increase the number of segregation sheds to at least 1 per 1000 households.'}
}


def add_sidebar_info_waste_management():
    st.sidebar.header("Additional Information on Waste Management", divider="blue")
    st.sidebar.write("""
    **Waste Management Analysis**:

    Effective waste management is crucial for maintaining public health, reducing environmental pollution, and promoting sustainability. In this analysis, key waste management metrics focus on community assets that support the collection, segregation, and processing of waste.

    ### Key Metrics:

    1. **Community Compost Pits**: 
       These are designated areas where organic waste is collected and composted into usable fertilizer. Composting reduces the volume of waste sent to landfills and provides a sustainable method for recycling nutrients back into the soil.
       - **Importance**: Helps in managing biodegradable waste, reducing the strain on landfill sites, and producing nutrient-rich compost.
       - **Metric**: The number of compost pits and the households covered by these pits are key indicators of the community's composting efficiency.

    2. **Community Bio-Gas Plants**:
       Bio-gas plants use organic waste (e.g., food scraps, agricultural waste) to produce methane gas, which can be used as an energy source. These plants contribute to renewable energy production while managing waste.
       - **Importance**: Reduces organic waste and provides an alternative, clean energy source for the community.
       - **Metric**: The number of bio-gas plants and the households they service reflect the community's capability to generate renewable energy from waste.

    3. **Vehicles for Collection & Transportation of Waste**:
       Waste collection vehicles are vital for transporting waste from residential areas to processing facilities, ensuring that waste is regularly removed and does not accumulate in communities.
       - **Importance**: Ensures timely and efficient transportation of waste to prevent public health hazards and environmental pollution.
       - **Metric**: The number of vehicles and the households they cover represent the community's capacity for efficient waste collection.

    4. **Segregation Bins at Community Places**:
       These are bins placed in public areas that encourage people to separate waste at the source (e.g., organic, recyclable, non-recyclable). Proper segregation helps reduce contamination in recycling streams and ensures efficient waste processing.
       - **Importance**: Promotes responsible waste segregation at the source, improving the efficiency of downstream waste processing.
       - **Metric**: The number of segregation bins and the households covered by these bins are key indicators of waste management success in public areas.

    5. **Waste Collection and Segregation Sheds**:
       These sheds are locations where waste is sorted and processed before it is sent to recycling facilities or landfills. Segregation sheds improve the overall efficiency of waste management by ensuring that different types of waste are processed correctly.
       - **Importance**: Ensures that recyclables, organic waste, and non-recyclable waste are properly segregated, reducing contamination and improving recycling rates.
       - **Metric**: The number of waste collection and segregation sheds and the households they cover indicate the community's waste processing capacity.

    ### Waste Management Impact:
    - **Public Health**: Proper waste collection and segregation reduce the spread of disease and promote cleaner living conditions.
    - **Environmental Protection**: Effective waste management reduces air, water, and soil pollution, conserving natural resources and minimizing environmental degradation.
    - **Resource Efficiency**: Initiatives such as composting, bio-gas generation, and recycling help reduce the need for new raw materials and lower the carbon footprint of waste disposal processes.

    ### Typical Standards:
    - **Household Coverage**: High coverage by waste management assets (compost pits, bio-gas plants, vehicles, etc.) is essential for comprehensive waste management.
    - **Asset Availability**: Adequate numbers of vehicles, bins, and sheds ensure timely waste collection and segregation, improving overall waste management efficiency.
    """)


def check_and_suggest_improvements(metric, city_name, asset_count, household_count):
    """Check if the asset per household ratio is below the threshold and suggest improvements if needed."""
    threshold = THRESHOLDS[metric]['assets_per_household']
    tip = THRESHOLDS[metric]['tip']

    # Calculate assets per household
    assets_per_household = asset_count / household_count if household_count else 0

    # Check if assets are below threshold
    if assets_per_household < threshold:
        st.warning(f"{metric} for {city_name} is below acceptable levels.")
        st.info(f"Tip: {tip}")



def waste_management():
    # Load the dataset
    data = pd.read_csv("E:/PyCharm/Projects/pythonProject/swaachbharat.csv")

    # Streamlit application
    st.header("Waste Management Data Analysis", divider="blue")
    add_sidebar_info_waste_management()

    # City selection - converting city names to uppercase
    cities = data['District Name'].str.upper().unique()
    city1 = st.selectbox("Select the first city", cities)
    city2 = st.selectbox("Select the second city", cities)

    if st.button("Show Analysis"):
        # Filter the data for selected cities
        city1_data = data[data['District Name'].str.upper() == city1]
        city2_data = data[data['District Name'].str.upper() == city2]

        # Create a DataFrame to compare assets and households covered for the selected cities
        comparison_data = pd.DataFrame({
            'Metric': [
                'Community Compost Pits',
                'Community Bio-Gas Plants',
                'Vehicles for Collection & Transportation of Waste',
                'Segregation Bins at Community Places',
                'Waste Collection and Segregation Sheds'
            ],
            f'{city1} - No. of Assets': [
                city1_data['No. of assets of community compost pits'].values[0],
                city1_data['No. of assets of Community Bio-Gas Plants'].values[0],
                city1_data['No. of assets for Vehicles for Collection & Transportation of Waste'].values[0],
                city1_data['No. of assets of Segregation Bins at Community Places in the village'].values[0],
                city1_data['No. of assets of Waste collection and segregation sheds'].values[0]
            ],
            f'{city1} - Households Covered': [
                city1_data['Household covered by community compost pits'].values[0],
                city1_data['Household covered by Community Bio-Gas Plants'].values[0],
                city1_data['Household covered for Vehicles for Collection & Transportation of Waste'].values[0],
                city1_data['Household covered for Segregation Bins at Community Places in the village'].values[0],
                city1_data['Household covered for Waste collection and segregation sheds'].values[0]
            ],
            f'{city2} - No. of Assets': [
                city2_data['No. of assets of community compost pits'].values[0],
                city2_data['No. of assets of Community Bio-Gas Plants'].values[0],
                city2_data['No. of assets for Vehicles for Collection & Transportation of Waste'].values[0],
                city2_data['No. of assets of Segregation Bins at Community Places in the village'].values[0],
                city2_data['No. of assets of Waste collection and segregation sheds'].values[0]
            ],
            f'{city2} - Households Covered': [
                city2_data['Household covered by community compost pits'].values[0],
                city2_data['Household covered by Community Bio-Gas Plants'].values[0],
                city2_data['Household covered for Vehicles for Collection & Transportation of Waste'].values[0],
                city2_data['Household covered for Segregation Bins at Community Places in the village'].values[0],
                city2_data['Household covered for Waste collection and segregation sheds'].values[0]
            ]
        })

        # Set the 'Metric' column as the index for better formatting
        comparison_data.set_index('Metric', inplace=True)

        # Display the comparison data
        st.write(comparison_data)

        # Check for improvements needed
        for metric in THRESHOLDS:
            check_and_suggest_improvements(
                metric,
                city1,
                comparison_data[f'{city1} - No. of Assets'][metric],
                comparison_data[f'{city1} - Households Covered'][metric]
            )
            check_and_suggest_improvements(
                metric,
                city2,
                comparison_data[f'{city2} - No. of Assets'][metric],
                comparison_data[f'{city2} - Households Covered'][metric]
            )

        # Plotting with Plotly
        fig = go.Figure()
        for column in comparison_data.columns:
            fig.add_trace(go.Bar(x=comparison_data.index, y=comparison_data[column], name=column))

        fig.update_layout(
            title='Comparison of Waste Management Metrics',
            xaxis_title='Metrics',
            yaxis_title='Values',
            barmode='group'
        )

        st.plotly_chart(fig)