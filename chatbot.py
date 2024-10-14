import streamlit as st
import pandas as pd
import spacy

# Load the spaCy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Load your cleanliness dataset (Make sure to use the correct path)
cleanliness_data = pd.read_csv("C:/Users/Hp/Downloads/Cleanest_Cities_India.csv")

# List of known cities (for validation)
known_cities = cleanliness_data['City Name'].str.upper().tolist()


# Function to provide hardcoded tips based on cleanliness score
def get_hardcoded_cleanliness_tips(cleanliness_score):
    tips = {
        "Low": [
            "Increase public awareness campaigns on cleanliness.",
            "Enhance waste management systems, including more bins and collection routes.",
            "Improve sanitation facilities in public areas."
        ],
        "Moderate": [
            "Implement regular cleanliness drives involving the community.",
            "Establish partnerships with local NGOs for waste management education.",
            "Improve maintenance of public spaces and facilities."
        ],
        "High": [
            "Continue to educate the public on maintaining cleanliness.",
            "Recognize and reward communities or individuals who contribute to cleanliness.",
            "Evaluate existing cleanliness programs for effectiveness."
        ]
    }

    if cleanliness_score < 2000:
        return "Tips for improvement:\n- " + "\n- ".join(tips["Low"])
    elif 2000 <= cleanliness_score < 3000:
        return "Tips for improvement:\n- " + "\n- ".join(tips["Moderate"])
    else:
        return "Your city has a high cleanliness score! Keep up the great work!"


# Function to get cleanliness scores
def get_cleanliness_score(city_name, year=None):
    # Handle the case for year 2021
    if year == 2021:
        return "Sorry, data for the year 2021 is not available."

    # Prepare the score column name based on the year
    if year:
        score_column = f"{year} Score"  # e.g., "2018 Score"
        # Check if the column exists in the DataFrame
        if score_column not in cleanliness_data.columns:
            return f"Sorry, data for the year {year} is not available."

        # Get the data for the specified city and year
        city_data = cleanliness_data[cleanliness_data['City Name'].str.upper() == city_name.upper()]

        if not city_data.empty:
            score = city_data[score_column].values[0]
            return f"The cleanliness score for {city_name} in {year} is {score}."
        else:
            return f"Sorry, no cleanliness data available for **{city_name}**."
    else:
        # Get all scores for the specified city (excluding 2021)
        city_data = cleanliness_data[cleanliness_data['City Name'].str.upper() == city_name.upper()]

        if not city_data.empty:
            # Create a DataFrame for scores from 2016 to 2023, excluding 2021
            scores = {year: city_data[f"{year} Score"].values[0] for year in range(2016, 2024) if
                      year != 2021 and f"{year} Score" in cleanliness_data.columns}

            if scores:
                return pd.DataFrame.from_dict(scores, orient='index', columns=['Score']).reset_index().rename(
                    columns={'index': 'Year'})
            else:
                return f"Sorry, no cleanliness scores are available for **{city_name}**."
        else:
            return f"Sorry, no cleanliness data available for **{city_name}**."


# Function to extract city and year from the user input using spaCy
def extract_city_and_year(user_input):
    # Process the user input with spaCy
    doc = nlp(user_input)

    year = None
    city = None

    # Extract the year from the processed text
    for token in doc:
        # Check if the token is numeric
        if token.like_num:
            # Check if it is a valid year in the range
            token_num = int(token.text)
            if token_num in [2016, 2017, 2018, 2019, 2020, 2022, 2023]:
                year = token_num
                break  # Stop after finding the first valid year

    # Find a potential city name from the recognized entities
    for ent in doc.ents:
        if ent.label_ == "GPE":  # GPE: Geopolitical entity (city names)
            city = ent.text
            break  # Stop after finding the first city

    # Fallback: Check if the city is in the known cities list
    if not city:
        for known_city in known_cities:
            if known_city in user_input.upper():
                city = known_city
                break

    return city, year


def chatbot_ui():
    # Streamlit UI

    # Sidebar for additional information
    st.sidebar.title("Chatbot Information")
    st.sidebar.markdown("""
            **How to Use the Chatbot:**
            - Type your query in the text box.
            - Ask about cleanliness scores for different cities and years.

            **Examples of Valid Queries:**
            - "What is the cleanliness score of Indore in 2022?"
            - "Give me the cleanliness scores for Pune."
            - "Tell me the cleanliness score for Bangalore in 2023."

            **Tips:**
            - Make sure to spell city names correctly.
            - You can ask for scores from the years 2016, 2017, 2018, 2019, 2020, 2022, and 2023 (note: 2021 data is not available).
        """)



    st.title("Cleanliness Score Chatbot")
    st.header("Welcome to the Cleanliness Score Chatbot!")

    # User input
    user_message = st.text_input("You: ", "")

    # Check if user input is empty
    if user_message.strip() == "":
        st.warning("Please enter a message to ask about the cleanliness score.")
        return  # Exit the function early if there's no input

    # Extract city and year from user input
    city, year = extract_city_and_year(user_message)

    # Check if city is defined and valid
    if city and city.upper() in known_cities:
        # If year is defined, get the score
        if year:
            score_response = get_cleanliness_score(city, year)
        else:
            # Get all scores for the city if no year is provided
            score_response = get_cleanliness_score(city)

        # Display the response
        if isinstance(score_response, pd.DataFrame):
            st.write(f"Bot: Here are the cleanliness scores for **{city}**:")
            st.table(score_response)  # Present scores in a table format

            # Check the latest score from the response
            latest_score = score_response['Score'].values[-1]
            st.write(f"Latest cleanliness score: {latest_score}")  # Debugging line

            # If the score is less than 3000, generate tips
            if latest_score < 3000:
                # Get hardcoded tips based on the cleanliness score
                tips = get_hardcoded_cleanliness_tips(latest_score)
                st.write(f"Bot: Here are some suggestions to improve the cleanliness score of {city}:")
                st.write(tips)
            else:
                st.write("Bot: Your city has a high cleanliness score! Keep up the great work!")
        else:
            st.write(f"Bot: {score_response}")
    else:
        if city:
            st.write(
                f"Bot: I couldn't find cleanliness data for **{city}**. Please check the spelling or use a different city.")
        else:
            st.write(
                "Bot: I couldn't identify the city from your input. Please provide a valid input format like 'Give me the cleanliness score of Indore in 2018.'")
