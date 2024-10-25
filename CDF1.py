import streamlit as st
import pandas as pd
import sqlite3
import json
from openai import OpenAI
import os
import chardet
import io
import re
from collections import Counter
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import uuid
from datetime import datetime


# Constants
DB_NAME = 'data.db'
CC_TABLE_NAME = "cc_data"
MV_TABLE_NAME = "mv_data"

DATA_OPTIONS = {
    "CC DATA": {
        "csv_file": "CCDATAOCT19.csv",
        "explanation_file": "CCQueriesDescription.txt",
        "table_name": CC_TABLE_NAME
    },
    "MV DATA": {
        "csv_file": "MVDATAOCT17.csv",
        "explanation_file": "MVQueriesDescription.txt",
        "table_name": MV_TABLE_NAME
    },
}
# OpenAI API setup

API_KEYS = [
    st.secrets["OPENAI_API_KEY"],
    st.secrets["OPENAI_API_KEY_2"],
    st.secrets["OPENAI_API_KEY_3"],
]
    
MODELS = ["gpt-4", "gpt-4","gpt-4"]


# Google Sheets API setup
gcreds = st.secrets["gcp_service_account"]
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]


def authenticate_google_sheets():
    # Use the service account credentials for authentication
    creds = ServiceAccountCredentials.from_json_keyfile_dict(gcreds, scope)
    client = gspread.authorize(creds)
    
    # Open the sheet (replace 'Your Google Sheet Name' with your actual sheet name)
    sheet = client.open('ChatData LOG').sheet1
    return sheet
    
def append_to_google_sheet(chat_id, user_input, json_answer, sql_query, gpt_response, dataset_name):
    sheet = authenticate_google_sheets()
    
    # Get the current date and time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Prepare the data to append (added dataset_name column)
    row_data = [chat_id, current_time, user_input, json_answer, sql_query, gpt_response, dataset_name]
    
    # Append the row to the Google Sheet
    sheet.append_row(row_data)


# Global variables for prompts
general_bot_prompt = ''' 
You are a friendly and helpful data assistant for Maids.cc, a company that provides maid services through both temporary (CC) and long-term visa (MV) contracts. Your responses should:

1. Always maintain context about being a data assistant
2. Be conversational but professional
3. When referencing data, ALWAYS specify whether it's from CC or MV (Maid Visa) dataset
4. Guide users towards data-related queries when possible
5. Be concise but informative
6. Show personality while staying focused on your role

Remember:
- CC data relates to temporary maid contracts and placements for Contract & Cancellation services
- MV data relates to long-term Maid Visa contracts and visa-related services
- Always prefix data references with "CC" or "MV" for clarity
- You can analyze patterns, trends, and statistics in both datasets'''

if 'sql_generation_prompt' not in st.session_state:
    st.session_state.sql_generation_prompt =  '''
    User's explanation of the CSV:
    {csv_explanation}

    A user will now chat with you. Your task is to transform the user's request into an SQL query that retrieves exactly what they are asking for.

    Rules:
    1. Return only two JSON variables: "Explanation" and "SQL".
    2. The Explanation should always specify whether the data is from CC or MV (Maid Visa) dataset.
    3. No matter how complex the user question is, return only one SQL query.
    4. Always return the SQL query in a one-line format.
    5. Consider the chat history when generating the SQL query.
    6. The query can return multiple rows if appropriate for the user's question.
    7. You shall not use functions like MONTH(),HOUR(),HOUR,YEAR(),DATEIFF(),.....
    8. Use only queries proper for sql LITE
    9. YOU CAN ONLY RETURN ONE SQL STATEMENT AT A TIME, COMBINE YOUR ANSWER IN ONLY ONE STATEMENT, NEVER 2 or MORE, Find workarounds.
    10. Ignore Null values in interpretations and calculations only consider them where they are relevant.

    Example output for CC data:
    {{
    "Explanation": "[CC DATA] The user is asking about the top 5 contract cancellations by age. We'll query the CC dataset to select the name and age columns from cancellation records, order by age descending, and limit to 5 results.",
    "SQL": "SELECT name, age FROM {table_name} ORDER BY age DESC LIMIT 5"
    }}

    Example output for MV data:
    {{
    "Explanation": "[MV DATA] The user is asking about visa processing times. We'll query the MV dataset to analyze the duration between application and approval dates, ordering by processing time descending.",
    "SQL": "SELECT application_date, approval_date FROM {table_name} ORDER BY approval_date - application_date DESC"
    }}

    Your prompt ends here. Everything after this is the chat with the user. Remember to always return the accurate SQL query with clear dataset identification.
    '''

if 'response_generation_prompt' not in st.session_state:
    st.session_state.response_generation_prompt = '''
    User's explanation of the CSV:
    {csv_explanation}

    Now you will receive a JSON containing the SQL output that answers the user's inquiry. The output may contain multiple rows of data. Your task is to use the SQL's output to answer the user's inquiry in plain English.

    Rules:
    1. ALWAYS start your response by indicating whether you're analyzing CC or MV (Maid Visa) data
    2. Use "[CC DATA]" or "[MV DATA]" prefix at the start of your response
    3. When mentioning specific metrics or trends, clarify which dataset they come from
    4. Consider the chat history when generating your response
    5. If there are multiple results, summarize them appropriately
    6. Be clear about which type of service (CC or MV) the insights relate to

    Example:
    "[CC DATA] Based on the contract cancellation records, I found that..."
    "[MV DATA] Looking at the visa application data, we can see that..."
    '''

############################################### HELPER FUNCTIONS ########################################################
def load_dataset_descriptions():
    descriptions = {}
    for dataset_name, files in DATA_OPTIONS.items():
        try:
            with open(files["explanation_file"], 'r', encoding='utf-8') as file:
                descriptions[dataset_name] = file.read()
        except Exception as e:
            st.error(f"Error loading description file for {dataset_name}: {e}")
            return None
    return descriptions
def generate_triage_prompt(cc_description, mv_description):
    return f'''
You are a triage bot that determines whether a query is about CC data, MV data, or if it's a general message.

CC DATA Description:
{cc_description}

MV DATA Description:
{mv_description}

RULES:
1. Output should be in JSON format with two fields: "classification" and "clarification_needed"
2. For "classification", use one of these values:
   - "CC DATA": For queries about Maids.cc company data, contract services, maid placements, etc.
   - "MV DATA": For queries about long-term maid visa contracts and related data
   - "GENERAL": For general messages, greetings, questions about capabilities, or conversation
3. Set "clarification_needed" to true if you need more context to determine the dataset
4. For ambiguous queries between CC and MV data, analyze the context carefully

Example outputs:
{{"classification": "GENERAL", "clarification_needed": false}}
{{"classification": "CC DATA", "clarification_needed": false}}
{{"classification": "MV DATA", "clarification_needed": true, "question": "Are you asking about new visa contracts or existing visa transfers?"}}
'''

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    return chardet.detect(raw_data)['encoding']

def get_data_type(values):
    if all(isinstance(val, (int, float)) for val in values if pd.notna(val)):
        return "Number"
    elif all(isinstance(val, str) for val in values if pd.notna(val)):
        return "Text"
    elif all(pd.to_datetime(val, errors='coerce') is not pd.NaT for val in values if pd.notna(val)):
        return "Date"
    else:
        return "Mixed"

def analyze_csv(file_path, max_examples=3):
    encoding = detect_encoding(file_path)
    df = pd.read_csv(file_path, encoding=encoding)
    
    prompt = "This CSV file contains the following columns:\n\n"
    
    for col in df.columns:
        values = df[col].dropna().tolist()
        data_type = get_data_type(values)
        
        unique_count = df[col].nunique()
        total_count = len(df)
        is_unique = unique_count == total_count
        
        examples = df[col].dropna().sample(min(max_examples, len(values))).tolist()
        
        prompt += f"Column: {col}\n"
        prompt += f"Data Type: {data_type}\n"
        prompt += f"Examples: {', '.join(map(str, examples))}\n"
        
        if is_unique:
            prompt += "Note: This column contains unique values for each row.\n"
        
        null_count = df[col].isnull().sum()
        if null_count > 0:
            prompt += f"Note: This column contains {null_count} NULL values.\n"
        
        if data_type == "Text":
            value_counts = Counter(values)
            most_common = value_counts.most_common(3)
            if len(most_common) < len(value_counts):
                prompt += f"Most common values: {', '.join(f'{val} ({count})' for val, count in most_common)}\n"
        
        prompt += "\n"
    
    return prompt

def reset_chat():
    st.session_state.messages = []

def display_sql_query(query):
    with st.expander("View SQL Query", expanded=False):
        st.code(query, language="sql")

def display_json_data(json_data):
    with st.expander("View JSON Data", expanded=False):
        if isinstance(json_data, list):
            for item in json_data:
                st.json(item)
        else:
            st.json(json_data)

def df_to_sqlite(df, table_name, db_name=DB_NAME):
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"An error occurred while creating the table: {e}")
        return False

# New function to update prompts
def update_prompt(prompt_type):
    if prompt_type == "SQL Generation":
        st.session_state.sql_generation_prompt = st.session_state.sql_generation_prompt_input
    elif prompt_type == "Response Generation":
        st.session_state.response_generation_prompt = st.session_state.response_generation_prompt_input

############################################## AI INTERACTION FUNCTIONS ######################################################

def try_api_call(func, *args, **kwargs):
    for api_key in API_KEYS:
        for model in MODELS:
            try:
                client = OpenAI(api_key=api_key)
                return func(client, model, *args, **kwargs)
            except Exception as e:
                print(f"Error with API key {api_key[:5]}... and model {model}: {str(e)}")
    return None  # If all combinations fail

def rephrase_query(user_input, attempt):
    # Slight modifications to the query based on attempt
    if attempt == 1:
        return user_input + " (simplified)"
    elif attempt == 2:
        return "Please find relevant data: " + user_input
    elif attempt == 3:
        return "Extract from data: " + user_input
    else:
        return user_input

def generate_sql_query(user_input, prompt, chat_history, retry_attempts=3):
    attempt = 0
    response = None

    while attempt < retry_attempts:
        attempt += 1
        try:
            # Limit the chat history context to a few latest messages
            reduced_chat_history = chat_history[-3:]

            def api_call(client, model, user_input, prompt, chat_history):
                messages = [
                    {"role": "system", "content": prompt},
                ]

                for message in chat_history:
                    messages.append({"role": message["role"], "content": message["content"]})

                messages.append({"role": "user", "content": user_input})

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=1,
                    stop=None,
                    temperature=0,
                )
                return response.choices[0].message.content.strip()

            # Modify the input based on attempt number
            modified_input = rephrase_query(user_input, attempt)
            response = try_api_call(api_call, modified_input, prompt, reduced_chat_history)

            if response:
                try:
                    sql_data = json.loads(response)
                    if "SQL" in sql_data:
                        return response  # Success on generating valid SQL query
                except json.JSONDecodeError:
                    st.warning(f"Attempt {attempt}: Failed to generate a valid SQL query response. Retrying...")

        except Exception as e:
            st.warning(f"Attempt {attempt}: Encountered an error: {str(e)}. Retrying...")

    st.error("Failed to generate a valid SQL query after multiple attempts. Please try rephrasing your question.")
    return None


def execute_query_and_save_json(input_string, dataset_type, db_name=DB_NAME, retry_attempts=3):
    try:
        # Parse the SQL query from input
        sql_data = json.loads(input_string)
        sql_query = sql_data["SQL"]
        
        # Replace the table name placeholder with the correct table
        table_name = DATA_OPTIONS[dataset_type]["table_name"]
        sql_query = sql_query.format(table_name=table_name)
        
    except json.JSONDecodeError:
        st.error("Failed to parse SQL query response.")
        return None

    # Connect to database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    try:
        # Execute the query
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Get column names
        column_names = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries with proper type handling
        result_list = []
        for row in results:
            row_dict = {}
            for i, value in enumerate(row):
                if value is None:
                    row_dict[column_names[i]] = None
                elif isinstance(value, (int, float)):
                    row_dict[column_names[i]] = value
                else:
                    row_dict[column_names[i]] = str(value)
            result_list.append(row_dict)

        return result_list

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        st.error(f"Database error: {e}")
        return None
    finally:
        conn.close()

def generate_response(json_data, prompt, chat_history):
    def api_call(client, model, json_data, prompt, chat_history):
        # Convert results to a clean string format
        if isinstance(json_data, list):
            data_str = json.dumps(json_data, ensure_ascii=False, indent=2)
        else:
            data_str = str(json_data)
            
        messages = [
            {"role": "system", "content": prompt},
        ]
        
        # Add relevant chat history
        for message in chat_history[-3:]:
            messages.append({"role": message["role"], "content": message["content"]})
        
        # Add the data context
        messages.append({
            "role": "user", 
            "content": f"Here are the query results:\n{data_str}\n\nPlease provide a natural language summary of these results."
        })
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            n=1,
            stop=None,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    
    return try_api_call(api_call, json_data, prompt, chat_history)


def general_bot_conversation(user_input, chat_history):
    def api_call(client, model, user_input, chat_history, prompt):
        messages = [
            {"role": "system", "content": prompt}
        ]
        
        # Include relevant chat history for context
        for message in chat_history[-3:]:  # Last 3 messages for context
            messages.append({"role": message["role"], "content": message["content"]})
            
        messages.append({"role": "user", "content": user_input})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            n=1,
            stop=None,
            temperature=0.7  # Slightly higher temperature for more natural conversation
        )
        return response.choices[0].message.content.strip()
    
    return try_api_call(api_call, user_input, chat_history, general_bot_prompt)

def triage_query(user_input, chat_history, dataset_descriptions):
    def api_call(client, model, user_input, chat_history, prompt):
        messages = [
            {"role": "system", "content": prompt},
        ]

        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            n=1,
            stop=None,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    triage_prompt = generate_triage_prompt(
        dataset_descriptions["CC DATA"],
        dataset_descriptions["MV DATA"]
    )
    
    response= try_api_call(api_call, user_input, chat_history, triage_prompt)
    try:
        triage_result = json.loads(response)
        if triage_result["classification"] == "GENERAL":
            general_response = general_bot_conversation(user_input, chat_history)
            return "GENERAL", general_response
        elif triage_result["clarification_needed"]:
            return "CLARIFICATION", triage_result.get("question", "Could you please clarify what data are you refering to?")
        else:
            return triage_result["classification"], None
    except (json.JSONDecodeError, KeyError):
        return "CC DATA", None  # Default to CC DATA if parsing fails


# Load data function (modified to work in the backend)
@st.cache_data
def load_data(file_path, table_name):
    try:
        encoding = detect_encoding(file_path)
        df = pd.read_csv(file_path, encoding=encoding)
        csv_analysis = analyze_csv(file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None, None

    if not df_to_sqlite(df, table_name):
        return None, None, None
    
    return df, table_name, csv_analysis

def initialize_database():
    """Initialize both tables at startup"""
    for dataset_info in DATA_OPTIONS.values():
        df, table_name, csv_analysis = load_data(
            dataset_info["csv_file"],
            dataset_info["table_name"]
        )
        if df is None:
            st.error(f"Failed to load {dataset_info['csv_file']}")
            return False
    return True

def load_explanation_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"An error occurred while reading the explanation file: {e}")
        return None

def main():
    st.set_page_config(layout="wide", page_title="Maids.cc DataChat", page_icon="📈")

    # Initialize session state variables
    if 'chat_id' not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'database_initialized' not in st.session_state:
        st.session_state.database_initialized = initialize_database()

    if not st.session_state.database_initialized:
        st.error("Failed to initialize database. Please check your data files.")
        return

    # Page header and reset button
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("Maids.cc Data Assistant")
    with col2:
        if st.button("Reset Chat", key="reset_top"):
            st.session_state.messages = []
            st.rerun()

    # Load dataset descriptions at startup
    if 'dataset_descriptions' not in st.session_state:
        st.session_state.dataset_descriptions = load_dataset_descriptions()
        if st.session_state.dataset_descriptions is None:
            st.error("Failed to load dataset descriptions. Please check the description files.")
            return

    # Display introductory message for new chats
    if len(st.session_state.messages) == 0:
        welcome_message = {
            "role": "assistant",
            "content": "👋 Hello! I'm your Maids.cc data assistant. If you have any questions about our CC or MV Data, feel free to ask!"
        }
        st.session_state.messages.append(welcome_message)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask me about Maids.cc data or start a conversation")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        result_list = None
        response = None
        sql_query = None

        with st.spinner("Processing your message..."):
            # Determine message type and appropriate response
            classification, additional_response = triage_query(
                user_input, 
                st.session_state.messages,
                st.session_state.dataset_descriptions
            )
            
            if classification == "GENERAL":
                # Handle general conversation
                response = additional_response
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

            elif classification == "CLARIFICATION":
                # Handle cases where clarification is needed
                response = additional_response
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)

            else:
                # Handle data queries (CC or MV)
                selected_dataset = classification
                explanation_file_path = DATA_OPTIONS[selected_dataset]["explanation_file"]

                # Load explanation text (data is already loaded in database)
                explanation_text = load_explanation_from_file(explanation_file_path)
                if explanation_text is None:
                    st.error("Failed to load the dataset explanation. Please try again later.")
                    return

                st.session_state.csv_explanation = explanation_text

                # Generate and execute SQL query
                sql_generation_prompt = st.session_state.sql_generation_prompt.format(
                    csv_explanation=st.session_state.csv_explanation,
                    table_name="{table_name}"  # Placeholder for later replacement
                )
                sql_query_response = generate_sql_query(user_input, sql_generation_prompt, st.session_state.messages[:-1])

                if sql_query_response:
                    try:
                        # Execute query with the correct dataset type
                        result_list = execute_query_and_save_json(sql_query_response, selected_dataset)
                        
                        if result_list is not None:
                            # Debug printing
                            print("Query results obtained:", len(result_list), "rows")
                            
                            # Generate natural language response
                            response_generation_prompt = st.session_state.response_generation_prompt.format(
                                csv_explanation=st.session_state.csv_explanation
                            )
                            
                            response = generate_response(result_list, response_generation_prompt, st.session_state.messages)
                            
                            if response:
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                with st.chat_message("assistant"):
                                    st.markdown(response)
                                
                                # Debug information
                                with st.expander("Query Details", expanded=False):
                                    try:
                                        sql_data = json.loads(sql_query_response)
                                        sql_query = sql_data["SQL"]
                                       # st.code(sql_data["SQL"], language="sql")
                                       # st.json(result_list[:5])  # Show first 5 results
                                    except Exception as e:
                                        st.error(f"Error displaying debug info: {e}")
                            else:
                                st.error("Failed to generate a response from the results.")
                        else:
                            st.error("No results returned from query execution.")
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.exception(e)  # Show full traceback in development
                else:
                    st.error("Failed to generate SQL query.")

        # Log interaction to Google Sheets
        try:
            append_to_google_sheet(
                st.session_state.chat_id,
                user_input,
                json.dumps(result_list) if result_list else None,
                sql_query,
                response,
                classification
            )
        except Exception as e:
            st.warning("Failed to log interaction, but your query was processed successfully.")

    # Add a footer with helpful information
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        💡 Please Note:
this tool is in the pilot phase, limit inquiries to one-layer questions (no follow-ups). 
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
