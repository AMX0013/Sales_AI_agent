import asyncio
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.exceptions import RefreshError
import os
import pickle
import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import aiohttp

SCOPES = ['https://www.googleapis.com/auth/calendar']



def get_calendar_service():
    creds = None
    if os.path.exists('token.pkl'):
        with open('token.pkl', 'rb') as token:
            creds = pickle.load(token)
    elif os.path.exists('credentials.json'):
            creds = Credentials.from_authorized_user_file('credentials.json')        
    
    if not creds or not creds.valid:
            
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as e:
                print(f"Failed to refresh access token: {e}")
                return None
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)            
        # write new valid creds to pickle and creds.json
        with open('token.pkl', 'wb') as token:
            pickle.dump(creds, token)
        with open('credentials.json', 'w') as f:
            f.write(creds.to_json())
    
    try:
        
        service = build('calendar', 'v3', credentials=creds)
    except HttpError as error:
        print(f"An error occurred while creating the service: {error}")
    return service

def check_availability(start_time, end_time):
    try:
        service = get_calendar_service()
        if service is None:
            print("Unable to spawn Calendar Service")
            return None
        
        # Formatting the start and end time correctly with 'Z'
        formatted_start_time = start_time.isoformat() + 'Z'
        formatted_end_time = end_time.isoformat() + 'Z'
        
        # Creating the request body
        body = {
            "timeMin": formatted_start_time,
            "timeMax": formatted_end_time,
            "items": [{"id": 'primary'}]
        }
        
        # Making the free/busy query request
        events_result = service.freebusy().query(body=body).execute()
        busy_times = events_result['calendars']['primary']['busy']
        
        # If there are no busy times, the time is available
        is_available = not busy_times
        return is_available
    
    except HttpError as error:
        print(f"An error occurred while checking availability: {error}")
        return None


def create_appointment(start_time, end_time, attendee, summary = "Checkout your perfect car at AMX autosalon", description = ""):
    try:
        service = get_calendar_service()
        if service is None:
            return
        event = {
            'summary': "Checkout your perfect car at AMX autosalon",
            'description': "BAsed on our conversation, scheduling the time you mentioned to vist out autosalon to testdrive and hopefully bring home what you've been looking for!",
            'start': {'dateTime': start_time.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': end_time.isoformat(), 'timeZone': 'UTC'},
            'attendees': [{'email': attendee}],
        }
        event = service.events().insert(calendarId='primary', body=event).execute()
        print(f"Event created: {event.get('htmlLink')}")
    except HttpError as error:
        print(f"Failed to create event: {error}")


