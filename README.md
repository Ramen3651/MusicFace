# MusicFace — Emotion-Based Music Recommendation App

## Project Overview
MusicFace is a mobile-style application that recommends music based on 
the user's detected facial emotion using real-time webcam capture, 
DeepFace emotion recognition, and a Spotify-style audio feature dataset.

## Student
Mohammad Tamjidur Rahman — w1976483  
BSc (Hons) Computer Science  
University of Westminster  
Supervisor: Alexandra Psarrou

## Requirements
Install dependencies with:
pip install kivy flask deepface opencv-python pandas matplotlib

## Dataset
Place music_features_subset.csv in the same folder as app.py before running.

## How to Run
Step 1 — Open Terminal 1 and run the Flask backend:
    python app.py

Step 2 — Open Terminal 2 and run the Kivy frontend:
    python mobile_app.py

## Testing DeepFace Independently
python test_emotion.py
This script tests emotion detection under different conditions and 
prints per-frame confidence scores. Used for accuracy testing in Chapter 7.

## Files
- app.py          — Flask backend API and recommendation engine
- mobile_app.py   — Kivy frontend mobile-style application  
- database.py     — SQLite database operations and user account management
- test_emotion.py — Standalone DeepFace accuracy testing script