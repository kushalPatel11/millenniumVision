# **Millennium Vision**

## Valorant Game Analysis and Improvement Suggestion Platform

## Overview

This project is a web application that takes a video stream of Valorant gameplay as input, analyzes it using a Convolutional Neural Network (CNN), and generates actionable insights and suggestions for improvement in future games. The platform is built using modern web development technologies such as **Nest.js**, **React.js**, **MongoDB**, and a custom **CNN model** for video analysis.

## Features

- **Video Upload**: Upload gameplay video streams of Valorant for analysis.
- **Game Analysis**: Extracts key gameplay metrics such as headshots, missed shots, and accuracy trends.
- **Improvement Suggestions**: Provides actionable insights and suggestions on how to improve for the next game.
- **User-Friendly Dashboard**: React.js frontend for displaying game statistics and recommendations.
- **Scalable Backend**: Nest.js backend to handle video processing and data analysis.
- **Database**: MongoDB for storing user data, game analytics, and improvement history.

## Technologies Used

- **Frontend**: React.js
- **Backend**: Nest.js
- **Database**: MongoDB
- **Machine Learning**: Convolutional Neural Network (CNN) for video analysis
- **Cloud Storage**: (Optional) For storing video files if needed
- **Deployment**: Docker for containerization (if used), any cloud platform for deployment (AWS, Heroku, etc.)



## Setup and Installation

### Prerequisites

- **Node.js** and **npm** installed
- **Python** installed (for training the model)
- **MongoDB** installed or access to a MongoDB Atlas cluster
- **Docker** (optional, for containerization)

## Usage

1. Upload Gameplay: Use the frontend interface to upload your Valorant gameplay video stream.
2. Generate Analysis: The backend processes the video, analyzes key gameplay events, and stores the results in MongoDB.
3. View Insights: The frontend displays game statistics and provides actionable insights on areas like accuracy, shot patterns, and headshot ratio.
4. Improve: Use the suggestions provided to enhance your gameplay for the next match.

## Challenges
Lack of Data: We had to generate data on our own as we needed Valorant game streams for model training. Since, there is not much data in terms of videos available, we faced this issue.


## Future Enhancements
1. Support for More Games: Extend the platform to support other games beyond Valorant.
2. Real-Time Analysis: Add functionality for real-time gameplay analysis.
3. User Authentication: Implement user authentication and profiles to track individual progress over time.