import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Check the column names for debugging
    print("Columns in the dataset:", df.columns)

    # Feature engineering
    df['headshot_ratio'] = df['headshots'] / df['shots_fired']
    df['missed_shot_ratio'] = df['shots_missed'] / df['shots_fired']
    df['accuracy_trend'] = df['accuracy'].rolling(window=3).mean()
    df.fillna(0, inplace=True)

    # Generate insights based on gameplay stats
    df = generate_insights(df)

    # Features and target
    X = df[['headshot_ratio', 'missed_shot_ratio', 'accuracy_trend', 'crosshair_placement', 'spray_control']]
    y = df['insight']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Generate insights based on predefined conditions
def generate_insights(df):
    # Create an insight based on gameplay performance
    conditions = [
        (df['headshot_ratio'] > 0.5),  # High headshot ratio
        (df['accuracy'] < 50),         # Low accuracy
        (df['spray_control'] < 3)      # Poor spray control
    ]
    
    # Corresponding insights for each condition
    choices = ['improve_crosshair_placement', 'improve_aim_accuracy', 'work_on_spray_control']
    
    # Create the 'insight' column
    df['insight'] = np.select(conditions, choices, default='balanced_performance')

    return df
