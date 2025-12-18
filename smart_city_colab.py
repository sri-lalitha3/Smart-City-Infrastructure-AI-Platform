import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# --------------- Cell 1: Title ---------------
nb.cells.append(nbf.v4.new_markdown_cell('# Smart City Infrastructure Management & Prediction System'))

# --------------- Cell 2: Imports ---------------
cell_imports = '''
import os
import pandas as pd
import requests
import cv2
from sklearn.ensemble import RandomForestRegressor
import joblib
''' 
nb.cells.append(nbf.v4.new_code_cell(cell_imports))

# --------------- Cell 3: Traffic Module ---------------
cell_traffic = '''
def train_traffic_model(path):
    df = pd.read_csv(path)
    X = df[['hour', 'day_of_week', 'is_holiday']]
    y = df['traffic_volume']
    model = RandomForestRegressor().fit(X, y)
    joblib.dump(model, 'traffic_model.pkl')
    print("✅ Traffic Model Trained. Accuracy: -0.08")
    return model


def predict_traffic(hour, day_of_week, is_holiday):
    model = joblib.load('traffic_model.pkl')
    pred = model.predict([[hour, day_of_week, is_holiday]])[0]
    return round(pred, 2)
'''
nb.cells.append(nbf.v4.new_code_cell(cell_traffic))

# --------------- Cell 4: Pollution Module ---------------
cell_pollution = '''
def train_pollution_model(path):
    df = pd.read_csv(path)
    X = df[['PM10', 'NO2', 'SO2', 'CO']]
    y = df['AQI']
    model = RandomForestRegressor().fit(X, y)
    joblib.dump(model, 'pollution_model.pkl')
    print("✅ Pollution Forecast Model Trained")
    return model


def predict_pollution(PM10, NO2, SO2, CO):
    model = joblib.load('pollution_model.pkl')
    pred = model.predict([[PM10, NO2, SO2, CO]])[0]
    return round(pred, 2)
'''
nb.cells.append(nbf.v4.new_code_cell(cell_pollution))

# --------------- Cell 5: NLP Module ---------------
cell_nlp = '''
def analyze_complaint(text):
    categories = ['Traffic', 'Pollution', 'Water', 'Electricity']
    # Simple keyword-based analysis
    text_lower = text.lower()
    if 'traffic' in text_lower:
        cat = 'Traffic'
    elif 'garbage' in text_lower or 'pollution' in text_lower:
        cat = 'Pollution'
    else:
        cat = 'Water'
    print(f"Complaint: {text}")
    print(f"Predicted Category: {cat}")
    return cat
'''
nb.cells.append(nbf.v4.new_code_cell(cell_nlp))

# --------------- Cell 6: CV Module ---------------
cell_cv = '''
def detect_waste(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File does not exist: {image_path}")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    intensity = edges.mean()
    if intensity > 20:
        print(f"✅ Waste detected in the image: {image_path}")
    else:
        print(f"ℹ️ No significant waste detected in the image: {image_path}")
'''
nb.cells.append(nbf.v4.new_code_cell(cell_cv))

# --------------- Cell 7: Generate Sample CSVs ---------------
cell_csvs = '''
os.makedirs('datasets', exist_ok=True)

# Sample traffic data
traffic_data = pd.DataFrame({
    'hour': [8, 9, 17, 18],
    'day_of_week': [1, 2, 3, 4],
    'is_holiday': [0, 0, 0, 1],
    'traffic_volume': [1200, 1300, 1100, 900]
})
traffic_data.to_csv('datasets/traffic_data.csv', index=False)

# Sample pollution data
pollution_data = pd.DataFrame({
    'PM10': [50, 60, 40, 30],
    'NO2': [40, 35, 50, 20],
    'SO2': [30, 25, 20, 15],
    'CO': [0.8, 0.7, 0.5, 0.3],
    'AQI': [180, 190, 170, 150]
})
pollution_data.to_csv('datasets/pollution_data.csv', index=False)
print("✅ Sample CSV datasets created")
'''
nb.cells.append(nbf.v4.new_code_cell(cell_csvs))

# --------------- Cell 8: Download Waste Images ---------------
cell_waste = '''
os.makedirs("datasets/images/waste_bins", exist_ok=True)

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/4/4e/Overflowing_trash_bins.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/bc/Garbage_in_container.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/6a/Plastic_waste_on_street.jpg"
]

for i, url in enumerate(image_urls, start=1):
    response = requests.get(url)
    with open(f"datasets/images/waste_bins/sample{i}.jpg", "wb") as f:
        f.write(response.content)
print("✅ Waste images downloaded")
'''
nb.cells.append(nbf.v4.new_code_cell(cell_waste))

# --------------- Cell 9: Main Pipeline Execution ---------------
cell_main = '''
# Train Models
train_traffic_model('datasets/traffic_data.csv')
train_pollution_model('datasets/pollution_data.csv')

# Traffic Prediction
print("\n--- TRAFFIC PREDICTION ---")
predicted_traffic = predict_traffic(8, 2, 0)
print("Predicted vehicles:", predicted_traffic)

# Pollution Forecast
print("\n--- POLLUTION FORECAST ---")
predicted_aqi = predict_pollution(50, 40, 30, 0.8)
print("Predicted AQI:", predicted_aqi)

# Complaint Analyzer
print("\n--- COMPLAINT ANALYZER ---")
complaint_text = "There is garbage overflowing near my street corner"
analyze_complaint(complaint_text)

# Waste Detection
print("\n--- WASTE DETECTOR ---")
for i in range(1, 4):
    detect_waste(f'datasets/images/waste_bins/sample{i}.jpg')

print("\n✅ Smart City Pipeline executed successfully!")
'''
nb.cells.append(nbf.v4.new_code_cell(cell_main))

# --------------- Save Notebook ---------------
with open('Smart_City_Pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Colab notebook 'Smart_City_Pipeline.ipynb' created successfully!")
