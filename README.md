# ml-app-auto-mpg-efficiency

The app is deployed on https://vehicle-mpg-prediction.onrender.com/ \
But you can only query using post request.

To test the app, run the following python code:
```
# sample values, you can give your own vehicle configuration as well 
vehicle_config = {
    'Cylinders': [4, 6, 8],
    'Displacement': [155.0, 160.0, 165.5],
    'Horsepower': [93.0, 130.0, 98.0],
    'Weight': [15.0, 14.0, 16.0],
    'Acceleration': [15.0, 14.0, 16.0],
    'Model Year': [81, 80, 78],
    'Origin': [3, 2, 1]
}



# Code to get the mpg values from the model deployed on render server
import requests

url  = 'https://vehicle-mpg-prediction.onrender.com/'
r = requests.post(url, json = vehicle_config)
print(r.text.strip())
```

Based on: https://github.com/dswh/fuel-consumption-end-to-end-ml
