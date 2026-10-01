import pandas as pd
import numpy as np

actual_data = {
    0: (1.5, 1.5, 1.5), 5: (20.1, 43.3, 37.5), 7: (64.3, 109.0, 85.8),
    9: (98.4, 143.9, 125.0), 13: (246.6, 475.8, 358.0), 17: (435.8, 743.4, 594.8),
    22: (873.2, 1364.7, 1135.5), 28: (1045.3, 1843.6, 1667.4)
}
actual_days = sorted(list(actual_data.keys()))
feeding_days = {0: 50, 5: 50, 7: 100, 9: 500, 13: 1000, 17: 2000, 22: 3000, 28: 1000}

data = []
np.random.seed(42)
cum_feed = 0

for day in range(0, 29):
    cum_feed += feeding_days.get(day, 0)
    
    if day in actual_data:
        prev_d, next_d = day, day
    else:
        prev_d = max([d for d in actual_days if d < day])
        next_d = min([d for d in actual_days if d > day])
        
    for i, sub in enumerate(['Banana', 'Tofu', 'Food Waste']):
        v_prev = actual_data[prev_d][i]
        v_next = actual_data[next_d][i]
        
        base_yield = v_prev if day in actual_data else v_prev + ((day - prev_d) * (v_next - v_prev) / (next_d - prev_d))
        
        for _ in range(3):
            temp = round(np.random.normal(29.0, 1.0), 2)
            hum = round(np.random.normal(75.0, 3.0), 2)
            
            if day == 0:
                noise = 0
            else:
                std_dev_variance = base_yield * 0.15 
                noise = round(np.random.normal(0, std_dev_variance), 2)
                
            yield_g = max(1.5, round(base_yield + noise, 2))
            
            data.append({
                'Day': day,
                'Substrate': sub,
                'Cumulative_Feed_Given (g)': cum_feed,
                'Temperature_C': temp,
                'Humidity_Percent': hum,
                'Yield_g': yield_g
            })

df = pd.DataFrame(data)
df.to_excel("THESIS BSF DATASET.xlsx", index=False)

print("SUCCESS: THESIS BSF DATASET.xlsx generated with 15% controlled synthetic variation.")