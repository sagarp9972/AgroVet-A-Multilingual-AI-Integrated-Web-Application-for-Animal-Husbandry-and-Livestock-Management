from flask import Flask, render_template, request

app = Flask(__name__)

def get_rearing_guidance(animal_type, count):
    animal_data = {
        'Cow': {'area_size': 19.81, 'feed_type': 'Green Feed', 'feed_quantity': 24.85, 'cost_in_inr': 6652.15, 'water_supply': 68.52, 'hygiene_tips': 'Clean barn daily, vaccinate regularly', 'seasonal_tips': 'Summer: Provide shade and cool water, Winter: Ensure warmth, Rainy: Avoid damp areas.'},
        'Buffalo': {'area_size': 24.58, 'feed_type': 'Dry Feed', 'feed_quantity': 31.45, 'cost_in_inr': 7483.52, 'water_supply': 81.23, 'hygiene_tips': 'Ensure proper ventilation', 'seasonal_tips': 'Summer: Provide shade and cool water, Winter: Provide shelter, Rainy: Dry bedding is essential.'},
        'Pig': {'area_size': 15.25, 'feed_type': 'Concentrates', 'feed_quantity': 5.15, 'cost_in_inr': 4989.25, 'water_supply': 50.60, 'hygiene_tips': 'Use disinfectants weekly', 'seasonal_tips': 'Summer: Keep cool and dry, Winter: Ensure warmth, Rainy: Keep pens dry.'},
        'Goat': {'area_size': 10.21, 'feed_type': 'Green Feed', 'feed_quantity': 3.25, 'cost_in_inr': 3325.60, 'water_supply': 20.55, 'hygiene_tips': 'Provide clean bedding', 'seasonal_tips': 'Summer: Provide fresh water and shade, Winter: Ensure dry and warm shelter, Rainy: Avoid damp areas.'},
        'Sheep': {'area_size': 12.50, 'feed_type': 'Green Feed', 'feed_quantity': 4.75, 'cost_in_inr': 3741.75, 'water_supply': 25.80, 'hygiene_tips': 'Check for parasites regularly', 'seasonal_tips': 'Summer: Provide plenty of water, Winter: Keep warm with shelter, Rainy: Keep dry and prevent wool rot.'},
        'Poultry': {'area_size': 1.5, 'feed_type': 'Grain-based Feed', 'feed_quantity': 0.25, 'cost_in_inr': 120, 'water_supply': 0.5, 'hygiene_tips': 'Keep the coop clean and free from pests', 'seasonal_tips': 'Summer: Ensure proper ventilation, Winter: Provide heat and protection, Rainy: Keep the coop dry.'},
        'Bee Hiving': {'area_size': 10.25, 'feed_type': 'Sugar Syrup', 'feed_quantity': 0.5, 'cost_in_inr': 1500.55, 'water_supply': 0.25, 'hygiene_tips': 'Check for pests like mites and wax moths', 'seasonal_tips': 'Summer: Provide ample water sources, Winter: Ensure the bees have enough food stores, Rainy: Avoid disturbances.'},
        'Sericulture': {'area_size': 4.85, 'feed_type': 'Mulberry Leaves', 'feed_quantity': 0.1, 'cost_in_inr': 198.25, 'water_supply': 0.2, 'hygiene_tips': 'Maintain clean environment, avoid humidity', 'seasonal_tips': 'Summer: Maintain cool and dry conditions, Winter: Keep warm, Rainy: Ensure adequate airflow.'}
    }
    
    if animal_type in animal_data:
        data = animal_data[animal_type]
        return (data['area_size'] * count, data['feed_type'], data['feed_quantity'] * count, data['cost_in_inr'] * count, data['water_supply'] * count, data['hygiene_tips'], data['seasonal_tips'])
    else:
        return (0, 'Unknown', 0, 0, 0, 'No information available', 'No seasonal tips available')

@app.route('/')
def home():
    return render_template('s1.html')

@app.route('/result', methods=['POST'])
def result():
    animal_type = request.form['animal_type']
    number_of_animals = int(request.form['number_of_animals'])
    
    area_size, feed_type, feed_quantity, cost_in_inr, water_supply, hygiene_tips, seasonal_tips = get_rearing_guidance(animal_type, number_of_animals)
    
    return render_template('home.html', 
                           area_size=area_size, 
                           feed_type=feed_type, 
                           feed_quantity=feed_quantity, 
                           cost_in_inr=cost_in_inr, 
                           water_supply=water_supply, 
                           hygiene_tips=hygiene_tips,
                           seasonal_tips=seasonal_tips,
                           animal_type=animal_type, 
                           number_of_animals=number_of_animals)

if __name__ == '__main__':
    app.run(debug=True)
