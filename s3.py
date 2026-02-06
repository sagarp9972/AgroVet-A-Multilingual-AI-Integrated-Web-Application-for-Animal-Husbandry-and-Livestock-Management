from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
import base64

app = Flask(__name__)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Medication suggestions for specific diseases
medications = {
    "1 LUMPY SKIN": "Tips for Lumpy Skin Disease (LSD):"
                    "Isolation: Immediately isolate affected animals to prevent the spread of the disease to other livestock."
                    "Hygiene: Maintain a clean environment by disinfecting animal housing, feeding equipment, and water troughs regularly."
                    "Supportive Care: Ensure animals are well-fed with a nutritious diet to strengthen their immune systems. Providing fresh, clean water is essential."
                    "Reduce Stress: Keep the animals in a calm and stress-free environment. Stress can worsen disease symptoms."
                    "Monitor for Secondary Infections: Keep an eye on skin lesions and prevent further infection by cleaning any open wounds.",
    "4 RINGWORM": "Management Tips for Ringworm:"
                    "Isolate Infected Animals to prevent spread."
                    "Maintain Cleanliness by changing bedding and disinfecting equipment."
                    "Improve Nutrition with a balanced diet and supplements like zinc and vitamin A."
                    "Use Topical Antifungal Ointments like clotrimazole or sulfur dips."
                    "Sun Exposure can help kill fungi."
                    "Groom Regularly to remove scabs and infected fur."
                    "Reduce Stress by providing a comfortable environment."
                    "Monitor Other Animals for signs and consult a vet if needed.",
    "6 MASTITS": "Tips and management for Mastitis"
                    "Frequent Milking: Regular milking helps prevent milk build-up and reduces the chances of infection."
                    "Proper Hygiene: Clean the udder and teat before and after milking to prevent bacterial entry. Use sanitized towels and gloves."
                    "Dry Cow Therapy: After the lactating period, administer dry cow therapy to prevent mastitis and keep the udder healthy."
                    "Diet Management: Ensure a balanced nutrition with adequate amounts of vitamins and minerals to maintain good udder health."
                    "Massaging: Gently massage the udder to improve circulation and promote milk flow, which can help reduce the risk of mastitis.",
    "3 SHEEP SCABIES": "Tips for Sheep Scabies (For Mouth Area)"
                        "Keep Sheep Clean: Regularly bathe the sheep in mild antiseptic solutions to help remove mites from the skin."
                        "Maintain Bedding: Change bedding frequently to prevent mite infestations. Clean the stalls with a disinfectant to kill any remaining mites."
                        "Improve Nutrition: Provide a balanced diet with minerals and vitamins that help strengthen the immune system and promote skin healing."
                        "Reduce Stress: Avoid overcrowding, and provide adequate space, ventilation, and quiet surroundings to reduce stress, which can worsen scabies."
                        "Skin Care: Apply soothing oils like neem oil or coconut oil to the affected areas, especially around the mouth, to help calm irritation and moisturize the skin."
                        "Monitor and Isolate: Isolate any affected animals to prevent the spread of the disease to healthy sheep.",
    "10 GREASY DISEASE": "Prevention and Management Tips for greasy disease:"
                        "Clean and Dry Conditions: Keep pens and bedding dry to prevent bacterial growth."
                        "Isolation: Separate infected pigs to avoid spreading."
                        "Boost Immunity: Provide a balanced diet with vitamins A, E and omega-3 fatty acids."
                        "Skin Care: Clean affected areas with antiseptic and apply antibacterial ointments."
                        "Herbal Remedies: Use neem oil and turmeric paste for their antibacterial properties."
                        "Hydration: Ensure access to fresh, clean water."
                        "Monitor: Watch for early signs and isolate any affected animals."
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    medication = None
    img_data = None  # Variable to store image data for displaying on the webpage
    if request.method == "POST":
        # Check if image file is provided
        if "file" not in request.files:
            return render_template("s3.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("s3.html", error="No selected file")

        if file:
            # Process the image
            image = Image.open(file).convert("RGB")

            # Resize the image to 224x224 and crop from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

            # Convert image to numpy array and normalize
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Create the array of the right shape to feed into the keras model
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predict the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()  # Strip any leading/trailing whitespace
            confidence_score = prediction[0][index]

            # Debugging: Log predicted class name and confidence score
            print(f"Predicted class: {class_name}")
            print(f"Confidence score: {confidence_score}")

            # Suggest medication if disease detected
            medication = medications.get(class_name, None)

            # Debugging: Log the suggested medication
            if medication:
                print(f"Suggested medication: {medication}")
            else:
                print("No medication suggestion available.")

            # Convert image to base64 for rendering on the webpage
            img_byte_array = io.BytesIO()
            image.save(img_byte_array, format="PNG")
            img_data = base64.b64encode(img_byte_array.getvalue()).decode("utf-8")

            return render_template("s3.html", prediction=class_name, confidence=confidence_score, medication=medication, img_data=img_data)

    return render_template("s3.html", prediction=prediction, confidence=confidence, medication=medication, img_data=img_data)

if __name__ == "__main__":
    app.run(debug=True)
