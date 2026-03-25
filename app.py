# app.py
import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def predict_diabetes(pregnancies, glucose, blood_pressure, bmi, age):
    payload = {
        "Pregnancies": int(pregnancies),
        "Glucose": float(glucose),
        "BloodPressure": float(blood_pressure),
        "BMI": float(bmi),
        "Age": int(age)
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        if result["diabetic"]:
            return (
                " High Risk: Diabetic",
                "The model predicts this person is likely **diabetic**.\n\n"
                "> ℹ️ This is a machine learning prediction, not a medical diagnosis. "
                "Please consult a healthcare professional."
            )
        else:
            return (
                " Low Risk: Not Diabetic",
                "The model predicts this person is likely **not diabetic**.\n\n"
                "> ℹ️ This is a machine learning prediction, not a medical diagnosis. "
                "Please consult a healthcare professional."
            )

    except requests.exceptions.ConnectionError:
        return (
            "Connection Error",
            "Could not connect to the prediction API. Make sure the FastAPI server is running."
        )
    except Exception as e:
        return ("Error", f"Something went wrong: {str(e)}")


# ---------- UI Layout ----------

with gr.Blocks(
    title="Diabetes Risk Predictor",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"]
    ),
    css="""
        .gradio-container { max-width: 780px !important; margin: auto; }
        .result-box textarea { font-size: 1.4rem !important; font-weight: 700; text-align: center; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown(
        """
        # 🩺 Diabetes Risk Predictor
        ### Powered by Random Forest · MLOps Demo Project
        Enter the patient's health metrics below and click **Predict** to assess diabetes risk.
        ---
        """
    )

    with gr.Row():
        with gr.Column():
            pregnancies = gr.Slider(
                minimum=0, maximum=17, step=1, value=2,
                label="Pregnancies",
                info="Number of times pregnant"
            )
            glucose = gr.Slider(
                minimum=0, maximum=200, step=1, value=120,
                label="Glucose Level (mg/dL)",
                info="Plasma glucose concentration"
            )
            blood_pressure = gr.Slider(
                minimum=0, maximum=130, step=1, value=70,
                label="Blood Pressure (mm Hg)",
                info="Diastolic blood pressure"
            )

        with gr.Column():
            bmi = gr.Slider(
                minimum=10.0, maximum=70.0, step=0.1, value=28.5,
                label="BMI",
                info="Body Mass Index"
            )
            age = gr.Slider(
                minimum=18, maximum=100, step=1, value=33,
                label="Age (years)"
            )

            gr.Markdown("<br>")
            predict_btn = gr.Button("🔍 Predict", variant="primary", size="lg")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            result_label = gr.Textbox(
                label="Prediction Result",
                interactive=False,
                elem_classes=["result-box"]
            )
        with gr.Column():
            result_detail = gr.Markdown(label="Details")

    predict_btn.click(
        fn=predict_diabetes,
        inputs=[pregnancies, glucose, blood_pressure, bmi, age],
        outputs=[result_label, result_detail]
    )

    gr.Markdown(
        """
        ---
        <center>
        Built with FastAPI · Scikit-learn · Gradio &nbsp;|&nbsp; 
        <a href="https://github.com/sufyanazar786/predict-health-random-forest-kube" target="_blank">GitHub</a>
        </center>
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
