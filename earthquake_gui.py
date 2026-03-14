import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle


# Load the trained model
try:
    with open("earthquake_damage_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    messagebox.showerror("Error", f"Could not load model: {e}")
    raise SystemExit


# Create GUI window
root = tk.Tk()
root.title(" Earthquake Damage Prediction")
root.geometry("750x900")
root.configure(bg="#eef2f7")  

# Scrollable canvas setup
canvas = tk.Canvas(root, bg="#eef2f7", highlightthickness=0)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#eef2f7")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")


# Feature definitions
fields = {
    'district_id_x': tk.IntVar(),
    'vdcmun_id_x': tk.IntVar(),
    'ward_id': tk.IntVar(),
    'count_floors_pre_eq': tk.IntVar(),
    'count_floors_post_eq': tk.IntVar(),
    'age_building': tk.IntVar(),
    'plinth_area_sq_ft': tk.DoubleVar(),
    'height_ft_pre_eq': tk.DoubleVar(),
    'height_ft_post_eq': tk.DoubleVar(),
    'land_surface_condition': tk.StringVar(),
    'foundation_type': tk.StringVar(),
    'roof_type': tk.StringVar(),
    'ground_floor_type': tk.StringVar(),
    'other_floor_type': tk.IntVar(),
    'position': tk.StringVar(),
    'plan_configuration': tk.StringVar(),
    'has_superstructure_adobe_mud': tk.IntVar(),
    'has_superstructure_mud_mortar_stone': tk.IntVar(),
    'has_superstructure_stone_flag': tk.IntVar(),
    'has_superstructure_cement_mortar_stone': tk.IntVar(),
    'has_superstructure_mud_mortar_brick': tk.IntVar(),
    'has_superstructure_cement_mortar_brick': tk.IntVar(),
    'has_superstructure_timber': tk.IntVar(),
    'has_superstructure_bamboo': tk.IntVar(),
    'has_superstructure_rc_non_engineered': tk.IntVar(),
    'has_superstructure_rc_engineered': tk.IntVar(),
    'has_superstructure_other': tk.IntVar(),
    'condition_post_eq': tk.IntVar(),
    'technical_solution_proposed': tk.IntVar(),
    'vdcmun_id_y': tk.IntVar(),
    'vdcmun_name': tk.IntVar(),
    'district_id_y': tk.IntVar(),
    'district_name': tk.IntVar(),
    'pred_intensity': tk.DoubleVar(),
    'pred_intensity_mun': tk.DoubleVar(),
    'intensity_ratio': tk.DoubleVar(),
    'avg_intensity': tk.DoubleVar(),
    'ward_damage_mean': tk.DoubleVar()
}

# Dropdown options for categorical columns
options = {
    "land_surface_condition": ["0 - Flat", "1 - Moderate slope", "2 - Steep slope"],
    "foundation_type": ["0 - Rubble", "1 - Mud mortar", "2 - Cement mortar", "3 - RCC", "4 - Other"],
    "roof_type": ["0 - Thatched", "1 - Reinforced concrete", "2 - Metal/CGI"],
    "ground_floor_type": ["0 - Mud", "1 - Cement", "2 - Timber", "3 - Tile", "4 - Other"],
    "position": ["0 - Detached", "1 - Semi-detached", "2 - Adjacent", "3 - Corner"],
    "plan_configuration": [str(i) for i in range(10)]
}

# Header
header = tk.Label(scrollable_frame, text=" Earthquake Damage Prediction Tool", 
                  font=("Helvetica", 18, "bold"), bg="#eef2f7", fg="#1f2a38")
header.grid(row=0, column=0, columnspan=2, pady=15)

separator = ttk.Separator(scrollable_frame, orient='horizontal')
separator.grid(row=1, column=0, columnspan=2, sticky="ew", pady=10)


# Layout all inputs with styled labels

row = 2
for key in fields:
    is_checkbox = "has_superstructure" in key or key in ["condition_post_eq", "technical_solution_proposed"]
    
    label = tk.Label(scrollable_frame, text=key.replace("_", " ").title(), 
                     bg="#eef2f7", anchor="w", font=("Helvetica", 11))
    label.grid(row=row, column=0, sticky="w", padx=15, pady=4)
    
    if key in options:
        cb = ttk.Combobox(scrollable_frame, textvariable=fields[key], values=options[key], state="readonly", width=25)
        cb.grid(row=row, column=1, padx=15, pady=4)
        cb.current(0)
    elif is_checkbox:
        chk = tk.Checkbutton(scrollable_frame, variable=fields[key], bg="#eef2f7")
        chk.grid(row=row, column=1, padx=15, pady=4)
    else:
        entry = tk.Entry(scrollable_frame, textvariable=fields[key], font=("Helvetica", 10))
        entry.grid(row=row, column=1, padx=15, pady=4)
    
    row += 1
model_columns = list(fields.keys())

def predict_damage():
    try:
        X_input = []
        for col in model_columns:
            val = fields[col].get()
            if col in options:
                val = int(val.split()[0])
            X_input.append(float(val))
        X_input = np.array(X_input).reshape(1, -1)
        pred = model.predict(X_input)
        pred_label = int(np.argmax(pred, axis=1)[0]) if pred.ndim > 1 else int(pred[0])
        inference = {
            0: "No damage",
            1: "Minor damage",
            2: "Moderate damage",
            3: "Extensive damage",
            4: "Complete damage"
        }
        msg = inference.get(pred_label, "Unknown damage grade")
        messagebox.showinfo("Prediction Result", f"Predicted Damage Grade: {pred_label}\nInference: {msg}")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")


predict_btn = tk.Button(scrollable_frame, text="Predict Damage Grade", command=predict_damage, 
                        bg="#1f78b4", fg="white", font=("Helvetica", 12, "bold"), padx=25, pady=8, relief="raised", bd=3)
predict_btn.grid(row=row, column=0, columnspan=2, pady=20)

footer = tk.Label(scrollable_frame, text="© 2025 VIT VELLORE AI Earthquake Damage Grade Predictor Project", 
                  font=("Helvetica", 9), bg="#eef2f7", fg="#555")
footer.grid(row=row+1, column=0, columnspan=2, pady=10)

root.mainloop()
