import React, { useState } from "react";
import "./App.css";
import { FaStethoscope } from "react-icons/fa"; // 👈 for header icon

function App() {
  const [formData, setFormData] = useState({
    Pregnancies: "",
    Glucose: "",
    BloodPressure: "",
    SkinThickness: "",
    Insulin: "",
    BMI: "",
    DiabetesPedigreeFunction: "",
    Age: "",
  });
  const [result, setResult] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await res.json();
      setResult(data.prediction);
    } catch (err) {
      console.error("Error:", err);
    }
  };

  return (
    <div className="container">
      <div className="form-card">
        <div className="title">
          <FaStethoscope color="#1565c0" />
          <span>Diabetes Prediction</span>
        </div>

        <form onSubmit={handleSubmit}>
          {Object.keys(formData).map((key) => (
            <div key={key}>
              <label>{key}</label>
              <input
                type="number"
                name={key}
                value={formData[key]}
                onChange={handleChange}
                required
              />
            </div>
          ))}
          <button type="submit">Predict</button>
        </form>

        {result && (
          <div
            className={`result ${
              result.includes("Non-Diabetic") ? "positive" : "negative"
            }`}
          >
            {result}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
