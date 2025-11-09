const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const { spawn } = require("child_process");

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.post("/predict", (req, res) => {
  const {
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age,
  } = req.body;

  const python = spawn("python", [
    "model_runner.py",
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age,
  ], { cwd: __dirname }); // ensure correct path

  let result = "";

  python.stdout.on("data", (data) => {
    result += data.toString();
  });

  python.stderr.on("data", (data) => {
    console.error(`Error: ${data}`);
  });

  python.on("close", () => {
    res.json({ prediction: result.trim() });
  });
});

app.listen(5000, () => console.log("✅ Server running on http://127.0.0.1:5000"));
