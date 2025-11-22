Title: Predictive Maintenance with Explainable AI for Critical Vehicle Systems
The Challenge:
Fleet operations and personal vehicle ownership suffer significant financial loss and risk due to unscheduled breakdowns of critical components, such as the engine's oil and cooling systems. Traditional maintenance relies on fixed schedules or reacting to dashboard warning lights, leading to either costly over-maintenance or catastrophic, unexpected failures.

While modern anomaly detection models (like Isolation Forest ) can flag component issues in real-time, maintenance technicians often distrust "black box" AI warnings, delaying essential repairs and negating the predictive benefit.

Problem Statement:
Develop an innovative, full-stack Predictive Maintenance (PdM) solution that continuously monitors simulated real-time vehicle data, accurately predicts component degradation (specifically, an anomaly in the Engine Oil System using RPM and Temperature data ), and, critically, provides a transparent, human-readable explanation for the warning to ensure rapid, justified maintenance action.

Core Objectives (MVP):
The solution must deliver the following four functional components:


Data Ingestion Pipeline: Create a system to simulate and ingest real-time time-series data streams (Engine RPM, Oil Temperature) into a local persistence layer (e.g., SQLite/CSV).


Anomaly Detection Engine: Implement an unsupervised Machine Learning model (e.g., Isolation Forest ) trained on "normal" operational data to generate a real-time Anomaly Score for the latest sensor readings.

Explainable AI (XAI) Layer: Integrate a mechanism (or a simulated module output) that, upon detecting an anomaly, attributes the prediction to the contributing sensor factors. For example, explain that the prediction is high because "Oil Temperature has deviated from the expected RPM-to-Temperature ratio by +5%."


Actionable Dashboard: Build a real-time web dashboard that clearly visualizes the sensor data, displays a prominent red ALERT banner upon anomaly detection, and immediately presents the clear XAI explanation and a maintenance recommendation.
