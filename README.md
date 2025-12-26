# vibrasafe-swarm-sim
Title: VibraSafe Swarm – Software-Only Factory Safety Digital Twin

Problem: Deaf / distracted workers alarms miss → accidents; current systems single-machine, no swarm view.
​

Solution: Streamlit dashboard + IsolationForest ML → simulate 6 vests, 3 hazard levels, heatmap + anomaly trend.

How to run:

pip install -r requirements.txt

streamlit run app.py --server.port 8502

Future work: plug ESP32 + IMU, MSME pilot in TN factory
