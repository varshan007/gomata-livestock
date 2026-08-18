from pymongo import MongoClient
client = MongoClient("mongodb://127.0.0.1:27017/livestock_monitoring")
col = client["livestock_monitoring"]["trainingevents"]

anomalies = col.count_documents({"source": "simulation_v3", "label": 1})
normals = col.count_documents({"source": "simulation_v3", "label": 0})
total = anomalies + normals
print(f"Total v3 Anomalies: {anomalies}")
print(f"Total v3 Normals: {normals}")
print(f"Total v3 Events: {total}")

doc = col.find_one({"source": "simulation_v3", "label": 1})
if doc:
    feats = doc.get("features", {})
    print(f"Sample anomaly has {len(feats)} features")
    for k, v in feats.items():
        print(f"  {k}: {v}")
else:
    print("No anomaly doc found")
