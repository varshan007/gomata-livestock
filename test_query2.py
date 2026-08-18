import sys
from pymongo import MongoClient
client = MongoClient("mongodb://127.0.0.1:27017/livestock_monitoring")
db = client["livestock_monitoring"]

event = db["trainingevents"].find_one({"source": {"$in": ["simulation", "simulation_v2"]}})
animal_id = event["animalId"]

# Print the most recent telemetry regardless of timestamp
tel = list(db.devicetelemetries.find({"animalId": animal_id}).sort("timestamp", -1).limit(100))
print(f"Total telemetry records for this animal: {len(tel)}")
if len(tel) > 0:
    print(f"Event Timestamp: {event.get('createdAt')}")
    print(f"telemetry 0 timestamp: val={tel[0].get('timestamp')}")
    print(f"telemetry -1 timestamp: val={tel[-1].get('timestamp')}")
else:
    print("no tel found")
