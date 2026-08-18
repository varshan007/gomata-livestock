import sys
from pymongo import MongoClient
client = MongoClient("mongodb://127.0.0.1:27017/livestock_monitoring")
db = client['livestock_monitoring']
event = db.trainingevents.find_one({"source": "simulation_v2"})
if not event:
    print("No events found")
    sys.exit(0)
animal_id = event.get("animalId")
print(f"Event Animal ID: {animal_id} type: {type(animal_id)}")

telemetry = db.devicetelemetries.find_one({"animalId": animal_id})
print(f"Telemetry using ObjectId: {telemetry is not None}")
telemetry_str = db.devicetelemetries.find_one({"animalId": str(animal_id)})
print(f"Telemetry using String: {telemetry_str is not None}")

# Let's see what a devicetelemetries from simulation_v2 looks like
sim_tel = db.devicetelemetries.find_one({"source": "simulation_v2"})
if sim_tel:
    print(f"Sample sim_tel animalId: {sim_tel.get('animalId')} type: {type(sim_tel.get('animalId'))}")
else:
    print("No sim_tel found")

