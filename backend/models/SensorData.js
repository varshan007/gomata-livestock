const mongoose = require('mongoose');

const sensorDataSchema = new mongoose.Schema({
  livestockId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Livestock',
    required: true
  },
  deviceId: {
    type: String,
    required: true
  },
  temperature: {
    type: Number,
    required: true
  },
  latitude: {
    type: Number,
    required: true
  },
  longitude: {
    type: Number,
    required: true
  },
  batteryLevel: Number,
  timestamp: {
    type: Date,
    default: Date.now
  }
});

// Index for faster queries
sensorDataSchema.index({ livestockId: 1, timestamp: -1 });

module.exports = mongoose.model('SensorData', sensorDataSchema);