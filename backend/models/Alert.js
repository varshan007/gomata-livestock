const mongoose = require('mongoose');

const alertSchema = new mongoose.Schema({
  livestockId: {
    type: mongoose.Schema.Types.ObjectId,
    required: true
  },
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  alertType: {
    type: String,
    enum: ['Temperature', 'Geofence', 'Inactivity', 'Battery', 'Health'],
    required: true
  },
  severity: {
    type: String,
    enum: ['Low', 'Medium', 'High', 'Critical'],
    required: true
  },
  message: String,

  // ── Rich metadata (denormalized for self-contained alerts) ──
  animalName: { type: String, default: '' },
  farmName: { type: String, default: '' },
  zoneName: { type: String, default: '' },
  breed: { type: String, default: '' },
  deviceId: { type: String, default: '' },
  diseaseProbability: { type: Number, default: null },
  alertSource: {
    type: String,
    enum: ['ml_health_agent', 'simulation', 'manual', 'system'],
    default: 'system'
  },
  explanation: { type: String, default: '' },

  resolved: {
    type: Boolean,
    default: false
  },
  assignedTo: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    default: null
  },
  status: {
    type: String,
    enum: ['Pending', 'Assigned', 'Acknowledged', 'Escalated', 'Resolved'],
    default: 'Pending'
  },
  acknowledgedAt: {
    type: Date,
    default: null
  },
  timestamp: {
    type: Date,
    default: Date.now
  }
});

module.exports = mongoose.model('Alert', alertSchema);