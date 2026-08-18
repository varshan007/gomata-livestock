const mongoose = require('mongoose');

const livestockSchema = new mongoose.Schema({
  tagNumber: {
    type: String,
    required: true,
    unique: true
  },
  name: {
    type: String,
    required: true
  },
  breed: String,
  type: String,
  about: String,
  age: Number,
  weight: Number,
  gender: {
    type: String,
    enum: ['Male', 'Female']
  },
  farmId: { type: mongoose.Schema.Types.ObjectId, ref: 'Farm', required: true },
  zoneId: { type: mongoose.Schema.Types.ObjectId, ref: 'Zone', required: true },
  deviceId: {
    type: String,
    required: true,
    unique: true
  },
  deviceType: {
    type: String,
    enum: ['Neck Collar', 'Ear Tag', 'Leg Band', 'Implantable Chip', 'Other'],
    default: 'Other'
  },
  vaccinationNotes: String,
  breedingNotes: String,
  additionalNotes: String,
  photoUrl: String,
  status: {
    type: String,
    enum: ['Active', 'Inactive', 'Medical'],
    default: 'Active'
  }
}, { timestamps: true });

module.exports = mongoose.model('Livestock', livestockSchema);