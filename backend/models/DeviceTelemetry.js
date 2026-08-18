const mongoose = require('mongoose');

const deviceTelemetrySchema = new mongoose.Schema({
    deviceId: {
        type: String,
        required: true,
        index: true
    },
    animalId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'LivestockMaster',
        required: false
    },
    tenantId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: false
    },
    temperature: {
        type: Number, // Celsius
        required: true
    },
    heartRate: {
        type: Number, // bpm
        required: true
    },
    activity: {
        type: Number, // 0-100 continuous score
        default: 75
    },
    battery: {
        type: Number, // percentage
        required: true
    },
    signalStrength: {
        type: Number, // dBm
        required: true
    },
    deviceStatus: {
        type: String,
        enum: ['Active', 'Offline', 'Low Battery'],
        default: 'Active'
    },
    location: {
        type: { type: String, enum: ['Point'], default: 'Point' },
        coordinates: {
            type: [Number], // [longitude, latitude]
            required: true
        }
    },
    timestamp: {
        type: Date,
        default: Date.now,
        index: true
    }
});

// For geospatial queries
deviceTelemetrySchema.index({ location: '2dsphere' });

module.exports = mongoose.model('DeviceTelemetry', deviceTelemetrySchema);
