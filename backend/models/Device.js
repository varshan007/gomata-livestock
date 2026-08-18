const mongoose = require('mongoose');

const deviceSchema = new mongoose.Schema({
    deviceId: {
        type: String,
        required: true,
        unique: true
    },
    deviceType: {
        type: String,
        enum: ['Neck Collar', 'Ear Tag', 'Leg Band', 'Implantable Chip', 'Other'],
        default: 'Neck Collar'
    },
    status: {
        type: String,
        enum: ['Active', 'Inactive', 'Unassigned'],
        default: 'Unassigned'
    },
    assignedTo: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Livestock',
        default: null
    },
    batteryLevel: {
        type: Number,
        min: 0,
        max: 100,
        default: 100
    },
    signalStrength: {
        type: Number, // dBm
        default: -50
    },
    lastPing: {
        type: Date,
        default: null
    }
}, { timestamps: true });

module.exports = mongoose.model('Device', deviceSchema);
