const mongoose = require('mongoose');

const geofenceSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
        trim: true
    },
    type: {
        type: String,
        enum: ['Safe', 'Exclusion', 'Water', 'Shelter'],
        default: 'Safe'
    },
    shape: {
        type: String,
        enum: ['Polygon', 'Circle'],
        default: 'Polygon'
    },
    // For Polygon: Array of {lat, lng}
    coordinates: [{
        latitude: Number,
        longitude: Number
    }],
    // For Circle: Center {lat, lng} + radius in meters
    center: {
        latitude: Number,
        longitude: Number
    },
    radius: {
        type: Number, // in meters
        default: 0
    },
    livestockId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Livestock',
        default: null // Null means applies to all
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
});

module.exports = mongoose.model('Geofence', geofenceSchema);
