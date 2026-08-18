const mongoose = require('mongoose');

const zoneSchema = new mongoose.Schema({
    farmId: { type: mongoose.Schema.Types.ObjectId, ref: 'Farm', required: true },
    name: { type: String, required: true },
    locationType: { type: String, enum: ['Polygon Mapping', 'Circular Mapping'], required: true },
    geofence: {
        type: {
            type: String,
            enum: ['Polygon', 'Point'],
            required: true
        },
        coordinates: {
            type: [],
            required: true
        },
        radius: { type: Number, default: null }
    }
}, { timestamps: true });

zoneSchema.index({ geofence: '2dsphere' });

module.exports = mongoose.model('Zone', zoneSchema);
