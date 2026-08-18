const mongoose = require('mongoose');

const farmSchema = new mongoose.Schema({
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    name: { type: String, required: true },
    locationType: { type: String, enum: ['Polygon Mapping', 'Circular Mapping'], required: true },
    geofence: {
        type: {
            type: String,
            enum: ['Polygon', 'Point'], // Mongoose GeoJSON requires Point for Circles
            required: true
        },
        coordinates: {
            type: [], // Array of Arrays for Polygons, Array of 2 for Points (lng, lat)
            required: true
        },
        radius: { type: Number, default: null } // Only applicable if 'Point' (Circular Mapping)
    }
}, { timestamps: true });

// Create a geospatial index on the geofence field for efficient mapping
farmSchema.index({ geofence: '2dsphere' });

module.exports = mongoose.model('Farm', farmSchema);
