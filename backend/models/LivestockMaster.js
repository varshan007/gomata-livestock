const mongoose = require('mongoose');

const livestockMasterSchema = new mongoose.Schema({
    // Identifiers
    livestock_id: { type: String, required: true, unique: true }, // Format: LS-FARM-ZONE-SEQ
    name: { type: String, required: true },
    species: { type: String },
    breed: { type: String },
    age: { type: Number },
    weight: { type: Number },

    // Ownership Binding for Multi-Tenant Isolation
    userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },

    // Location Binding
    farm_id: { type: String, required: true }, // Format: FM-SEQ
    farm_name: { type: String },
    farm_geofence: { type: mongoose.Schema.Types.Mixed }, // GeoJSON Polygon/Circle

    zone_id: { type: String, required: true }, // Format: ZN-FARM-SEQ
    zone_name: { type: String },
    zone_geofence: { type: mongoose.Schema.Types.Mixed }, // GeoJSON Polygon/Circle

    // Hardware Binding
    device_id: { type: String, required: true }, // e.g. GM-SN-1001
    device_type: { type: String }, // Collar, Ear Tag, etc.
    mapping_id: { type: String, required: true }, // Format: MAP-{DeviceID}-{LivestockID}

    // Vet & Custom Notes
    vaccination_notes: { type: String },
    breeding_notes: { type: String },
    additional_notes: { type: String },

    // Real-Time Derived Aggregation (Updated from DeviceTelemetry)
    temperature: { type: Number, default: null },
    heart_rate: { type: Number, default: null },
    battery: { type: Number, default: null },
    signal_strength: { type: Number, default: null },
    device_status: { type: String, default: 'Offline' },
    health_status: { type: String, default: 'Normal' },
    last_location: {
        type: { type: String, enum: ['Point'] },
        coordinates: { type: [Number] } // [lng, lat]
    },
    last_updated: { type: Date, default: null }

}, { timestamps: true });

// Ensure it can be spatially queried
livestockMasterSchema.index({ last_location: '2dsphere' });

module.exports = mongoose.model('LivestockMaster', livestockMasterSchema);
