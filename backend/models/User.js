const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const userSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
        trim: true
    },
    email: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        lowercase: true
    },
    password: {
        type: String,
        required: true
    },
    phone: {
        type: String,
        default: ''
    },
    role: {
        type: String,
        enum: ['Admin', 'Vet', 'Staff', 'User'],
        default: 'Admin'
    },
    dob: {
        type: Date
    },
    farm: {
        name: { type: String, default: '' },
        livestockType: { type: String, default: '' },
        livestockCount: { type: Number, default: 0 },
        hardwareId: { type: String, default: '' },
        geofences: [{
            latitude: Number,
            longitude: Number,
            radius: Number
        }],
        location: {
            address: { type: String, default: '' },
            city: { type: String, default: '' },
            state: { type: String, default: '' },
            pinCode: { type: String, default: '' },
            country: { type: String, default: '' },
            coordinates: {
                latitude: Number,
                longitude: Number
            }
        },
        size: { type: Number }, // in acres
        type: { type: String, enum: ['Dairy Farm', 'Cattle Ranch', 'Mixed Livestock', 'Other', ''], default: '' }
    },
    settings: {
        theme: { type: String, enum: ['light', 'dark', 'auto'], default: 'light' },
        units: {
            temperature: { type: String, enum: ['C', 'F'], default: 'C' },
            distance: { type: String, enum: ['km', 'miles'], default: 'km' }
        },
        notifications: {
            email: { type: Boolean, default: true },
            sms: { type: Boolean, default: false },
            alerts: { type: Boolean, default: true }
        }
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
}, {
    timestamps: true
});

// Hash password before saving
userSchema.pre('save', async function (next) {
    if (!this.isModified('password')) {
        return next();
    }
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
});

// Method to compare password
userSchema.methods.matchPassword = async function (enteredPassword) {
    return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model('User', userSchema);
