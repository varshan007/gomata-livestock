const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const staffSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
        trim: true
    },
    email: {
        type: String,
        trim: true,
        lowercase: true,
        sparse: true
    },
    phone: {
        type: String,
        required: true
    },
    userId: {
        type: String,
        required: true,
        unique: true,
        lowercase: true
    },
    password: {
        type: String
    },
    passwordSet: {
        type: Boolean,
        default: false
    },
    position: {
        type: String,
        enum: ['Farm Manager', 'Supervisor', 'Veterinarian', 'Technician', 'Worker', 'Other'],
        default: 'Worker'
    },
    role: {
        type: String,
        enum: ['Admin', 'Manager', 'Operator', 'Viewer'],
        default: 'Viewer'
    },
    assignedFarms: [{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Farm'
    }],
    assignedZones: [{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Zone'
    }],
    assignedDevices: [{
        type: String
    }],
    primaryResponsibility: {
        type: String,
        enum: ['Livestock Monitoring', 'Health Monitoring', 'Device Maintenance', 'Farm Operations', 'Breeding Management', 'Other'],
        default: 'Farm Operations'
    },
    assignedShift: {
        type: String,
        enum: ['Morning', 'Afternoon', 'Night', 'Full Day'],
        default: 'Full Day'
    },
    alertPreferences: [{
        type: String
    }],
    status: {
        type: String,
        enum: ['Active', 'Inactive'],
        default: 'Active'
    },
    adminUserId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
}, {
    timestamps: true
});

// Hash password before saving
staffSchema.pre('save', async function (next) {
    if (!this.isModified('password') || !this.password) {
        return next();
    }
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
});

// Method to compare password
staffSchema.methods.matchPassword = async function (enteredPassword) {
    return await bcrypt.compare(enteredPassword, this.password);
};

module.exports = mongoose.model('Staff', staffSchema);
