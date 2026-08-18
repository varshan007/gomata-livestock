const mongoose = require('mongoose');

const otpSchema = new mongoose.Schema({
    identifier: {
        type: String,
        required: true,
        trim: true,
    },
    code: {
        type: String,
        required: true,
    },
    type: {
        type: String,
        enum: ['email', 'phone', 'mobile'],
        required: true,
    },
    createdAt: {
        type: Date,
        default: Date.now,
        index: { expires: 300 } // Auto-delete documents after 5 minutes (300 seconds)
    }
});

module.exports = mongoose.model('OTP', otpSchema);
