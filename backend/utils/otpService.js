const logger = require('./logger');
const twilio = require('twilio');
const nodemailer = require('nodemailer');

// Initialize Twilio
let twilioClient;
if (process.env.TWILIO_ACCOUNT_SID && process.env.TWILIO_AUTH_TOKEN) {
    try {
        twilioClient = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
        logger.info("✅ Twilio Client Initialized");
    } catch (error) {
        logger.error("❌ Failed to initialize Twilio:", error);
    }
} else {
    logger.warn("⚠️ Twilio credentials missing in .env. SMS dispatch will only be mocked.");
}

// Ensure Nodemailer transporter exists
let transporter;
const initTransporter = async () => {
    if (process.env.SMTP_USER && process.env.SMTP_PASS) {
        transporter = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                user: process.env.SMTP_USER,
                pass: process.env.SMTP_PASS,
            },
        });
        logger.info("✅ Custom SMTP Transporter Initialized");
    } else {
        logger.warn("⚠️ Custom SMTP credentials missing. Email dispatch will be disabled or mocked.");
    }
};

initTransporter();

const sendEmailOTP = async (email, otpCode) => {
    if (!transporter) {
        logger.error("Transporter not initialized yet.");
        return false;
    }

    try {
        let info = await transporter.sendMail({
            from: `"GoMata Verification" <${process.env.SMTP_USER}>`,
            to: email,
            subject: "Your GoMata Verification Code",
            text: `Welcome to GoMata! Your verification code is: ${otpCode}. It will expire in 5 minutes.`,
            html: `
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 12px;">
                    <h2 style="color: #10b981;">GoMata Verification</h2>
                    <p style="font-size: 16px; color: #333;">Welcome to the GoMata Farm Application.</p>
                    <p style="font-size: 16px; color: #333;">Your verification code is:</p>
                    <div style="background: #f1f5f9; padding: 15px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; letter-spacing: 5px; color: #0f172a; margin: 20px 0;">
                        ${otpCode}
                    </div>
                    <p style="font-size: 14px; color: #64748b;">This code is valid for 5 minutes. Please do not share it with anyone.</p>
                </div>
            `,
        });

        logger.info("✅ Email sent: %s", info.messageId);
        return true;
    } catch (error) {
        logger.error("❌ Email dispatch failed:", error);
        throw error;
    }
};

const sendPhoneOTP = async (phone) => {
    if (!twilioClient || !process.env.TWILIO_VERIFY_SERVICE_SID) {
        logger.warn(`[MOCK SMS] Twilio not configured. Would send SMS to ${phone}`);
        return true;
    }

    try {
        let formattedPhone = phone;
        if (!formattedPhone.startsWith('+')) {
            formattedPhone = '+91' + formattedPhone.replace(/^0+/, '');
            logger.info(`Auto-formatted phone number to E.164 standard: ${formattedPhone}`);
        }

        const verification = await twilioClient.verify.v2
            .services(process.env.TWILIO_VERIFY_SERVICE_SID)
            .verifications.create({ to: formattedPhone, channel: 'sms' });

        logger.info(`✅ SMS sent via Twilio Verify successfully. Status: ${verification.status}`);
        return true;
    } catch (error) {
        logger.error("❌ Twilio Verify dispatch failed:", error);
        throw error;
    }
};

const verifyPhoneOTP = async (phone, code) => {
    if (!twilioClient || !process.env.TWILIO_VERIFY_SERVICE_SID) {
        logger.warn(`[MOCK SMS] Bypassing Twilio verify for ${phone}`);
        return true;
    }

    try {
        let formattedPhone = phone;
        if (!formattedPhone.startsWith('+')) {
            formattedPhone = '+91' + formattedPhone.replace(/^0+/, '');
        }

        const verificationCheck = await twilioClient.verify.v2
            .services(process.env.TWILIO_VERIFY_SERVICE_SID)
            .verificationChecks.create({ to: formattedPhone, code });

        if (verificationCheck.status === 'approved') {
            logger.info(`✅ SMS OTP verified via Twilio!`);
            return true;
        } else {
            logger.warn(`❌ Twilio Verify rejected OTP. Status: ${verificationCheck.status}`);
            return false;
        }
    } catch (error) {
        logger.error("❌ Twilio Verify check failed:", error);
        return false;
    }
};

module.exports = {
    sendEmailOTP,
    sendPhoneOTP,
    verifyPhoneOTP,
    twilioClient
};
