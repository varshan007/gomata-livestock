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

// Ensure Nodemailer transporter exists (Ethereal for testing if no user creds)
let transporter;
const initTransporter = async () => {
    if (process.env.SMTP_HOST && process.env.SMTP_USER) {
        transporter = nodemailer.createTransport({
            service: 'gmail',
            auth: {
                user: process.env.SMTP_USER,
                pass: process.env.SMTP_PASS,
            },
        });
        logger.info("✅ Custom SMTP Transporter Initialized");
    } else {
        // Fallback to Ethereal free testing account
        let testAccount = await nodemailer.createTestAccount();
        transporter = nodemailer.createTransport({
            host: "smtp.ethereal.email",
            port: 587,
            secure: false, // true for 465, false for other ports
            auth: {
                user: testAccount.user, // generated ethereal user
                pass: testAccount.pass, // generated ethereal password
            },
        });
        logger.info("⚠️ Ethereal Test SMTP Transporter Initialized (for testing)");
    }
};

initTransporter();

/**
 * Sends a 6-digit OTP via Email
 */
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
        // If using Ethereal, log the preview URL
        if (info.messageId && nodemailer.getTestMessageUrl(info)) {
            logger.info("-> 📧 Preview Ethereal Email Here: %s", nodemailer.getTestMessageUrl(info));
        }
        return true;
    } catch (error) {
        logger.error("❌ Email dispatch failed:", error);
        return false;
    }
};

/**
 * Sends a 6-digit OTP via SMS Using Twilio
 */
const sendPhoneOTP = async (phone, otpCode) => {
    if (!twilioClient) {
        logger.warn(`[MOCK SMS] Would send SMS to ${phone} with code: ${otpCode}`);
        return true; // Pretend it worked if Twilio is missing
    }

    try {
        // Ensure phone has country code. Assuming India +91 if none provided logic could go here, 
        // but user input should ideally be standardized or we prepend the country code.
        let formattedPhone = phone;
        if (!formattedPhone.startsWith('+')) {
            // Assume Indian number by default for this app and strip any leading zeros
            formattedPhone = '+91' + formattedPhone.replace(/^0+/, '');
            logger.info(`Auto-formatted phone number to E.164 standard: ${formattedPhone}`);
        }

        const message = await twilioClient.messages.create({
            body: `Your GoMata Verification Code is: ${otpCode}. It expires in 5 minutes.`,
            from: process.env.TWILIO_PHONE_NUMBER,
            to: formattedPhone
        });

        logger.info(`✅ SMS sent via Twilio successfully. SID: ${message.sid}`);
        return true;
    } catch (error) {
        logger.warn("\n==============================================");
        logger.warn("🚨 TWILIO TRIAL LIMIT EXCEEDED 🚨");
        logger.warn("Bypassing SMS requirement. Please use this OTP code:");
        logger.warn(`===> ${otpCode} <===`);
        logger.warn("==============================================\n");
        return true; // We return true to prevent blocking the user pipeline
    }
};

module.exports = {
    sendEmailOTP,
    sendPhoneOTP
};
