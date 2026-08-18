const logger = require('../../../utils/logger');
const { Queue, Worker } = require('bullmq');
const Alert = require('../../../models/Alert');
const User = require('../../../models/User');

class StaffAssignmentAgent {
    constructor(bus) {
        this.bus = bus;
        const connection = { host: '127.0.0.1', port: 6379 };

        // Setup BullMQ Queue for sending escalations
        this.escalationQueue = new Queue('AlertEscalationQueue', { connection });

        // Setup BullMQ Worker to process escalations after 10 minutes
        this.worker = new Worker('AlertEscalationQueue', async job => {
            const { alertId } = job.data;
            await this.handleEscalation(alertId);
        }, { connection });

        this.worker.on('failed', (job, err) => {
            logger.error(`[StaffAssignmentAgent] ❌ Escalation job failed for Alert ${job.data.alertId}:`, err);
        });
    }

    start() {
        logger.info('[StaffAssignmentAgent] Started. Listening for critical alerts to assign to staff...');

        this.bus.on('alert:saved', async (payload) => {
            const alertData = payload.data;
            const { alertId, severity } = alertData;

            // Only assign High or Critical alerts
            if (severity !== 'High' && severity !== 'Critical') {
                return;
            }

            try {
                // Find active Vets or Staff. 
                // For MVP, randomly assign to any user with role 'Vet' or 'Staff'.
                // If none exist, fallback to 'Admin'.
                let candidates = await User.find({ role: { $in: ['Vet', 'Staff'] } });

                if (candidates.length === 0) {
                    candidates = await User.find({ role: 'Admin' });
                }

                if (candidates.length === 0) {
                    logger.info(`[StaffAssignmentAgent] No staff available to assign Alert ${alertId}`);
                    return;
                }

                // Pick a random staff member
                const assignedUser = candidates[Math.floor(Math.random() * candidates.length)];

                // Update Alert in DB
                const updatedAlert = await Alert.findByIdAndUpdate(
                    alertId,
                    {
                        assignedTo: assignedUser._id,
                        status: 'Assigned'
                    },
                    { new: true }
                );

                logger.info(`[StaffAssignmentAgent] 👩‍⚕️ Alert ${alertId} assigned to ${assignedUser.name} (${assignedUser.role}). Starting 10m escalation timer.`);

                // Add to BullMQ with a 10-minute delay
                // Note: using 60 seconds delay for easier testing right now instead of 10 min
                await this.escalationQueue.add(
                    'escalate-task',
                    { alertId: alertId.toString(), previousAssignee: assignedUser._id.toString() },
                    { delay: 60000 } // 1 minute delay for realistic testing without waiting 10 minutes
                );

            } catch (error) {
                logger.error('[StaffAssignmentAgent] Error assigning staff:', error);
            }
        });
    }

    async handleEscalation(alertId) {
        try {
            const alert = await Alert.findById(alertId);

            if (!alert) return;

            // Check if it's already Acknowledged or Resolved
            if (alert.status === 'Acknowledged' || alert.status === 'Resolved' || alert.resolved === true) {
                logger.info(`[StaffAssignmentAgent] ⏱️ Timer finished for Alert ${alertId}. It was already acknowledged. No escalation needed.`);
                return;
            }

            logger.info(`[StaffAssignmentAgent] 🚨 Alert ${alertId} was NOT acknowledged within the SLA! Escalating to Admin...`);

            // Find an Admin to escalate to
            const admins = await User.find({ role: 'Admin' });
            let newAssignee = null;
            if (admins.length > 0) {
                newAssignee = admins.find(a => a._id.toString() !== alert.assignedTo?.toString()) || admins[0];
            }

            // Update Alert DB status
            alert.status = 'Escalated';
            if (newAssignee) {
                alert.assignedTo = newAssignee._id;
            }
            await alert.save();

            logger.info(`[StaffAssignmentAgent] 🛡️ Alert ${alertId} escalated to Admin ${newAssignee ? newAssignee.name : 'Unknown'}.`);

        } catch (error) {
            logger.error('[StaffAssignmentAgent] Error handling escalation:', error);
        }
    }
}

module.exports = StaffAssignmentAgent;
