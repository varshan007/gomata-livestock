const mongoose = require('mongoose');

mongoose.connect('mongodb://127.0.0.1:27017/livestock_monitoring')
    .then(async () => {
        try {
            const db = mongoose.connection.db;

            const user31 = await db.collection('users').findOne({ email: 'varshananand31@gmail.com' });
            const user15 = await db.collection('users').findOne({ email: 'varshananand15@gmail.com' });

            if (user31) {
                const res1 = await db.collection('livestockmasters').updateMany(
                    { farm_id: 'FM-NaN', userId: { $exists: false } },
                    { $set: { userId: user31._id } }
                );
                console.log(`Mapped ${res1.modifiedCount} FM-NaN livestock to ${user31.email}`);
            }

            if (user15) {
                const res2 = await db.collection('livestockmasters').updateMany(
                    { farm_id: 'FM-001', userId: { $exists: false } },
                    { $set: { userId: user15._id } }
                );
                console.log(`Mapped ${res2.modifiedCount} FM-001 livestock to ${user15.email}`);
            }

        } catch (err) {
            console.error(err);
        } finally {
            process.exit(0);
        }
    });
