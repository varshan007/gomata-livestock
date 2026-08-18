/**
 * GoMata Digital Twin Simulator v2 — Module Index
 * 
 * Usage:
 *   const { DigitalTwinSimulator, DatasetExporter } = require('./services/digitalTwin');
 */

module.exports = {
    DigitalTwinSimulator: require('./DigitalTwinSimulator'),
    CowPhysiologyEngine: require('./CowPhysiologyEngine'),
    EnvironmentModel: require('./EnvironmentModel'),
    SensorGenerator: require('./SensorGenerator'),
    EpisodeScheduler: require('./EpisodeScheduler'),
    HerdTransmissionModel: require('./HerdTransmissionModel'),
    InterventionEngine: require('./InterventionEngine'),
    DatasetExporter: require('./DatasetExporter')
};
