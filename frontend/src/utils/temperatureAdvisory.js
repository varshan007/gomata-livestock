export const getTemperatureStatus = (temperature) => {
    if (!temperature) return { status: 'Unknown', color: 'gray', advisory: 'No data available' };

    if (temperature >= 32 && temperature <= 38.5) {
        return {
            status: 'Normal',
            color: 'green',
            advisory: 'Animal health is normal. Temperature within safe range.',
            action: 'Continue regular monitoring.',
        };
    } else if (temperature > 38.5 && temperature <= 39) {
        return {
            status: 'Caution',
            color: 'yellow',
            advisory: 'Temperature slightly elevated. Monitor closely.',
            action: 'Check for signs of stress or mild illness. Ensure adequate water supply.',
        };
    } else if (temperature > 39) {
        return {
            status: 'Critical',
            color: 'red',
            advisory: '🚨 CRITICAL: High temperature detected!',
            action: 'IMMEDIATE ACTION REQUIRED: Contact veterinarian immediately. Animal may have severe infection or heat stress.',
        };
    } else if (temperature < 32) {
        return {
            status: 'Low',
            color: 'blue',
            advisory: 'Low temperature detected. Possible hypothermia or shock.',
            action: 'Check animal immediately. Provide warmth and shelter. Contact veterinarian if temperature continues to drop.',
        };
    }

    return { status: 'Unknown', color: 'gray', advisory: 'Temperature reading abnormal' };
};

export const getTemperatureColor = (temperature) => {
    const status = getTemperatureStatus(temperature);
    const colors = {
        green: '#10b981',
        yellow: '#f59e0b',
        orange: '#f97316',
        red: '#ef4444',
        blue: '#3b82f6',
        gray: '#6b7280',
    };
    return colors[status.color];
};