// backend/utils/stats.js

/**
 * Calculates the mean (average) of an array of numbers.
 */
function avg(arr) {
    if (!arr || arr.length === 0) return 0;
    const sum = arr.reduce((acc, val) => acc + val, 0);
    return sum / arr.length;
}

/**
 * Calculates the standard deviation of an array of numbers.
 */
function std(arr) {
    if (!arr || arr.length === 0) return 0;
    const mean = avg(arr);
    const squareDiffs = arr.map(value => {
        const diff = value - mean;
        return diff * diff;
    });
    const avgSquareDiff = avg(squareDiffs);
    return Math.sqrt(avgSquareDiff);
}

/**
 * Calculates the slope of a linear regression line over an array's indices.
 * Assumes x values are [0, 1, 2, ..., arr.length - 1]
 */
function slope(arr) {
    if (!arr || arr.length < 2) return 0;
    const n = arr.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumXX = 0;

    for (let i = 0; i < n; i++) {
        sumX += i;
        sumY += arr[i];
        sumXY += (i * arr[i]);
        sumXX += (i * i);
    }

    const denominator = (n * sumXX) - (sumX * sumX);
    if (denominator === 0) return 0;

    return ((n * sumXY) - (sumX * sumY)) / denominator;
}

module.exports = {
    avg,
    std,
    slope
};
