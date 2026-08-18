/**
 * Geocoding Service
 * Abstracted geocoding logic to easily swap providers (Nominatim, Google Maps, Mapbox, etc.)
 */

// Simple delay function for rate-limiting
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
let lastRequestTime = 0;

/**
 * Geocode a full address string to coordinates.
 * @param {string} address - The complete address string to geocode
 * @returns {Promise<{lat: number, lng: number, approximate: boolean, error?: string}>}
 */
export const geocodeAddress = async (address, fallbackAddress = '') => {
    if (!address || address.trim() === '') {
        return { error: 'Address is empty' };
    }

    const performQuery = async (queryAddress, forceApproximate = false) => {
        // Rate Limiting
        const now = Date.now();
        const timeSinceLastRequest = now - lastRequestTime;
        if (timeSinceLastRequest < 1500) {
            await delay(1500 - timeSinceLastRequest);
        }
        lastRequestTime = Date.now();

        try {
            const query = encodeURIComponent(queryAddress);
            const url = `https://nominatim.openstreetmap.org/search?format=json&q=${query}&addressdetails=1&limit=3&email=varshananand31@gmail.com`;
            
            const response = await fetch(url, { headers: { 'Accept-Language': 'en' } });
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const data = await response.json();
            if (data && data.length > 0) {
                const bestMatch = data[0];
                const type = bestMatch.type || bestMatch.class;
                const approximate = forceApproximate || ['city', 'town', 'village', 'county', 'state', 'administrative'].includes(type);

                return {
                    lat: parseFloat(bestMatch.lat),
                    lng: parseFloat(bestMatch.lon),
                    approximate: approximate,
                    displayName: bestMatch.display_name
                };
            }
            return null; // No results
        } catch (error) {
            console.error("Geocoding Error:", error);
            return null; // Network error
        }
    };

    // 1. Try primary full address
    let result = await performQuery(address, false);
    if (result) return result;

    // 2. Try fallback address if provided (e.g. City, State, Country)
    if (fallbackAddress && fallbackAddress.trim() !== '') {
        result = await performQuery(fallbackAddress, true); // Fallbacks are inherently approximate
        if (result) return result;
    }

    // 3. Complete failure - return a generic coordinate so they can drag the map
    return { 
        lat: 20.5937, 
        lng: 78.9629, 
        approximate: true, 
        error: 'No precise results found for this address. Defaulting to center of region.' 
    };
};
