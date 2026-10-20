/**
 * Celestial Ephemeris & 24-Hour Daylight Cycle.
 *
 * Computes solar elevation, azimuth, color temperature, and ambient light
 * for the 3D Research Island based on either real-world local time or manual scrubbing.
 */

export interface CelestialState {
	hour: number;
	period: "dawn" | "noon" | "sunset" | "night";
	labelZh: string;
	labelEn: string;
	sunPosition: [number, number, number];
	sunColor: number;
	sunIntensity: number;
	hemiSkyColor: number;
	hemiGroundColor: number;
	fillColor: number;
	fillIntensity: number;
	stageBackgroundCss: string;
}

export function getCurrentSolarHour(): number {
	if (typeof window === "undefined") return 14;
	const now = new Date();
	return now.getHours() + now.getMinutes() / 60;
}

export function computeCelestialState(hour: number): CelestialState {
	// Wrap hour into [0, 24)
	const h = ((hour % 24) + 24) % 24;

	let period: "dawn" | "noon" | "sunset" | "night";
	let labelZh = "";
	let labelEn = "";

	if (h >= 5 && h < 8.5) {
		period = "dawn";
		labelZh = "晨曦薄雾 (Dawn)";
		labelEn = "Dawn Mist";
	} else if (h >= 8.5 && h < 16.5) {
		period = "noon";
		labelZh = "明朗正午 (Noon)";
		labelEn = "Bright Daylight";
	} else if (h >= 16.5 && h < 19.5) {
		period = "sunset";
		labelZh = "金桃晚霞 (Sunset)";
		labelEn = "Golden Sunset";
	} else {
		period = "night";
		labelZh = "静夜星辰 (Midnight)";
		labelEn = "Starry Night";
	}

	// Solar elevation: peak at noon (12:00), lowest at midnight (00:00)
	// Solar angle: 0 at 6:00, PI/2 at 12:00, PI at 18:00, 3PI/2 at 24:00
	const solarAngleRad = ((h - 6) / 24) * 2 * Math.PI;
	const elevation = Math.sin(solarAngleRad);
	const azimuth = Math.cos(solarAngleRad);

	const sunX = azimuth * 18;
	const sunY = Math.max(2, elevation * 22 + 4);
	const sunZ = 12 + Math.sin(solarAngleRad * 0.5) * 6;

	let sunColor = 0xffe4ba;
	let sunIntensity = 2.15;
	let hemiSkyColor = 0xfff5e1;
	let hemiGroundColor = 0x849181;
	let fillColor = 0xc5dfd3;
	let fillIntensity = 1.0;
	let stageBackgroundCss =
		"radial-gradient(ellipse at 76% 20%, var(--brand-peach), transparent 48%), var(--brand-surface)";

	if (period === "dawn") {
		sunColor = 0xffc49b;
		sunIntensity = 1.6;
		hemiSkyColor = 0xfceee3;
		hemiGroundColor = 0x768779;
		fillColor = 0xe8cfc4;
		fillIntensity = 0.85;
		stageBackgroundCss =
			"radial-gradient(ellipse at 70% 35%, rgba(243, 199, 172, 0.65), transparent 52%), var(--brand-surface)";
	} else if (period === "noon") {
		sunColor = 0xfff3da;
		sunIntensity = 2.25;
		hemiSkyColor = 0xfff8ea;
		hemiGroundColor = 0x8da18f;
		fillColor = 0xc5dfd3;
		fillIntensity = 1.05;
		stageBackgroundCss =
			"radial-gradient(ellipse at 76% 20%, var(--brand-peach), transparent 48%), var(--brand-surface)";
	} else if (period === "sunset") {
		sunColor = 0xf59868;
		sunIntensity = 1.85;
		hemiSkyColor = 0xf7d9c4;
		hemiGroundColor = 0x614f44;
		fillColor = 0xd97e55;
		fillIntensity = 0.95;
		stageBackgroundCss =
			"radial-gradient(ellipse at 80% 40%, rgba(220, 120, 75, 0.45), transparent 58%), #ebd9cb";
	} else {
		// Midnight
		sunColor = 0x6e8898;
		sunIntensity = 0.55;
		hemiSkyColor = 0x2e4057;
		hemiGroundColor = 0x1b2836;
		fillColor = 0x3f5e78;
		fillIntensity = 0.45;
		stageBackgroundCss =
			"radial-gradient(ellipse at 75% 25%, rgba(70, 95, 115, 0.45), transparent 55%), #233430";
	}

	return {
		hour: h,
		period,
		labelZh,
		labelEn,
		sunPosition: [sunX, sunY, sunZ],
		sunColor,
		sunIntensity,
		hemiSkyColor,
		hemiGroundColor,
		fillColor,
		fillIntensity,
		stageBackgroundCss,
	};
}
