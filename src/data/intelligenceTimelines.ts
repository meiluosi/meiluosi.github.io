import data from "./intelligenceTimeline.generated.json";
export type TimelineRoute = (typeof data.routes)[number];
export type TimelineNode = TimelineRoute["nodes"][number];
export type TimelineRelation = TimelineRoute["relations"][number];
export const timelineRoutes: TimelineRoute[] = data.routes;
export const timelineRoundSix: typeof data.roundSix = data.roundSix;
export const timelineScope: string = data.scope;
export const timelineCounts: typeof data.historicalCounts =
	data.historicalCounts;
export function timelineRoute(id: string): TimelineRoute | undefined {
	return timelineRoutes.find((route) => route.id === id);
}
