/**
 * Code Cinema (代码电影) 核心类型契约。
 *
 * 设计原则：场景是「时间的纯函数」——给定全局时间 t，画面完全确定，
 * 因此时间轴可以任意拖动、回放、倍速，而不需要保存动画状态。
 */

export interface CheckpointParam {
	id: string;
	labelZh: string;
	labelEn: string;
	min: number;
	max: number;
	step: number;
	defaultVal: number;
}

export interface CinemaCheckpoint {
	id: string;
	/** 相对于本幕开始的秒数；播放越过该点时自动暂停并展开交互面板。 */
	pauseAtSec: number;
	promptZh: string;
	promptEn: string;
	guidanceZh: string;
	guidanceEn: string;
	params: CheckpointParam[];
}

export interface CinemaSubtitle {
	/** 全局时间范围 [startSec, endSec) */
	timeRange: [number, number];
	speakerZh?: string;
	speakerEn?: string;
	textZh: string;
	textEn: string;
	tagZh?: string;
	tagEn?: string;
}

export interface CinemaAct {
	id: string;
	actNumber: string; // e.g. "ACT I"
	titleZh: string;
	titleEn: string;
	subtitleZh: string;
	subtitleEn: string;
	startSec: number;
	durationSec: number;
	checkpoint?: CinemaCheckpoint;
	subtitles: CinemaSubtitle[];
}

export interface CinemaScript {
	id: string;
	titleZh: string;
	titleEn: string;
	subtitleZh: string;
	subtitleEn: string;
	totalDurationSec: number;
	acts: CinemaAct[];
}
