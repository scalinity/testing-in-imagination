// Every time on screen is US Eastern, with the zone named.

const TZ = "America/New_York";
const fClock = new Intl.DateTimeFormat("en-US", { timeZone: TZ, hour: "numeric", minute: "2-digit", timeZoneName: "short" });
const fClockS = new Intl.DateTimeFormat("en-US", { timeZone: TZ, hour: "numeric", minute: "2-digit", second: "2-digit", timeZoneName: "short" });
const fDay = new Intl.DateTimeFormat("en-US", { timeZone: TZ, weekday: "short", month: "short", day: "numeric" });
const fShortDay = new Intl.DateTimeFormat("en-US", { timeZone: TZ, month: "short", day: "numeric" });
const fHour = new Intl.DateTimeFormat("en-US", { timeZone: TZ, hour: "numeric" });

/** "3:15 AM EST" */
export const clock = (ms: number) => fClock.format(ms);
/** "3:15:02 AM EST" */
export const clockS = (ms: number) => fClockS.format(ms);
/** "Fri, Nov 13" */
export const day = (ms: number) => fDay.format(ms);
/** "Nov 13" */
export const shortDay = (ms: number) => fShortDay.format(ms);
/** "Fri, Nov 13, 3:15 AM EST" */
export const dayClock = (ms: number) => `${fDay.format(ms)}, ${fClock.format(ms)}`;
/** "3 AM" */
export const hour = (ms: number) => fHour.format(ms);
/** "EST" or "EDT": the zone name in force at that moment, for axes whose ticks omit it. */
export const zone = (ms: number) => fClock.format(ms).split(" ").pop() ?? "";

/** Milliseconds from an ISO 8601 string, or null. */
export function parseTime(s: string | null | undefined): number | null {
  if (!s) return null;
  const ms = Date.parse(s);
  return Number.isFinite(ms) ? ms : null;
}

/** "4 s", "3 min", "2 h 10 min", "3 d 4 h" */
export function span(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  if (h < 48) return m % 60 ? `${h} h ${m % 60} min` : `${h} h`;
  const d = Math.floor(h / 24);
  return h % 24 ? `${d} d ${h % 24} h` : `${d} d`;
}

export const ago = (then: number | null | undefined, now: number) => (then == null ? "never" : `${span(now - then)} ago`);

/** "3:15 AM EST (4 min ago)" */
export const stamp = (then: number | null | undefined, now: number) => (then == null ? "never" : `${dayClock(then)} (${span(now - then)} ago)`);

/**
 * Midnight US Eastern for a date between November 2026 and January 2027, when Eastern is EST
 * (UTC-5) throughout. Month is 1-based.
 */
export const easternMidnight = (year: number, month: number, dayOfMonth: number) => Date.UTC(year, month - 1, dayOfMonth, 5);

export const fmt = (n: number, digits = 0) => n.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: 0 });

/** Keeps the informative end of a long path. */
export function tailPath(p: string, max = 46): string {
  if (p.length <= max) return p;
  const parts = p.split("/");
  let out = parts.pop() ?? "";
  while (parts.length && out.length + parts[parts.length - 1].length + 1 <= max - 2) out = `${parts.pop()}/${out}`;
  return `…/${out}`;
}
