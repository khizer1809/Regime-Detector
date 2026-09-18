// Semantic color system (Section 10): candle color = price direction only,
// this module governs ONLY the regime background bands / legend / markers.
// Colors are derived from substrings so any actual project regime name
// (Uptrend/Downtrend/Ranging x Vol level, whatever the real labeling
// produces) maps sensibly without hard-coding an exact label list.

// Bold 3-way scheme (trend direction only -- vol level no longer affects
// band color, per explicit request): light green = Uptrend, light pink =
// Downtrend, yellow = Ranging (any vol level). Opacity raised well above the
// old subtle 0.08-0.12 so a state's extent is clearly visible at a glance,
// like manually drawing a highlight rectangle over it.
const PALETTE: { match: RegExp; band: string; solid: string }[] = [
  { match: /uptrend/i, band: "rgba(144, 238, 144, 0.28)", solid: "#90ee90" },
  { match: /downtrend/i, band: "rgba(255, 182, 193, 0.28)", solid: "#ffb6c1" },
  { match: /ranging|range/i, band: "rgba(255, 235, 59, 0.24)", solid: "#ffeb3b" },
];

const FALLBACK = { band: "rgba(120, 130, 140, 0.08)", solid: "#78828c" };

export function regimeBandColor(regime: string | null | undefined): string {
  if (!regime) return "transparent";
  const hit = PALETTE.find((p) => p.match.test(regime));
  return hit ? hit.band : FALLBACK.band;
}

export function regimeSolidColor(regime: string | null | undefined): string {
  if (!regime) return FALLBACK.solid;
  const hit = PALETTE.find((p) => p.match.test(regime));
  return hit ? hit.solid : FALLBACK.solid;
}
