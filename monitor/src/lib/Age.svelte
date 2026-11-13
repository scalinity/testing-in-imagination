<script lang="ts">
  import Glyph from "./Glyph.svelte";
  import { store } from "./store.svelte";
  import { dayClock, span } from "./time";
  import { tip } from "./tip.svelte";

  // The small live age every panel carries. It ticks every second and turns amber, with a
  // clock, once the source is older than its expected cadence allows.
  let {
    at,
    staleAfter,
    verb = "updated",
    what,
    cadence,
  }: {
    /** When the source was last read, on the monitor's clock. */
    at: number | null | undefined;
    /** Age in ms beyond which the reading counts as stale. */
    staleAfter: number;
    verb?: string;
    /** What was read, for the hover. */
    what: string;
    /** How often it is read, in words. */
    cadence: string;
  } = $props();

  const age = $derived(at == null ? null : Math.max(0, store.now - at));
  const stale = $derived(age == null || age > staleAfter);
</script>

<span
  class="age num"
  class:stale
  use:tip={() => ({
    title: stale ? "This reading is stale" : "This reading is current",
    body: stale
      ? `${what} should be read ${cadence}, and the last successful read is older than that. What is drawn is the last known state.`
      : `${what} is read ${cadence}.`,
    rows: [["Last read", at == null ? "never" : `${dayClock(at)}, ${span(store.now - at)} ago`]],
  })}
>
  {#if stale}<Glyph name="clock" size={11} />{/if}
  {verb}
  {age == null ? "never" : `${span(age)} ago`}
</span>

<style>
  .age {
    font-size: var(--fs-1);
    color: var(--ink-3);
    white-space: nowrap;
  }
  .stale {
    color: var(--wait-text);
  }
</style>
